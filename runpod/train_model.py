"""Treino do ensemble base (sem lags) — ROBUSTO, uma FASE por processo.

Uso:
  python train_model.py A      # fit 2019-20, val 2021-22 -> catboost_A.cbm
  python train_model.py B      # fit 2021-22, val 2019-20 -> catboost_B.cbm
  python train_model.py eval   # holdout 2023 + os 2 modelos -> métricas/importância

Cada fase é um processo independente: se uma quebrar, as outras não caem, e
relançar pula o que já tem .cbm. Normalização via mapa de ÚNICOS (rápida).
"""
import json, time, glob, os, gc, sys
os.environ.setdefault("OMP_NUM_THREADS", "8")
import re
import numpy as np
import pandas as pd

DATA = "/workspace/data/rais"
OUT = "/workspace/artifacts"
os.makedirs(OUT, exist_ok=True)

EXTRA_CAT = ["tipo_vinculo", "faixa_remuneracao", "natureza_juridica", "natureza_setor",
             "intermitente", "simples", "faixa_horas", "causa_afastamento"]
RAW = (["cbo", "cnae", "uf", "escolaridade", "tamanho_estab",
        "idade", "tempo_vinculo_meses", "qtd_dias_afastamento", "motivo_unificado"] + EXTRA_CAT)
CAT = (["cbo", "cbo4", "cbo2", "cbo1", "cnae", "cnae5", "cnae3", "cnae2",
        "uf", "escolaridade", "tamanho_estab"] + EXTRA_CAT)
NUM = ["idade", "tempo_vinculo_meses", "qtd_dias_afastamento"]
FEATURES = CAT + NUM
TARGET_MOTIVO = "involuntario_sjc"
HOLDOUT_ANO = 2023
EARLY = 50
MAX_ITER = 3000
CODIGOS_STRIP = ["faixa_remuneracao", "faixa_horas", "causa_afastamento"]
CODE_REMAP = {"causa_afastamento": {"999": "99"}}
_RX = re.compile(r"^0+(?=\d)")

t0 = time.time()
def _rss():
    for ln in open("/proc/self/status"):
        if ln.startswith("VmRSS"): return int(ln.split()[1]) / 1e6
    return -1
def log(m): print(f"[{time.time()-t0:7.0f}s | RAM {_rss():.1f}GB] {m}", flush=True)

def _fast_map(s, func):
    u = s.astype(str).unique()
    return s.astype(str).map({v: func(v) for v in u})
def _norm_strip(v): return _RX.sub("", v) if re.fullmatch(r"0*\d+", v) else v

def normalize(d):
    for c, mapa in CODE_REMAP.items():
        if c in d.columns: d[c] = _fast_map(d[c], lambda v: mapa.get(v, v))
    for c in CODIGOS_STRIP:
        if c in d.columns: d[c] = _fast_map(d[c], _norm_strip)
    d["cbo"] = _fast_map(d["cbo"], lambda v: v.zfill(6))
    d["cnae"] = _fast_map(d["cnae"], lambda v: v.zfill(7))
    cbo, cnae = d["cbo"], d["cnae"]
    d["cbo4"], d["cbo2"], d["cbo1"] = cbo.str[:4], cbo.str[:2], cbo.str[:1]
    d["cnae5"], d["cnae3"], d["cnae2"] = cnae.str[:5], cnae.str[:3], cnae.str[:2]
    return d

def load(anos):
    parts = []
    for a in anos:
        for f in sorted(glob.glob(f"{DATA}/ano={a}/*.parquet")):
            d = pd.read_parquet(f, columns=RAW)
            d = normalize(d)
            d["y"] = (d["motivo_unificado"] == TARGET_MOTIVO).astype("int8")
            for c in CAT: d[c] = d[c].astype(str).astype("category")
            for c in NUM: d[c] = pd.to_numeric(d[c], errors="coerce").fillna(-1).astype("float32")
            parts.append(d[FEATURES + ["y"]])
            log(f"  lido ano={a}/{os.path.basename(f)} ({len(d):,})")
    out = pd.concat(parts, ignore_index=True); del parts; gc.collect()
    return out

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
cat_idx = [FEATURES.index(c) for c in CAT]
PARAMS = dict(iterations=MAX_ITER, depth=8, learning_rate=0.08,
              loss_function="Logloss", eval_metric="Logloss",
              early_stopping_rounds=EARLY, use_best_model=True, od_type="Iter",
              task_type="GPU", devices="0", boosting_type="Plain", max_ctr_complexity=1,
              max_bin=128, bootstrap_type="Bernoulli", subsample=0.7, random_seed=42, verbose=50)

PHASES = {"A": ([2019, 2020], [2021, 2022]), "B": ([2021, 2022], [2019, 2020])}

def train(nome):
    cbm = f"{OUT}/catboost_{nome}.cbm"
    if os.path.exists(cbm):
        log(f"{nome}: já existe {cbm} — pulando"); return
    fit_anos, val_anos = PHASES[nome]
    log(f"=== Modelo {nome}: fit={fit_anos} val={val_anos} ===")
    dfit = load(fit_anos); p_fit = Pool(dfit[FEATURES], dfit["y"], cat_features=cat_idx)
    del dfit; gc.collect(); log("pool fit pronto")
    dval = load(val_anos); p_val = Pool(dval[FEATURES], dval["y"], cat_features=cat_idx)
    del dval; gc.collect(); log("pool val pronto — iniciando fit")
    m = CatBoostClassifier(**PARAMS)
    m.fit(p_fit, eval_set=p_val)
    bi = int(m.get_best_iteration())
    m.save_model(cbm)
    json.dump({"best_iteration": bi}, open(f"{OUT}/meta_{nome}.json", "w"))
    log(f"{nome} OK | best_iteration={bi} | salvo {cbm}")

def evaluate():
    mA = CatBoostClassifier(); mA.load_model(f"{OUT}/catboost_A.cbm")
    mB = CatBoostClassifier(); mB.load_model(f"{OUT}/catboost_B.cbm")
    te = load([HOLDOUT_ANO]); Xte = te[FEATURES]; yte = te["y"].values
    pA = mA.predict_proba(Xte)[:, 1]; pB = mB.predict_proba(Xte)[:, 1]; pE = (pA + pB) / 2
    def metr(y, p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return {"AUC": float(roc_auc_score(y, p)), "Brier": float(brier_score_loss(y, p)),
                "LogLoss": float(log_loss(y, p))}
    biA = json.load(open(f"{OUT}/meta_A.json"))["best_iteration"]
    biB = json.load(open(f"{OUT}/meta_B.json"))["best_iteration"]
    res = {"modelo_A": metr(yte, pA), "modelo_B": metr(yte, pB), "ensemble": metr(yte, pE),
           "best_iter_A": biA, "best_iter_B": biB, "holdout": HOLDOUT_ANO,
           "n_holdout": int(len(te)), "fit_A": PHASES["A"][0], "fit_B": PHASES["B"][0],
           "metrica": "Logloss (treino + early stopping)", "segundos": round(time.time() - t0, 1)}
    json.dump(res, open(f"{OUT}/metrics_ensemble.json", "w"), indent=2)
    log("RESULTADOS:\n" + json.dumps(res, indent=2))
    impA = np.array(mA.get_feature_importance()); impB = np.array(mB.get_feature_importance())
    pd.DataFrame({"feature": FEATURES, "imp_A": impA, "imp_B": impB,
                  "imp_ensemble": (impA + impB) / 2}).sort_values("imp_ensemble", ascending=False)\
      .to_csv(f"{OUT}/importancia_ensemble.csv", index=False)
    dfc = pd.DataFrame({"y": yte, "p": np.clip(pE, 1e-6, 1 - 1e-6)})
    dfc["bin"] = pd.qcut(dfc["p"], 10, duplicates="drop")
    dfc.groupby("bin", observed=True).agg(prevista=("p", "mean"), observada=("y", "mean"),
        n=("y", "size")).reset_index(drop=True).to_csv(f"{OUT}/calibracao_ensemble.csv", index=False)
    open(f"{OUT}/DONE", "w").write("ok")
    log("FIM eval — métricas/importância/calibração salvas")

if __name__ == "__main__":
    fase = sys.argv[1]
    if fase in PHASES: train(fase)
    elif fase == "eval": evaluate()
    else: raise SystemExit(f"fase inválida: {fase}")
