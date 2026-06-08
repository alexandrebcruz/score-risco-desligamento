"""Treino do ensemble COM LAGS — ROBUSTO, uma FASE por processo.
Lê o enriquecido de prep_lags.py (/workspace/data/rais_lags), sem normalize/join.

Uso:
  python train_model_lags.py A      # fit 2019-20, val 2021-22 -> catboost_lags_A.cbm
  python train_model_lags.py B      # fit 2021-22, val 2019-20 -> catboost_lags_B.cbm
  python train_model_lags.py eval   # holdout 2023 + os 2 modelos -> métricas/importância
"""
import json, time, glob, os, gc, sys, faulthandler
faulthandler.enable()   # dump de stack em segfault/abort (crash nativo do CatBoost)
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import pandas as pd

DATA = "/workspace/data/rais_lags"
OUT = "/workspace/artifacts_lags"
os.makedirs(OUT, exist_ok=True)

EXTRA_CAT = ["tipo_vinculo", "faixa_remuneracao", "natureza_juridica", "natureza_setor",
             "intermitente", "simples", "faixa_horas", "causa_afastamento"]
BASE_CAT = ["cbo", "cbo4", "cbo2", "cbo1", "cnae", "cnae5", "cnae3", "cnae2",
            "uf", "escolaridade", "tamanho_estab"]
CAT = BASE_CAT + EXTRA_CAT
BASE_NUM = ["idade", "tempo_vinculo_meses", "qtd_dias_afastamento"]
LAG_FEATURES = ["cbo", "cbo4", "cbo2", "cbo1", "cnae", "cnae5", "cnae3", "cnae2",
                "uf", "escolaridade", "tamanho_estab", "tipo_vinculo", "faixa_remuneracao",
                "natureza_juridica", "natureza_setor", "intermitente", "simples",
                "faixa_horas", "causa_afastamento"]
LAGS = (1, 2, 3)
LAG_COLS = [f"{f}_{k}_lag{L}" for f in LAG_FEATURES for L in LAGS for k in ("n", "k")]
NUM = BASE_NUM + LAG_COLS
FEATURES = CAT + NUM
HOLDOUT_ANO = 2023
EARLY = 50
MAX_ITER = 3000

t0 = time.time()
def _rss():
    for ln in open("/proc/self/status"):
        if ln.startswith("VmRSS"): return int(ln.split()[1]) / 1e6
    return -1
def log(m): print(f"[{time.time()-t0:7.0f}s | RAM {_rss():.1f}GB] {m}", flush=True)

def load(anos):
    parts = []
    for a in anos:
        for f in sorted(glob.glob(f"{DATA}/ano={a}/*.parquet")):
            d = pd.read_parquet(f)
            for c in CAT: d[c] = d[c].astype("str").astype("category")  # category: leve em RAM
            for c in NUM: d[c] = d[c].astype("float32")
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
def _bordas(nome): return f"{OUT}/borders_{nome}.dat"
def _ppath(nome, role): return f"{OUT}/p_{role}_{nome}.quant"

def poolfit(nome):
    """Processo isolado: monta o FIT pool, quantiza, salva (pool+bordas) em disco."""
    fit_anos, _ = PHASES[nome]
    log(f"=== poolfit {nome}: anos={fit_anos} ===")
    d = load(fit_anos); y = d.pop("y").values
    p = Pool(d, y, cat_features=cat_idx); del d, y; gc.collect()
    p.quantize(border_count=128)
    p.save_quantization_borders(_bordas(nome))
    p.save(_ppath(nome, "fit"))
    log(f"poolfit {nome} salvo (quantizado) em disco")

def poolval(nome):
    """Processo isolado: monta o VAL pool com as MESMAS bordas do fit e salva."""
    _, val_anos = PHASES[nome]
    log(f"=== poolval {nome}: anos={val_anos} ===")
    d = load(val_anos); y = d.pop("y").values
    p = Pool(d, y, cat_features=cat_idx); del d, y; gc.collect()
    p.quantize(input_borders=_bordas(nome), border_count=128)
    p.save(_ppath(nome, "val"))
    log(f"poolval {nome} salvo (quantizado) em disco")

def fit(nome):
    """Processo isolado: carrega os 2 pools quantizados (leves) e treina."""
    cbm = f"{OUT}/catboost_lags_{nome}.cbm"
    if os.path.exists(cbm): log(f"{nome}: já existe — pulando"); return
    log(f"=== fit {nome}: carregando pools quantizados do disco ===")
    p_fit = Pool(data="quantized://" + _ppath(nome, "fit"))
    p_val = Pool(data="quantized://" + _ppath(nome, "val"))
    m = CatBoostClassifier(**PARAMS)
    log("iniciando m.fit ...")
    m.fit(p_fit, eval_set=p_val)
    bi = int(m.get_best_iteration())
    m.save_model(cbm)
    json.dump({"best_iteration": bi}, open(f"{OUT}/meta_{nome}.json", "w"))
    # libera disco dos pools desse modelo
    for r in ("fit", "val"):
        try: os.remove(_ppath(nome, r))
        except OSError: pass
    log(f"{nome} OK | best_iteration={bi} | salvo {cbm} (pools removidos do disco)")

def evaluate():
    mA = CatBoostClassifier(); mA.load_model(f"{OUT}/catboost_lags_A.cbm")
    mB = CatBoostClassifier(); mB.load_model(f"{OUT}/catboost_lags_B.cbm")
    te = load([HOLDOUT_ANO]); Xte = te[FEATURES]; yte = te["y"].values
    pA = mA.predict_proba(Xte)[:, 1]; pB = mB.predict_proba(Xte)[:, 1]; pE = (pA + pB) / 2
    def metr(y, p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return {"AUC": float(roc_auc_score(y, p)), "Brier": float(brier_score_loss(y, p)),
                "LogLoss": float(log_loss(y, p))}
    biA = json.load(open(f"{OUT}/meta_A.json"))["best_iteration"]
    biB = json.load(open(f"{OUT}/meta_B.json"))["best_iteration"]
    res = {"modelo_A": metr(yte, pA), "modelo_B": metr(yte, pB), "ensemble": metr(yte, pE),
           "best_iter_A": biA, "best_iter_B": biB, "holdout": HOLDOUT_ANO, "n_holdout": int(len(te)),
           "n_features": len(FEATURES), "n_lag_cols": len(LAG_COLS),
           "metrica": "Logloss (treino + early stopping)", "segundos": round(time.time() - t0, 1)}
    json.dump(res, open(f"{OUT}/metrics_ensemble_lags.json", "w"), indent=2)
    log("RESULTADOS:\n" + json.dumps(res, indent=2))
    impA = np.array(mA.get_feature_importance()); impB = np.array(mB.get_feature_importance())
    pd.DataFrame({"feature": FEATURES, "imp_A": impA, "imp_B": impB,
                  "imp_ensemble": (impA + impB) / 2}).sort_values("imp_ensemble", ascending=False)\
      .to_csv(f"{OUT}/importancia_ensemble_lags.csv", index=False)
    dfc = pd.DataFrame({"y": yte, "p": np.clip(pE, 1e-6, 1 - 1e-6)})
    dfc["bin"] = pd.qcut(dfc["p"], 10, duplicates="drop")
    dfc.groupby("bin", observed=True).agg(prevista=("p", "mean"), observada=("y", "mean"),
        n=("y", "size")).reset_index(drop=True).to_csv(f"{OUT}/calibracao_ensemble_lags.csv", index=False)
    open(f"{OUT}/DONE", "w").write("ok")
    log("FIM eval lags")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "eval":
        evaluate()
    elif cmd == "poolfit": poolfit(sys.argv[2])
    elif cmd == "poolval": poolval(sys.argv[2])
    elif cmd == "fit": fit(sys.argv[2])
    else: raise SystemExit(f"comando inválido: {cmd}")
