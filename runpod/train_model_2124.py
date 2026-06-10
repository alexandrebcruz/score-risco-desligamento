"""Retreino do ensemble base sobre o INTERIM NOVO (leak-free, 21 features).

Treino cross-temporal DENTRO de 2021–2024:
  - Modelo A: fit 2021+2022, validação (early stopping) 2023+2024
  - Modelo B: fit 2023+2024, validação (early stopping) 2021+2022
  - Ensemble = média(A, B)

Avaliação: TODOS os anos 2016–2025, por ano, com AUC, KS, LogLoss e Brier
(p/ A, B e ensemble) -> /workspace/artifacts/metricas_por_ano.csv.
Anos 2016–2020 e 2025 são totalmente out-of-sample; 2021–2024 participaram de
fit/val (coluna `papel` documenta o uso de cada ano).

Uso (cada fase = processo isolado, robusto a quedas):
  python train_model_2124.py A
  python train_model_2124.py B
  python train_model_2124.py eval
"""
import json, time, glob, os, gc, sys
os.environ.setdefault("OMP_NUM_THREADS", "16")
import numpy as np
import pandas as pd

DATA = "/workspace/data/rais"
OUT = "/workspace/artifacts"
os.makedirs(OUT, exist_ok=True)

EXTRA_CAT = ["tipo_vinculo", "natureza_juridica", "natureza_setor",
             "intermitente", "simples"]
# Ordinais (a ORDEM do código tem significado) -> NUMÉRICAS; 99=ignorado -> -1.
ORD = ["escolaridade", "tamanho_estab", "faixa_remuneracao", "faixa_horas"]
RAW = (["cbo", "cnae", "uf",
        "idade", "tempo_vinculo_meses", "qtd_dias_afastamento", "motivo_unificado"] + ORD + EXTRA_CAT)
CAT = (["cbo", "cbo4", "cbo2", "cbo1", "cnae", "cnae5", "cnae3", "cnae2",
        "uf"] + EXTRA_CAT)
NUM = ["idade", "tempo_vinculo_meses", "qtd_dias_afastamento"] + ORD
FEATURES = CAT + NUM                      # 14 cat + 7 num = 21
TARGET_MOTIVO = "involuntario_sjc"
ANOS_EVAL = list(range(2016, 2026))
EARLY = 50
MAX_ITER = 3000
PHASES = {"A": ([2021, 2022], [2023, 2024]), "B": ([2023, 2024], [2021, 2022])}
PAPEL = {2021: "fitA_valB", 2022: "fitA_valB", 2023: "fitB_valA", 2024: "fitB_valA"}

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
            d = pd.read_parquet(f, columns=RAW)
            # interim novo já harmonizado; só deriva níveis hierárquicos
            cbo, cnae = d["cbo"].astype(str), d["cnae"].astype(str)
            d["cbo4"], d["cbo2"], d["cbo1"] = cbo.str[:4], cbo.str[:2], cbo.str[:1]
            d["cnae5"], d["cnae3"], d["cnae2"] = cnae.str[:5], cnae.str[:3], cnae.str[:2]
            d["y"] = (d["motivo_unificado"] == TARGET_MOTIVO).astype("int8")
            for c in CAT: d[c] = d[c].astype(str).astype("category")
            for c in NUM: d[c] = pd.to_numeric(d[c], errors="coerce").fillna(-1).astype("float32")
            for c in ("faixa_remuneracao", "faixa_horas"):
                d.loc[d[c] == 99, c] = -1            # 99 = ignorado -> sentinela
            parts.append(d[FEATURES + ["y"]])
            log(f"  lido ano={a}/{os.path.basename(f)} ({len(d):,})")
    out = pd.concat(parts, ignore_index=True); del parts; gc.collect()
    return out

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, roc_curve
cat_idx = [FEATURES.index(c) for c in CAT]
PARAMS = dict(iterations=MAX_ITER, depth=8, learning_rate=0.08,
              loss_function="Logloss", eval_metric="Logloss",
              early_stopping_rounds=EARLY, use_best_model=True, od_type="Iter",
              task_type="GPU", devices="0", boosting_type="Plain", max_ctr_complexity=1,
              max_bin=128, bootstrap_type="Bernoulli", subsample=0.7, random_seed=42, verbose=50)

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

def _ks(y, p):
    """KS = max |TPR - FPR| (separação máxima entre as CDFs de bons e maus)."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))

def _metr(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"AUC": float(roc_auc_score(y, p)), "KS": _ks(y, p),
            "LogLoss": float(log_loss(y, p)), "Brier": float(brier_score_loss(y, p))}

def _pred_batches(m, X, bs=10_000_000):
    out = np.empty(len(X), dtype="float64")
    for i in range(0, len(X), bs):
        out[i:i+bs] = m.predict_proba(X.iloc[i:i+bs])[:, 1]
    return out

def evaluate():
    mA = CatBoostClassifier(); mA.load_model(f"{OUT}/catboost_A.cbm")
    mB = CatBoostClassifier(); mB.load_model(f"{OUT}/catboost_B.cbm")
    rows = []
    for ano in ANOS_EVAL:
        log(f"=== avaliando {ano} ===")
        d = load([ano]); X = d[FEATURES]; y = d["y"].values
        pA = _pred_batches(mA, X); pB = _pred_batches(mB, X); pE = (pA + pB) / 2
        for nome, p in (("A", pA), ("B", pB), ("ensemble", pE)):
            met = _metr(y, p)
            rows.append({"ano": ano, "modelo": nome,
                         "papel": PAPEL.get(ano, "out_of_sample"),
                         "n": int(len(y)), "taxa_y": float(y.mean()), **met})
        log(f"  {ano}: n={len(y):,} taxa={y.mean():.4f} | "
            + " ".join(f"{r['modelo']}:AUC={r['AUC']:.4f}/KS={r['KS']:.4f}"
                       for r in rows[-3:]))
        # calibração por decil do ensemble (todas as linhas do ano)
        dfc = pd.DataFrame({"y": y, "p": np.clip(pE, 1e-6, 1 - 1e-6)})
        dfc["bin"] = pd.qcut(dfc["p"], 10, duplicates="drop")
        (dfc.groupby("bin", observed=True)
            .agg(prevista=("p", "mean"), observada=("y", "mean"), n=("y", "size"))
            .reset_index(drop=True)
            .to_csv(f"{OUT}/calibracao_{ano}.csv", index=False))
        del d, X, y, pA, pB, pE, dfc; gc.collect()
        pd.DataFrame(rows).to_csv(f"{OUT}/metricas_por_ano.csv", index=False)  # parcial/resumível
    pd.DataFrame(rows).to_csv(f"{OUT}/metricas_por_ano.csv", index=False)
    # importância (média A,B)
    impA = np.array(mA.get_feature_importance()); impB = np.array(mB.get_feature_importance())
    pd.DataFrame({"feature": FEATURES, "imp_A": impA, "imp_B": impB,
                  "imp_ensemble": (impA + impB) / 2}).sort_values("imp_ensemble", ascending=False)\
      .to_csv(f"{OUT}/importancia_ensemble.csv", index=False)
    biA = json.load(open(f"{OUT}/meta_A.json"))["best_iteration"]
    biB = json.load(open(f"{OUT}/meta_B.json"))["best_iteration"]
    json.dump({"fit_A": PHASES["A"][0], "val_A": PHASES["A"][1],
               "fit_B": PHASES["B"][0], "val_B": PHASES["B"][1],
               "best_iter_A": biA, "best_iter_B": biB,
               "features": FEATURES, "anos_eval": ANOS_EVAL,
               "segundos": round(time.time() - t0, 1)},
              open(f"{OUT}/metrics_ensemble.json", "w"), indent=2)
    open(f"{OUT}/DONE", "w").write("ok")
    log("FIM eval — metricas_por_ano.csv + importância + calibrações salvos")

if __name__ == "__main__":
    fase = sys.argv[1]
    if fase in PHASES: train(fase)
    elif fase == "eval": evaluate()
    else: raise SystemExit(f"fase inválida: {fase}")
