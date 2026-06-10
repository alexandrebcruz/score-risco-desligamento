"""STEP 2 — ensemble (A/B) com lag features, lendo o enriquecido pré-gerado.

Pré-requisito: prep_lags.py já gerou /workspace/data/rais_lags/ (com as 114 lags).
Aqui NÃO há join — só leitura. Memória controlada: 2 Pools reutilizados
(P_2019-20 e P_2021-22; A e B só invertem fit/val) e DataFrame liberado após
cada Pool. Métrica: Logloss (treino + early stopping). Holdout = 2023.

Uso: nohup python train_ensemble_lags.py > /workspace/train.log 2>&1 &
"""
import json, time, glob, os, gc
import numpy as np
import pandas as pd

DATA = "/workspace/data/rais_lags"      # enriquecido (prep_lags.py)
OUT = "/workspace/artifacts"
os.makedirs(OUT, exist_ok=True)

EXTRA_CAT = ["tipo_vinculo", "natureza_juridica", "natureza_setor",
             "intermitente", "simples", "causa_afastamento"]
# Ordinais (a ORDEM do código tem significado) -> NUMÉRICAS; 99=ignorado -> -1.
# (Seguem em LAG_FEATURES como chave de join por VALOR — convertidas após o join.)
ORD = ["escolaridade", "tamanho_estab", "faixa_remuneracao", "faixa_horas"]
BASE_CAT = ["cbo", "cbo4", "cbo2", "cbo1", "cnae", "cnae5", "cnae3", "cnae2",
            "uf"]
CAT = BASE_CAT + EXTRA_CAT
BASE_NUM = ["idade", "tempo_vinculo_meses", "qtd_dias_afastamento"]
LAG_FEATURES = ["cbo", "cbo4", "cbo2", "cbo1", "cnae", "cnae5", "cnae3", "cnae2",
                "uf", "escolaridade", "tamanho_estab", "tipo_vinculo", "faixa_remuneracao",
                "natureza_juridica", "natureza_setor", "intermitente", "simples",
                "faixa_horas", "causa_afastamento"]
LAGS = (1, 2, 3)
LAG_COLS = [f"{f}_{k}_lag{L}" for f in LAG_FEATURES for L in LAGS for k in ("n", "k")]
NUM = BASE_NUM + ORD + LAG_COLS
FEATURES = CAT + NUM
HOLDOUT_ANO = 2023
EARLY_STOPPING_ROUNDS = 50
MAX_ITERATIONS = 3000

t0 = time.time()
def _rss():
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("VmRSS"): return int(ln.split()[1]) / 1e6
    except Exception:
        return -1
def log(m): print(f"[{time.time()-t0:7.0f}s | RAM {_rss():.1f}GB] {m}", flush=True)

def load(anos):
    """Lê o enriquecido (já com lags). Sem joins. Garante dtypes p/ o CatBoost."""
    parts = []
    for a in anos:
        for fp in sorted(glob.glob(f"{DATA}/ano={a}/*.parquet")):
            d = pd.read_parquet(fp)
            parts.append(d)
            log(f"  lido ano={a}/{os.path.basename(fp)} ({len(d):,})")
    out = pd.concat(parts, ignore_index=True); del parts; gc.collect()
    for c in CAT:
        out[c] = out[c].astype(str).astype("category")
    for c in NUM:
        out[c] = out[c].astype("float32")
    return out

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
cat_idx = [FEATURES.index(c) for c in CAT]

PARAMS = dict(iterations=MAX_ITERATIONS, depth=8, learning_rate=0.08,
              loss_function="Logloss", eval_metric="Logloss",
              early_stopping_rounds=EARLY_STOPPING_ROUNDS, use_best_model=True, od_type="Iter",
              task_type="GPU", devices="0",
              boosting_type="Plain", max_ctr_complexity=1, max_bin=128,
              bootstrap_type="Bernoulli", subsample=0.7, random_seed=42, verbose=50)

log(f"{len(FEATURES)} features ({len(CAT)} cat + {len(NUM)} num) | montando 2 Pools...")
df = load([2019, 2020]); P1920 = Pool(df[FEATURES], df["y"], cat_features=cat_idx)
del df; gc.collect(); log("Pool 2019-2020 pronto (df liberado)")
df = load([2021, 2022]); P2122 = Pool(df[FEATURES], df["y"], cat_features=cat_idx)
del df; gc.collect(); log("Pool 2021-2022 pronto (df liberado)")

def treina(nome, p_fit, p_val):
    log(f"--- {nome} ---")
    m = CatBoostClassifier(**PARAMS)
    m.fit(p_fit, eval_set=p_val)
    bi = m.get_best_iteration()
    m.save_model(f"{OUT}/catboost_lags_{nome}.cbm")
    log(f"{nome} ok | best_iteration={bi}")
    return m, bi

modelA, biA = treina("A", P1920, P2122)   # fit 2019-20, val 2021-22
modelB, biB = treina("B", P2122, P1920)   # fit 2021-22, val 2019-20
del P1920, P2122; gc.collect()

log("carregando holdout 2023 ...")
te = load([HOLDOUT_ANO])
Xte = te[FEATURES]; yte = te["y"].values
pA = modelA.predict_proba(Xte)[:, 1]
pB = modelB.predict_proba(Xte)[:, 1]
pE = (pA + pB) / 2.0

def metr(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"AUC": float(roc_auc_score(y, p)), "Brier": float(brier_score_loss(y, p)),
            "LogLoss": float(log_loss(y, p))}

res = {"modelo_A": metr(yte, pA), "modelo_B": metr(yte, pB), "ensemble": metr(yte, pE),
       "best_iter_A": int(biA), "best_iter_B": int(biB),
       "n_features": len(FEATURES), "n_lag_cols": len(LAG_COLS),
       "fit_A": [2019, 2020], "val_A": [2021, 2022], "fit_B": [2021, 2022], "val_B": [2019, 2020],
       "holdout": HOLDOUT_ANO, "n_holdout": int(len(te)),
       "metrica": "Logloss (treino + early stopping)", "segundos": round(time.time() - t0, 1)}
json.dump(res, open(f"{OUT}/metrics_ensemble_lags.json", "w"), indent=2)
log(f"RESULTADOS: {json.dumps(res, indent=2)}")

impA = np.array(modelA.get_feature_importance())
impB = np.array(modelB.get_feature_importance())
imp = pd.DataFrame({"feature": FEATURES, "imp_A": impA, "imp_B": impB,
                    "imp_ensemble": (impA + impB) / 2}).sort_values("imp_ensemble", ascending=False)
imp.to_csv(f"{OUT}/importancia_ensemble_lags.csv", index=False)
log("top 25 importância (ensemble):\n" + imp.head(25).round(3).to_string(index=False))

dfc = pd.DataFrame({"y": yte, "p": np.clip(pE, 1e-6, 1 - 1e-6)})
dfc["bin"] = pd.qcut(dfc["p"], 10, duplicates="drop")
dfc.groupby("bin", observed=True).agg(prevista=("p", "mean"), observada=("y", "mean"),
    n=("y", "size")).reset_index(drop=True).to_csv(f"{OUT}/calibracao_ensemble_lags.csv", index=False)

open(f"{OUT}/DONE", "w").write("ok")
log("FIM — artefatos em /workspace/artifacts/")
