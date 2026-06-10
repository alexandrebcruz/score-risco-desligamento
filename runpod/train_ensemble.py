"""Ensemble de 2 CatBoosts (cross-temporal) — roda no Pod RunPod (GPU).

- Modelo A: fit = 2019+2020, validação (early stopping) = 2021+2022
- Modelo B: fit = 2021+2022, validação (early stopping) = 2019+2020
- Modelo final = média das probabilidades (ensemble) de A e B
- Holdout (teste) = 2023 (nunca visto por nenhum dos dois no fit/early stopping)

Métrica ÚNICA: Logloss — tanto a função de perda (treino) quanto a métrica de
early stopping. Cardinalidade completa (sem top-K). Artefatos em /workspace/artifacts/.

Uso: nohup python train_ensemble.py > /workspace/train.log 2>&1 &
"""
import json, time, glob, os, gc
import numpy as np
import pandas as pd

DATA = "/workspace/data/rais"
OUT = "/workspace/artifacts"
os.makedirs(OUT, exist_ok=True)

EXTRA_CAT = ["tipo_vinculo", "natureza_juridica", "natureza_setor",
             "intermitente", "simples", "causa_afastamento"]
# Ordinais (a ORDEM do código tem significado: escolaridade 1..11, tamanho 1..10,
# faixas 0..11/1..6) -> tratadas como NUMÉRICAS; 99/{ñ class}/ausente -> -1.
ORD = ["escolaridade", "tamanho_estab", "faixa_remuneracao", "faixa_horas"]
RAW = (["cbo", "cnae", "uf",
        "idade", "tempo_vinculo_meses", "qtd_dias_afastamento", "motivo_unificado"] + ORD + EXTRA_CAT)
CAT = (["cbo", "cbo4", "cbo2", "cbo1", "cnae", "cnae5", "cnae3", "cnae2",
        "uf"] + EXTRA_CAT)
NUM = ["idade", "tempo_vinculo_meses", "qtd_dias_afastamento"] + ORD
FEATURES = CAT + NUM
TARGET_MOTIVO = "involuntario_sjc"
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
    parts = []
    for a in anos:
        for f in sorted(glob.glob(f"{DATA}/ano={a}/*.parquet")):
            d = pd.read_parquet(f, columns=RAW)
            # NOTA: remap 999->99, strip de zeros e zfill NÃO são mais necessários — o
            # interim novo (src/cleaning.clean_rais_real) já entrega tudo harmonizado
            # (faixas/escolaridade int64; cbo/cnae zfillados). Só derivamos os níveis.
            cbo, cnae = d["cbo"].astype(str), d["cnae"].astype(str)
            d["cbo4"], d["cbo2"], d["cbo1"] = cbo.str[:4], cbo.str[:2], cbo.str[:1]
            d["cnae5"], d["cnae3"], d["cnae2"] = cnae.str[:5], cnae.str[:3], cnae.str[:2]
            d["y"] = (d["motivo_unificado"] == TARGET_MOTIVO).astype("int8")
            for c in CAT:
                d[c] = d[c].astype(str).astype("category")
            for c in NUM:
                d[c] = pd.to_numeric(d[c], errors="coerce").fillna(-1).astype("float32")
            for c in ("faixa_remuneracao", "faixa_horas"):
                d.loc[d[c] == 99, c] = -1            # 99 = ignorado -> sentinela
            parts.append(d[FEATURES + ["y"]])
            log(f"  lido ano={a}/{os.path.basename(f)} ({len(d):,})")
    out = pd.concat(parts, ignore_index=True); del parts; gc.collect()
    return out

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
cat_idx = [FEATURES.index(c) for c in CAT]

PARAMS = dict(iterations=MAX_ITERATIONS, depth=8, learning_rate=0.08,
              loss_function="Logloss", eval_metric="Logloss",   # <- Logloss p/ treino E stopping
              early_stopping_rounds=EARLY_STOPPING_ROUNDS, use_best_model=True, od_type="Iter",
              task_type="GPU", devices="0",
              boosting_type="Plain", max_ctr_complexity=1, max_bin=128,
              bootstrap_type="Bernoulli", subsample=0.7, random_seed=42, verbose=50)

# Carrega cada ano UMA vez (reutilizado entre fit/val dos dois modelos)
log("carregando 2019,2020,2021,2022 + holdout 2023 ...")
dfs = {a: load([a]) for a in (2019, 2020, 2021, 2022)}
te = load([HOLDOUT_ANO])
log("dados carregados")

def make_pool(anos):
    df = pd.concat([dfs[a] for a in anos], ignore_index=True)
    return Pool(df[FEATURES], df["y"], cat_features=cat_idx)

def treina(nome, fit_anos, val_anos):
    log(f"--- {nome}: fit={fit_anos} val={val_anos} ---")
    p_fit, p_val = make_pool(fit_anos), make_pool(val_anos)
    m = CatBoostClassifier(**PARAMS)
    m.fit(p_fit, eval_set=p_val)
    bi = m.get_best_iteration()
    del p_fit, p_val; gc.collect()
    m.save_model(f"{OUT}/catboost_{nome}.cbm")
    log(f"{nome} ok | best_iteration={bi}")
    return m, bi

modelA, biA = treina("A", [2019, 2020], [2021, 2022])
modelB, biB = treina("B", [2021, 2022], [2019, 2020])

# Predições no holdout 2023
log("avaliando no holdout 2023 ...")
Xte = te[FEATURES]; yte = te["y"].values
pA = modelA.predict_proba(Xte)[:, 1]
pB = modelB.predict_proba(Xte)[:, 1]
pE = (pA + pB) / 2.0

def metr(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"AUC": float(roc_auc_score(y, p)),
            "Brier": float(brier_score_loss(y, p)),
            "LogLoss": float(log_loss(y, p))}

res = {"modelo_A": metr(yte, pA), "modelo_B": metr(yte, pB), "ensemble": metr(yte, pE),
       "best_iter_A": int(biA), "best_iter_B": int(biB),
       "fit_A": [2019, 2020], "val_A": [2021, 2022],
       "fit_B": [2021, 2022], "val_B": [2019, 2020],
       "holdout": HOLDOUT_ANO, "n_holdout": int(len(te)),
       "metrica": "Logloss (treino + early stopping)", "segundos": round(time.time() - t0, 1)}
json.dump(res, open(f"{OUT}/metrics_ensemble.json", "w"), indent=2)
log(f"RESULTADOS: {json.dumps(res, indent=2)}")

# Feature importance: por modelo + ensemble (média das duas).
# get_feature_importance() retorna PredictionValuesChange (cada uma soma 100%);
# a média também soma 100% -> interpretável como importância do ensemble.
impA = np.array(modelA.get_feature_importance())
impB = np.array(modelB.get_feature_importance())
imp = pd.DataFrame({"feature": FEATURES, "imp_A": impA, "imp_B": impB,
                    "imp_ensemble": (impA + impB) / 2}).sort_values("imp_ensemble", ascending=False)
imp.to_csv(f"{OUT}/importancia_ensemble.csv", index=False)
log("importância (ensemble = média A,B):\n" + imp.round(3).to_string(index=False))

# calibração do ensemble + amostra de predições
dfc = pd.DataFrame({"y": yte, "p": np.clip(pE, 1e-6, 1 - 1e-6)})
dfc["bin"] = pd.qcut(dfc["p"], 10, duplicates="drop")
dfc.groupby("bin", observed=True).agg(prevista=("p", "mean"), observada=("y", "mean"),
    n=("y", "size")).reset_index(drop=True).to_csv(f"{OUT}/calibracao_ensemble.csv", index=False)
out = te[FEATURES].head(5000).copy()
out["prob_A"], out["prob_B"], out["prob_ensemble"] = pA[:5000], pB[:5000], pE[:5000]
out.to_parquet(f"{OUT}/predicoes_ensemble.parquet", index=False)

open(f"{OUT}/DONE", "w").write("ok")
log("FIM — artefatos em /workspace/artifacts/")
