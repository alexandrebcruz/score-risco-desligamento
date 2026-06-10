"""Treino CatBoost FULL-DATA (sem amostragem) — roda no Pod RunPod.

Standalone. Lê /workspace/data/rais/ano=*/*.parquet, monta features (com
prefixos hierárquicos de CBO/CNAE) e treina o CatBoost em TODOS os vínculos.

Split TEMPORAL (sem vazamento):
  - fit      = anos de treino, exceto o último   (ex.: 2020, 2021)
  - validação= último ano de treino             (ex.: 2022)  -> EARLY STOPPING
  - holdout  = HOLDOUT_ANO                       (ex.: 2023)  -> AVALIAÇÃO FINAL

Redução de cardinalidade: categorias com frequência < MIN_FREQ no FIT são
colapsadas em "Demais" (vocabulário definido SÓ no fit e aplicado a val/holdout
— sem vazamento). Reduz o custo de CTR do CatBoost -> treino bem mais rápido.

Uso: nohup python train_full.py > /workspace/train.log 2>&1 &
"""
import json, time, glob, os, gc
import numpy as np
import pandas as pd

DATA = "/workspace/data/rais"
OUT = "/workspace/artifacts"
os.makedirs(OUT, exist_ok=True)

# novas features (existem em todos os anos 2020-2023)
EXTRA_CAT = ["tipo_vinculo", "faixa_remuneracao", "natureza_juridica", "natureza_setor",
             "intermitente", "simples", "faixa_horas", "causa_afastamento"]
RAW = (["cbo", "cnae", "uf", "escolaridade", "tamanho_estab",
        "idade", "tempo_vinculo_meses", "qtd_dias_afastamento", "motivo_unificado"]
       + EXTRA_CAT)
CAT = (["cbo", "cbo4", "cbo2", "cbo1", "cnae", "cnae5", "cnae3", "cnae2",
        "uf", "escolaridade", "tamanho_estab"] + EXTRA_CAT)
NUM = ["idade", "tempo_vinculo_meses", "qtd_dias_afastamento"]
FEATURES = CAT + NUM
TARGET_MOTIVO = "involuntario_sjc"
HOLDOUT_ANO = 2023
EARLY_STOPPING_ROUNDS = 50
MAX_ITERATIONS = 3000
TOPK = 5000             # teto alto: na GPU não precisa cortar (cbo máx ~2657) -> cardinalidade ~completa
RARE = "Demais"
TASK_TYPE = "GPU"       # GPU acelera o cálculo de CTR ordens de magnitude

t0 = time.time()
def _rss_gb():
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("VmRSS"): return int(ln.split()[1]) / 1e6
    except Exception:
        return -1
def log(m): print(f"[{time.time()-t0:7.0f}s | RAM {_rss_gb():.1f}GB] {m}", flush=True)

def load(anos):
    parts = []
    for a in anos:
        for f in sorted(glob.glob(f"{DATA}/ano={a}/*.parquet")):
            d = pd.read_parquet(f, columns=RAW)
            cbo = d["cbo"].astype(str).str.zfill(6)
            cnae = d["cnae"].astype(str).str.zfill(7)
            d["cbo4"], d["cbo2"], d["cbo1"] = cbo.str[:4], cbo.str[:2], cbo.str[:1]
            d["cnae5"], d["cnae3"], d["cnae2"] = cnae.str[:5], cnae.str[:3], cnae.str[:2]
            d["y"] = (d["motivo_unificado"] == TARGET_MOTIVO).astype("int8")
            for c in CAT:
                d[c] = d[c].astype(str)            # mantém string p/ collapse depois
            for c in NUM:
                d[c] = pd.to_numeric(d[c], errors="coerce").fillna(-1).astype("float32")
            parts.append(d[FEATURES + ["y"]])
            log(f"  lido ano={a}/{os.path.basename(f)} ({len(d):,})")
    out = pd.concat(parts, ignore_index=True); del parts; gc.collect()
    return out

def build_vocab(df, topk):
    """Para cada cat feature, as TOPK categorias mais frequentes (no fit)."""
    vocab = {}
    for c in CAT:
        vc = df[c].value_counts()
        keep = vc.head(topk).index.tolist()
        vocab[c] = keep
        log(f"  vocab[{c}]: {len(vc):,} -> {len(keep):,} cats (resto -> '{RARE}')")
    return vocab

def apply_collapse(df, vocab):
    """Substitui categorias fora do vocab por RARE e converte p/ category."""
    for c in CAT:
        keep = set(vocab[c])
        s = df[c].astype(str)
        df[c] = np.where(s.isin(keep), s, RARE)
        df[c] = df[c].astype("category")
    return df

anos = sorted(int(p.split("=")[1]) for p in glob.glob(f"{DATA}/ano=*"))
treino_anos = [a for a in anos if a != HOLDOUT_ANO]
fit_anos, val_ano = treino_anos[:-1], treino_anos[-1]
log(f"anos={anos} | fit={fit_anos} | validação={val_ano} | holdout={HOLDOUT_ANO} | TOPK={TOPK}")

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
cat_idx = [FEATURES.index(c) for c in CAT]

log("carregando FIT..."); fit = load(fit_anos)
n_fit, taxa_fit = len(fit), float(fit["y"].mean())
log(f"fit={n_fit:,} taxa_y={taxa_fit:.4f} -> construindo vocabulário (collapse raras)")
vocab = build_vocab(fit, TOPK)
json.dump(vocab, open(f"{OUT}/vocab.json", "w"))     # p/ reaplicar o modelo depois
fit = apply_collapse(fit, vocab)
pool_fit = Pool(fit[FEATURES], fit["y"], cat_features=cat_idx)
del fit; gc.collect()

log("carregando VALIDAÇÃO..."); val = apply_collapse(load([val_ano]), vocab)
n_val, taxa_val = len(val), float(val["y"].mean())
pool_val = Pool(val[FEATURES], val["y"], cat_features=cat_idx)
del val; gc.collect()
log(f"val={n_val:,} taxa_y={taxa_val:.4f} -> Pools prontos")

log(f"treinando CatBoost (até {MAX_ITERATIONS} iters, early stopping={EARLY_STOPPING_ROUNDS} na val {val_ano})...")
model = CatBoostClassifier(
    iterations=MAX_ITERATIONS, depth=8, learning_rate=0.08,
    loss_function="Logloss", eval_metric="AUC",
    early_stopping_rounds=EARLY_STOPPING_ROUNDS, use_best_model=True, od_type="Iter",
    task_type=TASK_TYPE, devices="0",               # treino em GPU
    boosting_type="Plain", max_ctr_complexity=1, max_bin=128,
    bootstrap_type="Bernoulli", subsample=0.7, random_seed=42,
    verbose=1, metric_period=1)                     # log iteração a iteração
model.fit(pool_fit, eval_set=pool_val)
best_it = model.get_best_iteration()
del pool_fit, pool_val; gc.collect()
log(f"treino concluído | best_iteration={best_it} (de até {MAX_ITERATIONS})")

# ── Artefatos ──────────────────────────────────────────────────────────
model.save_model(f"{OUT}/catboost_full.cbm")
model.save_model(f"{OUT}/catboost_full.json", format="json")

log("carregando HOLDOUT (pós-treino)..."); te = apply_collapse(load([HOLDOUT_ANO]), vocab)
log(f"holdout={len(te):,} taxa_y={te['y'].mean():.4f}")
p_hold = np.clip(model.predict_proba(te[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
best_val_auc = model.get_best_score().get("validation", {}).get("AUC")
metrics = {"n_fit": n_fit, "n_val": n_val, "n_holdout": int(len(te)),
           "taxa_y_fit": taxa_fit, "taxa_y_val": taxa_val,
           "taxa_y_holdout": float(te["y"].mean()),
           "best_iteration": int(best_it), "max_iterations": MAX_ITERATIONS,
           "early_stopping_rounds": EARLY_STOPPING_ROUNDS, "topk": TOPK,
           "AUC_val_bestmodel": (float(best_val_auc) if best_val_auc is not None else None),
           "fit_anos": fit_anos, "val_ano": val_ano, "holdout_ano": HOLDOUT_ANO,
           "AUC_holdout": float(roc_auc_score(te["y"], p_hold)),
           "Brier_holdout": float(brier_score_loss(te["y"], p_hold)),
           "LogLoss_holdout": float(log_loss(te["y"], p_hold)),
           "segundos": round(time.time() - t0, 1)}
json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=2)
log(f"METRICS: {json.dumps(metrics)}")

imp = pd.DataFrame({"feature": FEATURES,
                    "importancia": model.get_feature_importance()}
                   ).sort_values("importancia", ascending=False)
imp.to_csv(f"{OUT}/importancia.csv", index=False)
log(f"importância:\n{imp.to_string(index=False)}")

df = pd.DataFrame({"y": te["y"].values, "p": p_hold})
df["bin"] = pd.qcut(df["p"], 10, duplicates="drop")
df.groupby("bin", observed=True).agg(prevista=("p", "mean"), observada=("y", "mean"),
    n=("y", "size")).reset_index(drop=True).to_csv(f"{OUT}/calibracao.csv", index=False)

te_s = te[FEATURES].head(5000).copy(); te_s["prob_demissao"] = p_hold[:5000]
te_s.to_parquet(f"{OUT}/predicoes_amostra.parquet", index=False)

open(f"{OUT}/DONE", "w").write("ok")
log("FIM — artefatos em /workspace/artifacts/")
