"""Aplica o MELHOR modelo (ensemble base = média de catboost_A + catboost_B) em
TODA a base de 2023 e salva um parquet com: as 22 features do modelo + alvo (y) +
prob prevista (prob_A, prob_B, prob_desligamento = média).

Pré-processamento idêntico ao treino (src.cleaning.normalize_short_codes + zfill/hier).
Processa partição a partição (baixa memória, resumível) e no fim faz um único parquet.

Uso:  /tmp/consig_venv/bin/python predict_ensemble_base_2023.py
Saída: outputs/predicoes_2023_ensemble_base.parquet  (+ parciais em outputs/predicoes_2023/)
"""
import glob, os, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostClassifier
from src.cleaning import normalize_short_codes

DATA = "data/interim/rais/ano=2023"
MODELS = "outputs/runpod_ensemble_base"
PARTS = "outputs/predicoes_2023"
OUT = "outputs/predicoes_2023_ensemble_base.parquet"
os.makedirs(PARTS, exist_ok=True)

EXTRA_CAT = ["tipo_vinculo", "natureza_juridica", "natureza_setor",
             "intermitente", "simples"]
# Ordinais (a ORDEM do código tem significado) -> NUMÉRICAS; 99=ignorado -> -1.
ORD = ["escolaridade", "tamanho_estab", "faixa_remuneracao", "faixa_horas"]
CAT = (["cbo", "cbo4", "cbo2", "cbo1", "cnae", "cnae5", "cnae3", "cnae2",
        "uf"] + EXTRA_CAT)
NUM = ["idade", "tempo_vinculo_meses", "qtd_dias_afastamento"] + ORD
FEATURES = CAT + NUM
RAW = (["cbo", "cnae", "uf",
        "idade", "tempo_vinculo_meses", "qtd_dias_afastamento", "motivo_unificado"] + ORD + EXTRA_CAT)
TARGET = "involuntario_sjc"

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)

def prep(d):
    """Mesmo pré-processamento do treino (train_model.py)."""
    d = normalize_short_codes(d)                       # remap 999->99 + strip de zeros
    d["cbo"] = d["cbo"].astype(str).str.zfill(6)
    d["cnae"] = d["cnae"].astype(str).str.zfill(7)
    cbo, cnae = d["cbo"], d["cnae"]
    d["cbo4"], d["cbo2"], d["cbo1"] = cbo.str[:4], cbo.str[:2], cbo.str[:1]
    d["cnae5"], d["cnae3"], d["cnae2"] = cnae.str[:5], cnae.str[:3], cnae.str[:2]
    for c in CAT: d[c] = d[c].astype(str)
    for c in NUM: d[c] = pd.to_numeric(d[c], errors="coerce").fillna(-1).astype("float32")
    for c in ("faixa_remuneracao", "faixa_horas"):
        d.loc[d[c] == 99, c] = -1                    # 99 = ignorado -> sentinela
    return d

log("carregando modelos A e B ...")
mA = CatBoostClassifier(); mA.load_model(f"{MODELS}/catboost_A.cbm")
mB = CatBoostClassifier(); mB.load_model(f"{MODELS}/catboost_B.cbm")

BATCH = 3_000_000   # lotes de linhas -> resumível por lote (sobrevive aos kills do sandbox)
files = sorted(glob.glob(f"{DATA}/*.parquet"))
log(f"{len(files)} partições de 2023 a pontuar (lotes de {BATCH:,})")
for fp in files:
    região = os.path.basename(fp).replace(".parquet", "")
    pf = pq.ParquetFile(fp)
    for bi, rb in enumerate(pf.iter_batches(batch_size=BATCH, columns=RAW)):
        dest = f"{PARTS}/{região}__b{bi:02d}.parquet"
        if os.path.exists(dest):
            continue
        d = rb.to_pandas()
        n = len(d)
        ymot = (d["motivo_unificado"] == TARGET).astype("int8")
        d = prep(d)
        X = d[FEATURES]
        pA = mA.predict_proba(X)[:, 1].astype("float32")
        pB = mB.predict_proba(X)[:, 1].astype("float32")
        out = d[FEATURES].copy()
        out["y"] = ymot.values
        out["prob_A"] = pA
        out["prob_B"] = pB
        out["prob_desligamento"] = ((pA + pB) / 2.0).astype("float32")
        out.to_parquet(dest, index=False)
        del d, X, out, rb
        log(f"  {região} lote {bi}: {n:,} -> {os.path.basename(dest)}")

# --- merge final em um único parquet (streaming, baixa memória) ---
log("merge final em um único parquet ...")
writer = None
for fp in sorted(glob.glob(f"{PARTS}/*.parquet")):
    t = pq.read_table(fp)
    if writer is None:
        writer = pq.ParquetWriter(OUT, t.schema, compression="zstd")
    writer.write_table(t)
if writer: writer.close()
tot = pq.ParquetFile(OUT).metadata.num_rows
log(f"FIM — {tot:,} linhas em {OUT}")
