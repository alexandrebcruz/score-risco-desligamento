"""Materializa `categoria_risco` (1..K*) nas predições de TODOS os anos (2016–2025),
usando as bordas de outputs/tables/binning_infogain_escolhido_2124.csv (referência
2021–2024). Atribuição via searchsorted nas bordas inferiores (prob_min das cats 2..K).

In-place por partição (lê -> adiciona coluna -> grava tmp -> substitui), RESUMÍVEL:
pula partições que já têm a coluna `categoria_risco`. /tmp+copy (DrvFs).

Uso: nohup /tmp/consig_venv/bin/python -u add_categoria_risco_2124.py > /tmp/cat2124.log 2>&1 &
"""
import os, glob, shutil, time
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

PRED = "data/processed/predicoes_2124"
BINS = "outputs/tables/binning_infogain_escolhido_2124.csv"

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

det = pd.read_csv(BINS)
K = int(det["categoria"].max())
# bordas inferiores das categorias 2..K: prob >= borda[i] -> categoria i+2
bordas = det["prob_min"].to_numpy("float64")[1:]
log(f"K*={K} | bordas: {np.round(bordas, 4).tolist()}")

for fp in sorted(glob.glob(f"{PRED}/ano=*/*.parquet")):
    sch = pq.ParquetFile(fp).schema_arrow
    if "categoria_risco" in sch.names:
        log(f"{fp}: já tem categoria_risco, pulando"); continue
    d = pd.read_parquet(fp)
    p = d["prob_desligamento"].to_numpy("float64")
    d["categoria_risco"] = (np.searchsorted(bordas, p, side="right") + 1).astype("int16")
    tmp = "/tmp/cat_" + os.path.basename(fp)
    pq.write_table(pa.Table.from_pandas(d, preserve_index=False), tmp, compression="zstd")
    shutil.move(tmp, fp)
    log(f"{fp}: {len(d):,} categorizados (cats {d['categoria_risco'].min()}–{d['categoria_risco'].max()})")

log("FIM categorização")
print("FIM_CAT_2124", flush=True)
