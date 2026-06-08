"""Adiciona a coluna `categoria_risco` (1..23) às predições de 2023.

A categoria vem da discretização ótima por ganho de informação (tune_bins_infogain.py),
cujas bordas estão em outputs/tables/binning_infogain_escolhido.csv (coluna prob_max).
categoria_risco = 1 (menor risco) .. 23 (maior risco), atribuída pela faixa de prob.

Faz streaming em lotes (baixa memória) e preserva TODAS as colunas originais.
Saída: outputs/predicoes_2023_ensemble_base_categorizado.parquet
"""
import shutil, time, os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC = "outputs/predicoes_2023_ensemble_base.parquet"
TMP = "/tmp/predicoes_2023_categorizado.parquet"
DEST = "outputs/predicoes_2023_ensemble_base_categorizado.parquet"
BINS = "outputs/tables/binning_infogain_escolhido.csv"

t0 = time.time()
edges = pd.read_csv(BINS)["prob_max"].to_numpy()[:-1]   # 22 bordas internas -> 23 categorias
K = len(edges) + 1
print(f"{K} categorias | {len(edges)} bordas internas", flush=True)

pf = pq.ParquetFile(SRC)
writer = None
tot = 0
for i, batch in enumerate(pf.iter_batches(batch_size=4_000_000)):
    tbl = pa.Table.from_batches([batch])
    prob = batch.column("prob_desligamento").to_numpy(zero_copy_only=False)
    cat = (np.searchsorted(edges, prob, side="right") + 1).astype("int8")   # 1..23
    tbl = tbl.append_column("categoria_risco", pa.array(cat, type=pa.int8()))
    if writer is None:
        writer = pq.ParquetWriter(TMP, tbl.schema, compression="zstd")
    writer.write_table(tbl)
    tot += len(prob)
    print(f"  lote {i}: {tot:,} linhas", flush=True)
writer.close()

n = pq.ParquetFile(TMP).metadata.num_rows
print(f"escrito {n:,} linhas em {time.time()-t0:.0f}s; copiando p/ {DEST} ...", flush=True)
shutil.copy(TMP, DEST)
os.remove(TMP)
print(f"FIM: {os.path.getsize(DEST)/1e6:.0f} MB | colunas: "
      f"{[f.name for f in pq.ParquetFile(DEST).schema]}", flush=True)
