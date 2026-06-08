"""Agrega as level tables usando SÓ os anos de treino (2020-2022), para uma
comparação JUSTA com o CatBoost (que também não vê 2023).

Saída: data/processed/rates_treino/  (não sobrescreve as tabelas de produção,
que usam todos os anos para pontuar pessoas de verdade).
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).parent; sys.path.insert(0, str(ROOT))
import pandas as pd, pyarrow.parquet as pq
from src.config import load_config
from src import binning, cells, rates

ANOS_TREINO = [2020, 2021, 2022]          # 2023 fica FORA (holdout)
cfg = load_config()
interim = cfg["abs"]["interim"]; motivos = cfg["motivos"]
out_dir = cfg["abs"]["processed"] / "rates_treino"
niveis = cells.active_levels(cfg)
acc = rates.Accumulator(motivos, levels=niveis)
files = [f for f in sorted((interim / "rais").rglob("*.parquet"))
         if int(f.parent.name.split("=")[1]) in ANOS_TREINO]
print(f"anos treino={ANOS_TREINO} | partições={len(files)}", flush=True)
t0 = time.time()
for f in files:
    n = 0
    for batch in pq.ParquetFile(f).iter_batches(batch_size=3_000_000):
        df = cells.add_cell_keys(binning.add_bins(batch.to_pandas(), cfg))
        acc.add(df); n += len(df)
    print(f"  ok {f.parent.name}/{f.name}: {n:,} ({time.time()-t0:.0f}s)", flush=True)
rates.save_level_tables(acc.tables(), motivos, out_dir)
g = pd.read_parquet(out_dir / "level_global.parquet")
print(f"SALVO em {out_dir} | exposição treino={int(g['n'].sum()):,} ({time.time()-t0:.0f}s)")
print("OK FIM", flush=True)
