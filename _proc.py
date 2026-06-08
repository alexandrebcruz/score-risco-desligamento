"""Reprocessa as partições RAIS -> interim (com as colunas novas). Idempotente,
escrita atômica (.part -> rename), re-extrai os .7z do cache."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).parent; sys.path.insert(0, str(ROOT))
import pyarrow as pa, pyarrow.parquet as pq
from src.config import load_config
from src import io_utils, cleaning

cfg = load_config()
raw = cfg["abs"]["raw"]; interim = cfg["abs"]["interim"]
out_root = interim / "rais"; out_root.mkdir(parents=True, exist_ok=True)
ufs = cfg.get("ufs_subset")

def escreve(dest, chunks):
    w = None; n = 0
    for ch in chunks:
        t = pa.Table.from_pandas(ch, preserve_index=False)
        if w is None: w = pq.ParquetWriter(dest, t.schema)
        w.write_table(t); n += len(ch)
    if w: w.close()
    return n

t0 = time.time()
for ano in cfg["anos"]:
    pdir = out_root / f"ano={ano}"; pdir.mkdir(parents=True, exist_ok=True)
    for z in sorted((raw / "RAIS" / str(ano)).glob("*.7z")):
        regiao = z.stem.replace("RAIS_VINC_PUB_", "")
        dest = pdir / f"{regiao}.parquet"
        if dest.exists():
            continue
        print(f"{ano}/{regiao}: extraindo...", flush=True)
        extr = io_utils.extract_7z(z, raw / "RAIS" / str(ano))
        arq = next(p for p in extr if p.suffix.upper() in (".COMT", ".TXT"))
        tmp = dest.with_suffix(".parquet.part")
        n = escreve(tmp, cleaning.iter_rais_clean_chunks(arq, ano, ufs))
        tmp.rename(dest); arq.unlink()
        print(f"{ano}/{regiao}: {n:,} ({time.time()-t0:.0f}s)", flush=True)
print("OK todas as partições", flush=True)
