"""Reprocessa o interim da RAIS 2023 INCLUINDO o campo `mes_admissao`, numa pasta
NOVA (data/interim/rais_v2/ano=2023), sem tocar no interim atual.

Replica exatamente o build do nb01: extrai cada .7z, lê em chunks limpos
(cleaning.iter_rais_clean_chunks com ufs_subset=None -> SEM filtro de linha), grava
por região via ParquetWriter (chunks na ordem do arquivo) e apaga o .COMT.

Como ufs_subset=None e o cleaning não filtra/reordena, a ordem das linhas é idêntica
ao interim atual — verificada depois por verifica_ordem_2023.py.

Uso: /tmp/consig_venv/bin/python rebuild_interim_2023.py
"""
import sys, time, glob, os
import pyarrow as pa, pyarrow.parquet as pq
from src import io_utils, cleaning

ANO = 2023
RAW = f"data/raw/RAIS/{ANO}"
OUT = f"data/interim/rais_v2/ano={ANO}"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)

def escreve(dest, chunks):
    writer = None; n = 0
    for ch in chunks:
        t = pa.Table.from_pandas(ch, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(dest, t.schema)
        writer.write_table(t); n += len(ch)
    if writer: writer.close()
    return n

for z in sorted(glob.glob(f"{RAW}/*.7z")):
    regiao = os.path.basename(z).replace("RAIS_VINC_PUB_", "").replace(".7z", "")
    dest = f"{OUT}/{regiao}.parquet"
    if os.path.exists(dest):
        log(f"{regiao}: já processado, pulando"); continue
    log(f"{regiao}: extraindo {os.path.basename(z)} ...")
    extr = io_utils.extract_7z(__import__("pathlib").Path(z), __import__("pathlib").Path(RAW))
    arq = next(p for p in extr if p.suffix.upper() in (".COMT", ".TXT"))
    log(f"{regiao}: lendo+limpando {arq.name} ...")
    tmp = dest + f".{os.getpid()}.part"      # tmp único por processo (evita corrida)
    n = escreve(tmp, cleaning.iter_rais_clean_chunks(str(arq), ANO, None))
    os.replace(tmp, dest)
    arq.unlink()
    log(f"{regiao}: {n:,} vínculos -> {os.path.basename(dest)} (extraído removido)")

log("FIM rebuild 2023")
cols = [f.name for f in pq.ParquetFile(sorted(glob.glob(f'{OUT}/*.parquet'))[0]).schema]
log(f"colunas do novo interim: {cols}")
