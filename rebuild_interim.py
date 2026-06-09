"""Reprocessa o interim da RAIS para TODOS os anos do config -> pasta NOVA
`data/interim/rais_v2/ano=YYYY`, sem tocar no interim atual. Usa o pipeline vivo
(io_utils.extract_7z + cleaning.iter_rais_clean_chunks -> clean_rais_real), que já:
- guarda os CÓDIGOS CRUS das categóricas (escolaridade 1..11, motivo_desligamento);
- harmoniza formato entre anos (normalize_short_codes + zfill de cbo/cnae no interim).

Características:
- só processa arquivos de VÍNCULOS (ignora ESTAB); lida com 2016/2017 por UF
  (AC2016.7z…) e 2018+ por região (RAIS_VINC_PUB_*), incluindo o NI;
- idempotente (pula partição já gravada) e atômico (.part único por PID -> rename);
- ufs_subset=None -> sem filtro/reordenação de linha (ordem idêntica à do RAW);
- remove o .COMT extraído ao fim de cada arquivo.

Uso:
  /tmp/consig_venv/bin/python rebuild_interim.py            # todos os anos do config
  /tmp/consig_venv/bin/python rebuild_interim.py 2024 2025  # só estes anos

ATENÇÃO: volume nacional (~563M+ vínculos) -> rodar em POD ou localmente por ano
(é resumível). Depois de validar, trocar data/interim/rais_v2 -> rais.
"""
import sys, os, re, glob, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pyarrow as pa, pyarrow.parquet as pq
from src.config import load_config
from src import io_utils, cleaning

cfg = load_config()
RAW = cfg["abs"]["raw"]; INTERIM = cfg["abs"]["interim"]
OUT_ROOT = INTERIM / "rais_v2"
ANOS = [int(a) for a in sys.argv[1:]] or list(cfg["anos"])

t0 = time.time()
def log(m): print(f"[{time.time()-t0:8.0f}s] {m}", flush=True)

def raw_dir(ano):
    """data/raw/rais/<ano> (DrvFs é case-insensitive; tenta variações)."""
    for sub in ("rais", "RAIS"):
        d = RAW / sub / str(ano)
        if d.exists():
            return d
    return RAW / "rais" / str(ano)

def is_vinc(p):
    up = os.path.basename(p).upper()
    if "ESTAB" in up or up.startswith("ESTB"):
        return False
    if up.startswith("RAIS_VINC_PUB_"):
        return True
    return bool(re.match(r"^[A-Z]{2}\d{4}\.7Z$", up))      # 2016/2017 por UF (ex.: AC2016.7z)

def part_name(p):
    b = os.path.basename(p)
    if b.upper().startswith("RAIS_VINC_PUB_"):
        return re.sub(r"(?i)^RAIS_VINC_PUB_", "", b)[:-3]   # remove prefixo e ".7z"
    return os.path.splitext(b)[0]                           # ex.: AC2016

def escreve(dest, chunks):
    writer = None; n = 0
    for ch in chunks:
        t = pa.Table.from_pandas(ch, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(dest, t.schema)
        writer.write_table(t); n += len(ch)
    if writer:
        writer.close()
    return n

log(f"anos a processar: {ANOS}")
for ano in ANOS:
    rdir = raw_dir(ano)
    zs = sorted(z for z in glob.glob(str(rdir / "*.7z")) if is_vinc(z))
    if not zs:
        log(f"{ano}: NENHUM arquivo de vínculos em {rdir} — pulando"); continue
    odir = OUT_ROOT / f"ano={ano}"; odir.mkdir(parents=True, exist_ok=True)
    log(f"{ano}: {len(zs)} arquivos de vínculos")
    for z in zs:
        part = part_name(z)
        dest = odir / f"{part}.parquet"
        if dest.exists():
            log(f"  {ano}/{part}: já existe, pulando"); continue
        log(f"  {ano}/{part}: extraindo {os.path.basename(z)} ...")
        extr = io_utils.extract_7z(Path(z), rdir)
        arq = next(p for p in extr if p.suffix.upper() in (".COMT", ".TXT"))
        tmp = Path(str(dest) + f".{os.getpid()}.part")        # tmp único por processo
        log(f"  {ano}/{part}: lendo+limpando {arq.name} ...")
        n = escreve(tmp, cleaning.iter_rais_clean_chunks(str(arq), ano, None))
        os.replace(tmp, dest)
        arq.unlink()
        log(f"  {ano}/{part}: {n:,} vínculos -> {dest.name}")

log("FIM rebuild_interim")
parts = sorted(glob.glob(str(OUT_ROOT / "ano=*/*.parquet")))
if parts:
    cols = [f.name for f in pq.ParquetFile(parts[0]).schema]
    log(f"{len(parts)} partições gravadas em {OUT_ROOT}")
    log(f"colunas do novo interim: {cols}")
