"""Finaliza o build SEM Pool: processa sequencialmente as partições que faltam
(idempotente) e faz o merge final dos parciais -> agg_<feature>.parquet."""
import glob, os, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("ARROW_NUM_THREADS", "1")
import pandas as pd
import build_aggs_pod as b

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

files = sorted(glob.glob(b.DATA + "/ano=*/*.parquet"))
log(f"{len(files)} partições no total; processando faltantes sequencialmente...")
for fp in files:
    idx = os.path.basename(fp).replace(".parquet", "")
    ano = int(fp.split("ano=")[1].split("/")[0])
    if os.path.exists(f"{b.PARTS}/{ano}__{idx}.parquet"):
        continue
    log(f"processando {ano}/{idx} ...")
    b.process_partition(fp)
    log(f"  ok {ano}/{idx}")

parts = sorted(glob.glob(b.PARTS + "/*.parquet"))
log(f"merge final de {len(parts)} parciais...")
allp = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
for f in b.LAG_FEATURES:
    sub = allp[allp["feature"] == f]
    (sub.groupby(["valor", "ano"], as_index=False)[["n", "k_sjc"]].sum()
        .to_parquet(f"{b.OUT}/agg_{f}.parquet", index=False))
open(b.OUT + "/DONE", "w").write("ok")
log(f"FIM — {len(b.LAG_FEATURES)} aggs em {b.OUT}")
