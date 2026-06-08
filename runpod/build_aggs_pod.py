"""Build PARALELO das agregações de lag, para rodar na pod CPU (muitos núcleos).

Cada partição do interim (/workspace/data/rais/ano=*/*.parquet) é processada por
um worker independente (multiprocessing.Pool) -> agregações parciais por feature;
o processo principal concatena e soma por (valor, ano), salvando agg_<f>.parquet.

Lógica de normalização espelha src/cleaning.normalize_short_codes + src/lags._derive_hier:
  - causa_afastamento: remap '999'->'99' (default "sem afastamento" recodificado em 2023)
  - faixa_remuneracao / faixa_horas / causa_afastamento: remove zero-padding ('02'->'2')
  - cbo/cnae: zfill(6/7) ANTES de fatiar (2023 perde o zero à esquerda)

Uso na pod:  python build_aggs_pod.py
Saída:       /workspace/out_aggs/agg_<feature>.parquet
"""
import glob, os, time, sys
# Limita threads internas (BLAS/pyarrow) ANTES de importar pandas — evita
# contenção/deadlock quando há muitos processos.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp
import pandas as pd

DATA = "/workspace/data/rais"
OUT = "/workspace/out_aggs"
PARTS = "/workspace/agg_parts"   # parciais por partição (worker grava aqui)
os.makedirs(OUT, exist_ok=True)
os.makedirs(PARTS, exist_ok=True)
MOTIVO = "involuntario_sjc"
NWORKERS = int(os.environ.get("NWORKERS", "24"))

LAG_FEATURES = ["cbo", "cbo4", "cbo2", "cbo1", "cnae", "cnae5", "cnae3", "cnae2",
                "uf", "escolaridade", "tamanho_estab", "tipo_vinculo", "faixa_remuneracao",
                "natureza_juridica", "natureza_setor", "intermitente", "simples",
                "faixa_horas", "causa_afastamento"]
BASE_COLS = ["cbo", "cnae", "uf", "escolaridade", "tamanho_estab", "tipo_vinculo",
             "faixa_remuneracao", "natureza_juridica", "natureza_setor",
             "intermitente", "simples", "faixa_horas", "causa_afastamento",
             "motivo_unificado"]
CODIGOS_STRIP = ["faixa_remuneracao", "faixa_horas", "causa_afastamento"]
CODE_REMAP = {"causa_afastamento": {"999": "99"}}


import re as _re
_RX = _re.compile(r"^0+(?=\d)")

def _fast_map(s, func):
    """Aplica `func` apenas aos valores ÚNICOS (poucos) e mapeia de volta —
    evita operação de string por linha em colunas de milhões de linhas."""
    u = s.astype(str).unique()
    return s.astype(str).map({v: func(v) for v in u})

def _norm_strip(v):
    return _RX.sub("", v) if _re.fullmatch(r"0*\d+", v) else v

def _normalize(d):
    for c, mapa in CODE_REMAP.items():
        if c in d.columns:
            d[c] = _fast_map(d[c], lambda v: mapa.get(v, v))
    for c in CODIGOS_STRIP:
        if c in d.columns:
            d[c] = _fast_map(d[c], _norm_strip)
    # zfill consistente + substrings hierárquicas (também via mapa de únicos)
    d["cbo"] = _fast_map(d["cbo"], lambda v: v.zfill(6))
    d["cnae"] = _fast_map(d["cnae"], lambda v: v.zfill(7))
    cbo, cnae = d["cbo"], d["cnae"]
    d["cbo4"], d["cbo2"], d["cbo1"] = cbo.str[:4], cbo.str[:2], cbo.str[:1]
    d["cnae5"], d["cnae3"], d["cnae2"] = cnae.str[:5], cnae.str[:3], cnae.str[:2]
    return d


def process_partition(fp):
    """Worker: 1 partição -> grava 1 parquet com TODAS as features empilhadas
    (colunas: feature, valor, ano, n, k_sjc) e retorna só o caminho (sem pickles
    grandes pelo pipe)."""
    idx = os.path.basename(fp).replace(".parquet", "")
    ano = int(fp.split("ano=")[1].split("/")[0])
    out_fp = f"{PARTS}/{ano}__{idx}.parquet"
    if os.path.exists(out_fp):
        return out_fp
    d = pd.read_parquet(fp, columns=BASE_COLS)
    d = _normalize(d)
    d["k"] = (d["motivo_unificado"] == MOTIVO).astype("int64")
    blocks = []
    for f in LAG_FEATURES:
        g = (d.groupby(f, observed=True)
               .agg(n=("k", "size"), k_sjc=("k", "sum")).reset_index())
        g = g.rename(columns={f: "valor"})
        g["valor"] = g["valor"].astype(str)
        g["feature"] = f
        g["ano"] = ano
        blocks.append(g[["feature", "valor", "ano", "n", "k_sjc"]])
    tmp = out_fp + ".tmp"
    pd.concat(blocks, ignore_index=True).to_parquet(tmp, index=False)
    os.replace(tmp, out_fp)
    return out_fp


def main():
    t0 = time.time()
    files = sorted(glob.glob(f"{DATA}/ano=*/*.parquet"))
    print(f"{len(files)} partições | {NWORKERS} workers (spawn) | {os.cpu_count()} núcleos", flush=True)
    ctx = mp.get_context("spawn")          # evita deadlock de fork-após-threads
    done = 0
    part_files = []
    with ctx.Pool(processes=NWORKERS, maxtasksperchild=4) as pool:
        for pf in pool.imap_unordered(process_partition, files):
            part_files.append(pf)
            done += 1
            print(f"[{time.time()-t0:6.1f}s] {done}/{len(files)} partições agregadas", flush=True)
    print(f"[{time.time()-t0:6.1f}s] merge final ({len(part_files)} parciais)...", flush=True)
    allp = pd.concat([pd.read_parquet(p) for p in part_files], ignore_index=True)
    for f in LAG_FEATURES:
        sub = allp[allp["feature"] == f]
        big = sub.groupby(["valor", "ano"], as_index=False)[["n", "k_sjc"]].sum()
        big.to_parquet(f"{OUT}/agg_{f}.parquet", index=False)
    open(f"{OUT}/DONE", "w").write("ok")
    print(f"[{time.time()-t0:6.1f}s] FIM — {len(LAG_FEATURES)} agg em {OUT}", flush=True)


if __name__ == "__main__":
    main()
