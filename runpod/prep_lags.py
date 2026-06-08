"""STEP 1 — gera o dataset ENRIQUECIDO com lag features, no pod (1x, salvo em disco).

Para cada partição do interim (/workspace/data/rais/ano=*/*.parquet), aplica os
joins de lag (n e k_sjc das categorias nos 3 anos anteriores) e salva o parquet
enriquecido em /workspace/data/rais_lags/ano=Y/<regiao>.parquet.

Processa UMA partição por vez (memória baixa) e é idempotente (pula prontas).
Depois, train_ensemble_lags.py só LÊ esses arquivos — sem refazer joins.

Uso: nohup python prep_lags.py > /workspace/prep.log 2>&1 &
"""
import time, glob, os, gc, re
import numpy as np
import pandas as pd
_RX = re.compile(r"^0+(?=\d)")
def _fast_map(s, func):
    u = s.astype(str).unique()
    return s.astype(str).map({v: func(v) for v in u})
def _norm_strip(v): return _RX.sub("", v) if re.fullmatch(r"0*\d+", v) else v

DATA = "/workspace/data/rais"
OUT = "/workspace/data/rais_lags"
LAG_DIR = "/workspace/lags"
ANOS = [2019, 2020, 2021, 2022, 2023]

EXTRA_CAT = ["tipo_vinculo", "faixa_remuneracao", "natureza_juridica", "natureza_setor",
             "intermitente", "simples", "faixa_horas", "causa_afastamento"]
BASE_CAT = ["cbo", "cbo4", "cbo2", "cbo1", "cnae", "cnae5", "cnae3", "cnae2",
            "uf", "escolaridade", "tamanho_estab"]
CAT = BASE_CAT + EXTRA_CAT
RAW = (["cbo", "cnae", "uf", "escolaridade", "tamanho_estab",
        "idade", "tempo_vinculo_meses", "qtd_dias_afastamento", "motivo_unificado"] + EXTRA_CAT)
BASE_NUM = ["idade", "tempo_vinculo_meses", "qtd_dias_afastamento"]
LAG_FEATURES = ["cbo", "cbo4", "cbo2", "cbo1", "cnae", "cnae5", "cnae3", "cnae2",
                "uf", "escolaridade", "tamanho_estab", "tipo_vinculo", "faixa_remuneracao",
                "natureza_juridica", "natureza_setor", "intermitente", "simples",
                "faixa_horas", "causa_afastamento"]
LAGS = (1, 2, 3)
LAG_COLS = [f"{f}_{k}_lag{L}" for f in LAG_FEATURES for L in LAGS for k in ("n", "k")]
FEATURES = CAT + BASE_NUM + LAG_COLS
TARGET_MOTIVO = "involuntario_sjc"

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)

AGGS = {f: pd.read_parquet(f"{LAG_DIR}/agg_{f}.parquet") for f in LAG_FEATURES}
log(f"agregações carregadas ({len(LAG_FEATURES)} features)")

# Códigos curtos com zero-padding inconsistente entre anos (2019-22 padded; 2023 não).
# Normalizar (remover zeros à esquerda) é essencial: senão '02'(treino)!='2'(2023)
# -> feature inútil no holdout e join de lag falha. Espelha cleaning.normalize_short_codes.
CODIGOS_A_NORMALIZAR = ["faixa_remuneracao", "faixa_horas", "causa_afastamento"]
# Remap de conteúdo: causa_afastamento default '99'(<=2022) virou '999'(2023, ~84%).
CODE_REMAP = {"causa_afastamento": {"999": "99"}}

def _norm_codes(d):
    for c, mapa in CODE_REMAP.items():
        if c in d.columns:
            d[c] = _fast_map(d[c], lambda v: mapa.get(v, v))
    for c in CODIGOS_A_NORMALIZAR:
        if c in d.columns:
            d[c] = _fast_map(d[c], _norm_strip)
    return d

def enrich(d, ano):
    d = _norm_codes(d)
    # zfill consistente (2023 vem sem o zero à esquerda): sobrescreve a base p/ que
    # a categoria E a chave de join do lag casem entre anos.
    d["cbo"] = _fast_map(d["cbo"], lambda v: v.zfill(6))
    d["cnae"] = _fast_map(d["cnae"], lambda v: v.zfill(7))
    cbo, cnae = d["cbo"], d["cnae"]
    d["cbo4"], d["cbo2"], d["cbo1"] = cbo.str[:4], cbo.str[:2], cbo.str[:1]
    d["cnae5"], d["cnae3"], d["cnae2"] = cnae.str[:5], cnae.str[:3], cnae.str[:2]
    d["y"] = (d["motivo_unificado"] == TARGET_MOTIVO).astype("int8")
    for c in CAT:
        d[c] = d[c].astype(str)
    for c in BASE_NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(-1).astype("float32")
    for f in LAG_FEATURES:
        a = AGGS[f]
        for L in LAGS:
            sub = a[a["ano"] == ano - L][["valor", "n", "k_sjc"]].copy()
            sub["valor"] = sub["valor"].astype(str)
            sub = sub.rename(columns={"valor": f, "n": f"{f}_n_lag{L}", "k_sjc": f"{f}_k_lag{L}"})
            d = d.merge(sub, on=f, how="left")
    for c in LAG_COLS:
        d[c] = d[c].astype("float32")
    for c in CAT:
        d[c] = d[c].astype("category")
    return d[FEATURES + ["y"]]

for ano in ANOS:
    odir = f"{OUT}/ano={ano}"; os.makedirs(odir, exist_ok=True)
    for fp in sorted(glob.glob(f"{DATA}/ano={ano}/*.parquet")):
        nome = os.path.basename(fp)
        dest = f"{odir}/{nome}"
        if os.path.exists(dest):
            log(f"  {ano}/{nome}: já existe"); continue
        d = pd.read_parquet(fp, columns=RAW)
        n = len(d)
        d = enrich(d, ano)
        tmp = dest + ".part"
        d.to_parquet(tmp, index=False, compression="zstd")
        os.rename(tmp, dest)
        del d; gc.collect()
        log(f"  {ano}/{nome}: {n:,} -> enriquecido ({os.path.getsize(dest)/1e6:.0f} MB)")

open(f"{OUT}/PREP_DONE", "w").write("ok")
log(f"FIM — enriquecido em {OUT} ({len(FEATURES)} features)")
