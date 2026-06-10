"""STEP 1 — gera o dataset ENRIQUECIDO com lag features, no pod (1x, salvo em disco).

Para cada partição do interim (/workspace/data/rais/ano=*/*.parquet), aplica os
joins de lag (n e k_sjc das categorias nos 3 anos anteriores) e salva o parquet
enriquecido em /workspace/data/rais_lags/ano=Y/<regiao>.parquet.

Processa UMA partição por vez (memória baixa) e é idempotente (pula prontas).
Depois, train_ensemble_lags.py só LÊ esses arquivos — sem refazer joins.

Uso: nohup python prep_lags.py > /workspace/prep.log 2>&1 &
"""
import time, glob, os, gc
import numpy as np
import pandas as pd
DATA = "/workspace/data/rais"
OUT = "/workspace/data/rais_lags"
LAG_DIR = "/workspace/lags"
ANOS = [2019, 2020, 2021, 2022, 2023]

EXTRA_CAT = ["tipo_vinculo", "natureza_juridica", "natureza_setor",
             "intermitente", "simples"]
# Ordinais (a ORDEM do código tem significado) -> NUMÉRICAS; 99=ignorado -> -1.
# (Seguem em LAG_FEATURES como chave de join por VALOR — convertidas após o join.)
ORD = ["escolaridade", "tamanho_estab", "faixa_remuneracao", "faixa_horas"]
BASE_CAT = ["cbo", "cbo4", "cbo2", "cbo1", "cnae", "cnae5", "cnae3", "cnae2",
            "uf"]
CAT = BASE_CAT + EXTRA_CAT
RAW = (["cbo", "cnae", "uf",
        "idade", "tempo_vinculo_meses", "qtd_dias_afastamento", "motivo_unificado"] + ORD + EXTRA_CAT)
BASE_NUM = ["idade", "tempo_vinculo_meses", "qtd_dias_afastamento"]
LAG_FEATURES = ["cbo", "cbo4", "cbo2", "cbo1", "cnae", "cnae5", "cnae3", "cnae2",
                "uf", "escolaridade", "tamanho_estab", "tipo_vinculo", "faixa_remuneracao",
                "natureza_juridica", "natureza_setor", "intermitente", "simples",
                "faixa_horas"]
LAGS = (1, 2, 3)
LAG_COLS = [f"{f}_{k}_lag{L}" for f in LAG_FEATURES for L in LAGS for k in ("n", "k")]
FEATURES = CAT + ORD + BASE_NUM + LAG_COLS
TARGET_MOTIVO = "involuntario_sjc"

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)

AGGS = {f: pd.read_parquet(f"{LAG_DIR}/agg_{f}.parquet") for f in LAG_FEATURES}
log(f"agregações carregadas ({len(LAG_FEATURES)} features)")

# NOTA: remap 999->99, strip de zeros e zfill de cbo/cnae NÃO são mais feitos aqui —
# o interim novo (src/cleaning.clean_rais_real) já entrega tudo harmonizado (faixas/
# escolaridade como int64; cbo/cnae zfillados). Só derivamos os níveis hierárquicos.
# ATENÇÃO: os aggs de lag (agg_*.parquet) construídos sobre o interim ANTIGO têm
# valores no formato antigo (ex.: escolaridade "superior") -> RECONSTRUIR os aggs
# (build_aggs_pod.py) sobre o interim novo antes de rodar este prep, senão o join
# por valor falha p/ escolaridade/faixas.

def enrich(d, ano):
    cbo, cnae = d["cbo"].astype(str), d["cnae"].astype(str)
    d["cbo4"], d["cbo2"], d["cbo1"] = cbo.str[:4], cbo.str[:2], cbo.str[:1]
    d["cnae5"], d["cnae3"], d["cnae2"] = cnae.str[:5], cnae.str[:3], cnae.str[:2]
    d["y"] = (d["motivo_unificado"] == TARGET_MOTIVO).astype("int8")
    for c in CAT:
        d[c] = d[c].astype(str)
    for c in ORD:
        d[c] = d[c].astype(str)        # str p/ o JOIN de lag por valor (volta a num após)
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
    # ordinais de volta a numéricas (99 = ignorado -> -1)
    for c in ORD:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(-1)
        d.loc[d[c] == 99, c] = -1
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
