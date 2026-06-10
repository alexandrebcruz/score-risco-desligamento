"""Predict do ENSEMBLE NOVO (retreino 2021–2024, leak-free) sobre TODOS os anos
2016–2025 do interim, salvando 1 parquet por partição (região/UF) por ano.

- Modelos: outputs/runpod_retreino_2124/catboost_{A,B}.cbm (21 features).
- Saída:   data/processed/predicoes_2124/ano=YYYY/<part>.parquet com
           id_linha + features do interim + desfecho (mes_admissao, vinculo_ativo,
           mes_deslig, motivo_unificado) + y + prob_A/prob_B/prob_desligamento.
           (id_linha permite join explícito de volta ao interim.)
- RESUMÍVEL por partição (pula as prontas) e em lotes de 2M linhas (RAM baixa).
- pyarrow não cria arquivo direto no /mnt/d -> escreve em /tmp e copia.

Uso: nohup /tmp/consig_venv/bin/python -u predict_ensemble_2124_todos_anos.py > /tmp/pred2124.log 2>&1 &
"""
import os, glob, time, shutil, gc
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from catboost import CatBoostClassifier

DATA = "data/interim/rais"
OUT = "data/processed/predicoes_2124"
MODELS = "outputs/runpod_retreino_2124"
ANOS = list(range(2016, 2026))
BATCH = 2_000_000

EXTRA_CAT = ["tipo_vinculo", "natureza_juridica", "natureza_setor",
             "intermitente", "simples"]
ORD = ["escolaridade", "tamanho_estab", "faixa_remuneracao", "faixa_horas"]
CAT = (["cbo", "cbo4", "cbo2", "cbo1", "cnae", "cnae5", "cnae3", "cnae2",
        "uf"] + EXTRA_CAT)
NUM = ["idade", "tempo_vinculo_meses", "qtd_dias_afastamento"] + ORD
FEATURES = CAT + NUM
TARGET = "involuntario_sjc"
# colunas lidas do interim (features + id + desfecho p/ sobrevivência downstream)
LER = (["id_linha", "ano", "cbo", "cnae", "uf", "idade", "tempo_vinculo_meses",
        "qtd_dias_afastamento"] + ORD + EXTRA_CAT
       + ["mes_admissao", "vinculo_ativo", "mes_deslig", "motivo_unificado"])

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)

mA = CatBoostClassifier(); mA.load_model(f"{MODELS}/catboost_A.cbm")
mB = CatBoostClassifier(); mB.load_model(f"{MODELS}/catboost_B.cbm")
log("modelos A e B carregados")

def prep_X(d):
    """Mesmo pré-processamento do treino (train_model_2124.load)."""
    cbo, cnae = d["cbo"].astype(str), d["cnae"].astype(str)
    d["cbo4"], d["cbo2"], d["cbo1"] = cbo.str[:4], cbo.str[:2], cbo.str[:1]
    d["cnae5"], d["cnae3"], d["cnae2"] = cnae.str[:5], cnae.str[:3], cnae.str[:2]
    for c in CAT: d[c] = d[c].astype(str).astype("category")
    for c in NUM: d[c] = pd.to_numeric(d[c], errors="coerce").fillna(-1).astype("float32")
    for c in ("faixa_remuneracao", "faixa_horas"):
        d.loc[d[c] == 99, c] = -1
    return d

for ano in ANOS:
    odir = f"{OUT}/ano={ano}"; os.makedirs(odir, exist_ok=True)
    for fp in sorted(glob.glob(f"{DATA}/ano={ano}/*.parquet")):
        part = os.path.basename(fp)
        dest = f"{odir}/{part}"
        if os.path.exists(dest):
            log(f"{ano}/{part}: já existe, pulando"); continue
        tmp = f"/tmp/pred_{ano}_{part}"
        pf = pq.ParquetFile(fp)
        writer = None; n = 0
        for batch in pf.iter_batches(batch_size=BATCH, columns=LER):
            d = batch.to_pandas()
            d = prep_X(d)
            X = d[FEATURES]
            pA = mA.predict_proba(X)[:, 1]
            pB = mB.predict_proba(X)[:, 1]
            out = d[LER].copy()
            out["y"] = (d["motivo_unificado"] == TARGET).astype("int8")
            out["prob_A"] = pA.astype("float32")
            out["prob_B"] = pB.astype("float32")
            out["prob_desligamento"] = ((pA + pB) / 2).astype("float32")
            t = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, t.schema, compression="zstd")
            writer.write_table(t); n += len(out)
            del d, X, out, t; gc.collect()
        if writer: writer.close()
        shutil.move(tmp, dest)
        log(f"{ano}/{part}: {n:,} pontuados -> {dest}")

log("FIM predict 2016-2025")
print("FIM_PREDICT_TODOS", flush=True)
