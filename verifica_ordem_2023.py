"""Verifica se o interim reprocessado (rais_v2) preserva a ORDEM e as linhas do
interim atual — comparando, por região, colunas estáveis linha a linha.
Se bater 100%, a ordem é idêntica e o parquet de predições/sobrevivência segue válido.

Uso: /tmp/consig_venv/bin/python verifica_ordem_2023.py
"""
import glob, os, numpy as np, pyarrow.parquet as pq

CUR = "data/interim/rais/ano=2023"
NEW = "data/interim/rais_v2/ano=2023"
CHK = ["mes_deslig", "motivo_unificado", "cbo", "idade", "tamanho_estab"]  # colunas estáveis

def _isnan(x):
    return np.isnan(x) if x.dtype.kind == "f" else np.zeros(len(x), bool)

ok_all = True
for nf in sorted(glob.glob(f"{NEW}/*.parquet")):
    reg = os.path.basename(nf)
    cf = f"{CUR}/{reg}"
    if not os.path.exists(cf):
        print(f"{reg}: SEM correspondente no interim atual — pulando"); continue
    cur = pq.ParquetFile(cf).read(columns=CHK).to_pandas()
    new = pq.ParquetFile(nf).read(columns=CHK + ["mes_admissao"]).to_pandas()
    same_n = len(cur) == len(new)
    diffs = {}
    if same_n:
        for c in CHK:
            a, b = cur[c].to_numpy(), new[c].to_numpy()
            # compara tratando NaN==NaN como igual
            ne = (a != b) & ~(_isnan(a) & _isnan(b)) if a.dtype.kind == "f" else (a != b)
            diffs[c] = int(np.sum(ne))
    ma = new["mes_admissao"].to_numpy()
    vig = int(np.sum(ma == 0)); novo = int(np.sum((ma >= 1) & (ma <= 12)))
    ok = same_n and all(v == 0 for v in diffs.values())
    ok_all &= ok
    print(f"{reg}: n_cur={len(cur):,} n_new={len(new):,} same_n={same_n} | "
          f"diffs={diffs} | mes_adm: vigentes(0)={vig:,} novos(1-12)={novo:,} | {'OK' if ok else 'DIVERGE!'}")
print("\n==> ORDEM PRESERVADA EM TODAS AS REGIÕES" if ok_all else "\n==> ATENÇÃO: há divergência")
