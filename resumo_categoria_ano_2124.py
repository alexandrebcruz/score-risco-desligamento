"""Tabela-resumo categoria_risco × ano (2016–2025) sobre as predições categorizadas:
n, taxa real de dispensa (y médio) e share da base, por categoria em cada ano.
Mostra a estabilidade das 14 faixas DENTRO (2021–24) e FORA (2016–20, 2025) da
janela de referência da categorização.

Saídas: outputs/tables/categoria_ano_n_2124.csv      (n por categoria × ano)
        outputs/tables/categoria_ano_taxa_2124.csv   (taxa_y por categoria × ano)
        outputs/tables/categoria_ano_share_2124.csv  (% da base do ano por categoria)
"""
import glob, time
import pandas as pd
import pyarrow.parquet as pq

PRED = "data/processed/predicoes_2124"
t0 = time.time()

# agrega (ano, categoria) -> n, k usando só 3 colunas (memória baixa)
acc = {}
for fp in sorted(glob.glob(f"{PRED}/ano=*/*.parquet")):
    t = pq.ParquetFile(fp).read(columns=["ano", "categoria_risco", "y"]).to_pandas()
    g = t.groupby(["ano", "categoria_risco"])["y"].agg(["size", "sum"])
    for (a, c), (n, k) in g.iterrows():
        key = (int(a), int(c))
        pn, pk = acc.get(key, (0, 0))
        acc[key] = (pn + int(n), pk + int(k))
    print(f"[{time.time()-t0:5.0f}s] {fp}", flush=True)

rows = [{"ano": a, "categoria": c, "n": n, "k": k, "taxa_y": k / n}
        for (a, c), (n, k) in sorted(acc.items())]
df = pd.DataFrame(rows)
df["share_ano_%"] = 100 * df["n"] / df.groupby("ano")["n"].transform("sum")

piv_n = df.pivot(index="categoria", columns="ano", values="n")
piv_t = df.pivot(index="categoria", columns="ano", values="taxa_y")
piv_s = df.pivot(index="categoria", columns="ano", values="share_ano_%")
piv_n.to_csv("outputs/tables/categoria_ano_n_2124.csv")
piv_t.to_csv("outputs/tables/categoria_ano_taxa_2124.csv")
piv_s.to_csv("outputs/tables/categoria_ano_share_2124.csv")

pd.set_option("display.width", 220)
print("\n=== TAXA REAL (y médio) por categoria × ano ===")
print((piv_t * 100).round(1).to_string())
print("\n=== SHARE da base do ano (%) por categoria ===")
print(piv_s.round(1).to_string())
print("\n=== monotonicidade por ano (taxa estritamente crescente?) ===")
for a in piv_t.columns:
    col = piv_t[a].dropna()
    print(f"  {a}: {'OK' if (col.diff().dropna() > 0).all() else 'QUEBRA'}")
print("FIM_RESUMO_2124", flush=True)
