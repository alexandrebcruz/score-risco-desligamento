"""Taxa de juros MÍNIMA (break-even nominal) por categoria × prazo, a partir da
cobertura esperada de parcelas (consignado_cobertura_parcelas_2124.csv).

Modelo (Tabela Price, hipótese conservadora de ZERO recuperação após o desligamento):
  parcela A = P·i / (1 − (1+i)^−T)
  recebido esperado = A · Σ_{m=1..T} S(m) = A · T · cobertura(T)
  break-even (receber ≥ principal P):  i·T·c / (1 − (1+i)^−T) ≥ 1
Resolve i (mensal) por bisseção; anualiza por (1+i)^12 − 1.

Saída: outputs/tables/consignado_taxa_breakeven_2124.csv
  colunas: categoria, m_T{6..60} (% a.m.), a_T{6..60} (% a.a.)

Uso: /tmp/consig_venv/bin/python tabelas_taxa_breakeven_2124.py
"""
import numpy as np, pandas as pd

COB = "outputs/tables/consignado_cobertura_parcelas_2124.csv"
OUT = "outputs/tables/consignado_taxa_breakeven_2124.csv"
TS = [6, 12, 18, 24, 36, 48, 60]

cob = pd.read_csv(COB).set_index("categoria")


def breakeven_mensal(T, c):
    """Menor i (mensal) com i·T·c/(1−(1+i)^−T) ≥ 1.  c = cobertura (fração)."""
    if c <= 0:
        return np.inf
    if c >= 1:
        return 0.0
    f = lambda i: i * T * c / (1 - (1 + i) ** (-T)) - 1.0
    lo, hi = 1e-12, 5.0                     # teto de busca: 500%/mês
    if f(hi) < 0:
        return np.inf
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) >= 0:
            hi = mid
        else:
            lo = mid
    return hi


rows = []
for cat in cob.index:
    r = {"categoria": int(cat)}
    for T in TS:
        c = float(cob.loc[cat, f"T_{T}"]) / 100.0
        im = breakeven_mensal(T, c)
        ia = (1 + im) ** 12 - 1 if np.isfinite(im) else np.inf
        r[f"m_T{T}"] = round(im * 100, 3)   # % ao mês
        r[f"a_T{T}"] = round(ia * 100, 2)   # % ao ano
    rows.append(r)

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)

pd.set_option("display.width", 200)
print("=== TAXA MÍNIMA (break-even) % ao MÊS ===")
print(res[["categoria"] + [f"m_T{t}" for t in TS]].to_string(index=False))
print("\n=== % ao ANO ===")
print(res[["categoria"] + [f"a_T{t}" for t in TS]].to_string(index=False))
print(f"\n-> {OUT}")
print("FIM_TAXA_2124")
