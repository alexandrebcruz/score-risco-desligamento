"""Taxa de juros por NPV (Valor Presente Líquido) por categoria × prazo, com custo
de captação e ROI-alvo — a taxa de PRICING (não apenas o piso de quebra-zero).

Modelo:
  NPV = −P + Σ_{m=1..T} A·S(m) / (1+r_f)^m ,  A = P·i/(1−(1+i)^−T)   (Tabela Price)
  Pricing-alvo: NPV = ROI·P  (lucro a valor presente = ROI do principal).
  Seja D(c,T) = Σ S(m)/(1+r_f)^m  (cobertura esperada DESCONTADA, em parcelas-VP).
  ⇒  i/(1−(1+i)^−T) = (1+ROI)/D(c,T)   →  resolve i (mensal) por bisseção.
  anual = (1+i)^12 − 1.
  (break-even nominal = caso particular r_f=0, ROI=0.)

Parâmetros: r_f = 1,2%/mês; ROI ∈ {10%, 20%}.  S = KM MOB (≤12) + Weibull (>12).

Saída: outputs/tables/consignado_taxa_npv_2124.csv
  colunas: categoria, m10_T{t}/a10_T{t} (ROI 10%), m20_T{t}/a20_T{t} (ROI 20%)

Uso: /tmp/consig_venv/bin/python tabelas_taxa_npv_2124.py
"""
import math
import numpy as np, pandas as pd

KM = "outputs/tables/sobrevivencia_km_mob_2124.csv"
PAR = "outputs/tables/sobrevivencia_weibull_params_mob_2124.csv"
OUT = "outputs/tables/consignado_taxa_npv_2124.csv"

R_FUND = 0.012          # custo de captação 1,2% ao mês
ROIS = [("10", 0.10), ("20", 0.20)]
TS = [6, 12, 18, 24, 36, 48, 60]

km = pd.read_csv(KM); par = pd.read_csv(PAR).set_index("categoria")
ks = sorted(km["categoria"].unique())
kmS = {(int(r.categoria), int(r.mob)): float(r.S) for r in km.itertuples()}


def S(c, m):
    if m <= 12:
        return kmS[(c, m)]
    lam, p = float(par.loc[c, "escala_lambda_meses"]), float(par.loc[c, "shape_p"])
    return math.exp(-(m / lam) ** p)


def crf(i, T):
    """Fator de recuperação de capital  i/(1−(1+i)^−T)  (cresce com i)."""
    return i / (1 - (1 + i) ** (-T))


def taxa_npv(c, T, roi):
    """Menor i (mensal) com NPV = ROI·P, i.e. crf(i,T)·D = 1+ROI."""
    D = sum(S(c, m) / (1 + R_FUND) ** m for m in range(1, T + 1))
    if D <= 0:
        return np.inf
    target = (1 + roi) / D
    f = lambda i: crf(i, T) - target
    lo, hi = 1e-12, 5.0
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
for c in ks:
    r = {"categoria": int(c)}
    for tag, roi in ROIS:
        for T in TS:
            im = taxa_npv(int(c), T, roi)
            ia = (1 + im) ** 12 - 1 if np.isfinite(im) else np.inf
            r[f"m{tag}_T{T}"] = round(im * 100, 3)
            r[f"a{tag}_T{T}"] = round(ia * 100, 2)
    rows.append(r)

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)

pd.set_option("display.width", 220)
for tag, roi in ROIS:
    print(f"=== TAXA NPV (r_f=1,2%/mês, ROI={int(roi*100)}%) — % ao MÊS ===")
    print(res[["categoria"] + [f"m{tag}_T{t}" for t in TS]].to_string(index=False))
    print()
print(f"-> {OUT}")
print("FIM_NPV_2124")
