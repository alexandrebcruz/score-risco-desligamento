"""Extrapolação de Weibull das curvas de sobrevivência (uma por categoria de risco),
por REGRESSÃO PURA sobre os 12 pontos observados de S(t) (sem ancorar em ponto algum).

Método (por categoria):
  1) ajuste por linearização cloglog:  ln(-ln S(t)) = p*ln t + ln(alpha)   (regressão linear
     OLS sobre os meses 1..12 com 0<S<1)  ->  shape p, alpha, escala lambda=alpha^(-1/p).
  2) curva ajustada/extrapolada = o PRÓPRIO Weibull, sem ancoragem:
        S(t) = exp(-alpha * t^p) = exp(-(t/lambda)^p)
     (não passa exatamente por nenhum ponto observado; é o melhor ajuste de mínimos quadrados).
  Mediana (forma fechada):  t_0.5 = lambda * (ln 2)^(1/p).

Saídas:
  outputs/tables/sobrevivencia_weibull_params_2023.csv   (p, lambda, R2, S12_fit vs S12_obs, S24, S36)
  outputs/tables/sobrevivencia_weibull_extrap_2023.csv   (longo: categoria x mes 0..36 -> S_obs/S_weib)
  outputs/figures/sobrevivencia_weibull_extrap_2023.png

Uso:  MPLCONFIGDIR=/tmp/mpl /tmp/consig_venv/bin/python extrap_weibull_categorias.py
"""
import os, shutil, math
import numpy as np, pandas as pd

KM = "outputs/tables/sobrevivencia_km_2023.csv"
RES = "outputs/tables/sobrevivencia_resumo_2023.csv"
TAB = "outputs/tables"; FIG = "outputs/figures"
H_OBS = 12               # janela observada (marcadores)
FIT_LO, FIT_HI = 1, 12   # janela de AJUSTE = todos os 12 pontos
H_EXT = 36               # horizonte de extrapolação (meses)

km = pd.read_csv(KM)
res = pd.read_csv(RES).set_index("categoria")
ks = sorted(km["categoria"].unique())

param_rows, curve_rows = [], []
for k in ks:
    g = km[(km.categoria == k)].sort_values("mes")
    Sobs = {int(m): float(s) for m, s in zip(g.mes, g.S)}     # S observado 0..12

    # --- (1) ajuste Weibull por regressão OLS na linearização cloglog (12 pontos) ---
    fit = g[(g.mes >= FIT_LO) & (g.mes <= FIT_HI) & (g.S > 0) & (g.S < 1)]
    t = fit.mes.values.astype(float); S = fit.S.values
    x = np.log(t); y = np.log(-np.log(S))
    p, lnα = np.polyfit(x, y, 1)
    alpha = float(np.exp(lnα)); p = float(p)
    lam = alpha ** (-1.0 / p)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)

    # --- (2) curva pura (sem ancoragem): S(t)=exp(-alpha*t^p) ---
    def S_w(tt):
        tt = np.asarray(tt, float)
        return np.exp(-alpha * tt ** p)

    # --- estatísticas da distribuição Weibull fitada (tempo até desligamento T) ---
    # quantil q: t_q = lambda*(-ln(1-q))^(1/p)   |   média: lambda*Gamma(1+1/p)
    def tq(q): return float(lam * (-math.log(1.0 - q)) ** (1.0 / p))
    media   = float(lam * math.gamma(1.0 + 1.0 / p))
    q1      = tq(0.25)
    mediana = tq(0.50)            # = lambda*(ln2)^(1/p)
    q3      = tq(0.75)

    param_rows.append(dict(
        categoria=int(k), shape_p=round(p, 4), escala_lambda_meses=round(lam, 2),
        alpha=round(alpha, 6), R2_ajuste=round(r2, 4),
        S12_obs=round(Sobs[H_OBS], 4), S12_fit=round(float(S_w(12)), 4),
        S24_weib=round(float(S_w(24)), 4), S36_weib=round(float(S_w(H_EXT)), 4),
        risco_24m=round(1 - float(S_w(24)), 4), risco_36m=round(1 - float(S_w(H_EXT)), 4),
        # estatísticas do tempo-até-desligamento (meses), pela Weibull extrapolada:
        media_meses=round(media, 1), q1_meses=round(q1, 1),
        mediana_meses=round(mediana, 1), q3_meses=round(q3, 1),
        IQR_meses=round(q3 - q1, 1),
    ))

    for m in range(0, H_EXT + 1):
        curve_rows.append(dict(
            categoria=int(k), mes=m,
            S_obs=(round(Sobs[m], 5) if m in Sobs else np.nan),
            S_weibull=round(float(S_w(m)) if m >= 1 else 1.0, 5),
        ))

params = pd.DataFrame(param_rows)
curves = pd.DataFrame(curve_rows)
os.makedirs(TAB, exist_ok=True)
params.to_csv(f"{TAB}/sobrevivencia_weibull_params_2023.csv", index=False)
curves.to_csv(f"{TAB}/sobrevivencia_weibull_extrap_2023.csv", index=False)

# tabela enxuta só com as estatísticas do tempo-até-desligamento por categoria
stats = params[["categoria", "shape_p", "escala_lambda_meses",
                "media_meses", "q1_meses", "mediana_meses", "q3_meses", "IQR_meses"]]
stats.to_csv(f"{TAB}/sobrevivencia_weibull_estatisticas_2023.csv", index=False)

# ---------------------------------------------------------------------------
# Figura: observado (marcadores 0..12) + curva Weibull ajustada/extrapolada (0..36)
# ---------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

cmap = plt.get_cmap("RdYlGn_r"); norm = Normalize(vmin=min(ks), vmax=max(ks))
fig, ax = plt.subplots(figsize=(11.5, 6.4))
for k in ks:
    c = cmap(norm(k))
    o = curves[(curves.categoria == k) & (curves.mes <= H_OBS)]
    win = curves[(curves.categoria == k) & (curves.mes <= H_OBS)]      # fit dentro de 12m (sólido)
    ext = curves[(curves.categoria == k) & (curves.mes >= H_OBS)]      # extrapolação (tracejado)
    ax.plot(o.mes, o.S_obs, color=c, lw=0, marker="o", markersize=3,
            markeredgecolor="white", markeredgewidth=0.4, zorder=3)    # pontos observados
    ax.plot(win.mes, win.S_weibull, color=c, lw=1.2, alpha=0.9, zorder=2)        # curva ajustada
    ax.plot(ext.mes, ext.S_weibull, color=c, lw=1.2, ls="--", alpha=0.85, zorder=1)  # extrapolação
ax.axvline(H_OBS, color="#444", lw=1, ls=":", zorder=0)
ax.text(H_OBS + 0.2, 0.02, "ajuste (12m)  |  extrapolação (Weibull, regressão pura)",
        fontsize=8.5, color="#444")
ax.set_xlim(0, H_EXT); ax.set_xticks(range(0, H_EXT + 1, 3)); ax.set_ylim(0, 1.001)
ax.set_xlabel("meses desde jan/2023"); ax.set_ylabel("S(t) = P(continuar empregado)")
ax.set_title("Sobrevivência por categoria — Weibull por regressão pura (12 pts) + extrapolação até 36m")
ax.grid(alpha=0.25)
sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=ax); cb.set_label("categoria de risco (1=mínimo ... %d=alto)" % max(ks))
fig.tight_layout()
TMP = "/tmp/sobrevivencia_weibull_extrap_2023.png"
fig.savefig(TMP, dpi=130); os.makedirs(FIG, exist_ok=True)
shutil.copy(TMP, f"{FIG}/sobrevivencia_weibull_extrap_2023.png")

pd.set_option("display.width", 200, "display.max_rows", 30)
print("=== ESTATÍSTICAS DO TEMPO-ATÉ-DESLIGAMENTO (Weibull fitada, em meses) ===")
print(stats.to_string(index=False))
print(f"\nR2 médio do ajuste: {params.R2_ajuste.mean():.3f}")
gap = (params.S12_fit - params.S12_obs).abs().mean()
print(f"|S12_fit - S12_obs| médio = {gap:.4f}  (regressão pura, sem ancoragem)")
print("OBS: média/Q1/Q3 assumem que T~Weibull até o infinito (todos eventualmente desligados);")
print("     nas categorias de baixo risco são EXTRAPOLAÇÕES bem além dos 12m observados.")
