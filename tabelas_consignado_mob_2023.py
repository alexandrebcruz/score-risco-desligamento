"""[VERSÃO MOB] Tabelas de aplicação ao crédito consignado, a partir da sobrevivência MOB:
- prazo máximo por categoria para confiança 95/90/85/80% de seguir empregado: t=λ·(−ln c)^(1/p);
- cobertura esperada de parcelas (% pagas em folha) por prazo T=6..60: Σ S(m)/T,
  com S = KM MOB (≤12m) + Weibull MOB (>12m).

Entradas: sobrevivencia_km_mob_2023.csv + sobrevivencia_weibull_params_mob_2023.csv
Saídas:
  outputs/tables/consignado_prazo_max_mob_2023.csv
  outputs/tables/consignado_cobertura_parcelas_mob_2023.csv
  outputs/figures/consignado_tabelas_mob_2023.png  (2 heatmaps)
"""
import os, math, shutil
import numpy as np, pandas as pd

KM = "outputs/tables/sobrevivencia_km_mob_2023.csv"
PAR = "outputs/tables/sobrevivencia_weibull_params_mob_2023.csv"
TAB = "outputs/tables"; FIG = "outputs/figures"

km = pd.read_csv(KM)                                  # colunas: categoria, mob, S, ...
par = pd.read_csv(PAR).set_index("categoria")
ks = sorted(km["categoria"].unique())
kmS = {(int(r.categoria), int(r.mob)): float(r.S) for r in km.itertuples()}

def S(c, m):
    if m <= 12: return kmS[(c, m)]
    lam, p = float(par.loc[c, "escala_lambda_meses"]), float(par.loc[c, "shape_p"])
    return math.exp(-(m / lam) ** p)

def termo(c, conf):
    lam, p = float(par.loc[c, "escala_lambda_meses"]), float(par.loc[c, "shape_p"])
    return lam * (-math.log(conf)) ** (1.0 / p)

def cov(c, T): return sum(S(c, m) for m in range(1, T + 1)) / T * 100

CONFS = [("conf_95", 0.95), ("conf_90", 0.90), ("conf_85", 0.85), ("conf_80", 0.80)]
TS = [6, 12, 18, 24, 36, 48, 60]

prazo = pd.DataFrame({"categoria": ks})
for lbl, c in CONFS:
    prazo[lbl] = [round(termo(k, c), 1) for k in ks]
prazo.to_csv(f"{TAB}/consignado_prazo_max_mob_2023.csv", index=False)

cobertura = pd.DataFrame({"categoria": ks})
for T in TS:
    cobertura[f"T_{T}"] = [round(cov(k, T), 1) for k in ks]
cobertura.to_csv(f"{TAB}/consignado_cobertura_parcelas_mob_2023.csv", index=False)

# figura: 2 heatmaps
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 7))
M1 = prazo[[l for l, _ in CONFS]].to_numpy()
im1 = a1.imshow(np.clip(M1, 0, 60), aspect="auto", cmap="RdYlGn", vmin=0, vmax=60)
a1.set_xticks(range(len(CONFS))); a1.set_xticklabels(["95%", "90%", "85%", "80%"])
a1.set_yticks(range(len(ks))); a1.set_yticklabels(ks, fontsize=7)
a1.set_title("[MOB] Prazo máx. (meses) por confiança de seguir empregado")
a1.set_xlabel("confiança"); a1.set_ylabel("categoria")
for i in range(len(ks)):
    for j in range(len(CONFS)):
        v = M1[i, j]; a1.text(j, i, ("120+" if v > 120 else f"{v:.0f}"), ha="center", va="center", fontsize=6)
M2 = cobertura[[f"T_{T}" for T in TS]].to_numpy()
im2 = a2.imshow(M2, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
a2.set_xticks(range(len(TS))); a2.set_xticklabels([f"T={t}" for t in TS])
a2.set_yticks(range(len(ks))); a2.set_yticklabels(ks, fontsize=7)
a2.set_title("[MOB] Cobertura esperada de parcelas (% pagas em folha)")
a2.set_xlabel("prazo T (meses)"); a2.set_ylabel("categoria")
for i in range(len(ks)):
    for j in range(len(TS)):
        a2.text(j, i, f"{M2[i, j]:.0f}", ha="center", va="center", fontsize=6)
fig.tight_layout()
TMP = "/tmp/consignado_tabelas_mob_2023.png"
fig.savefig(TMP, dpi=130); os.makedirs(FIG, exist_ok=True)
shutil.copy(TMP, f"{FIG}/consignado_tabelas_mob_2023.png")

pd.set_option("display.width", 200)
print("=== [MOB] PRAZO MÁXIMO (meses) ==="); print(prazo.to_string(index=False))
print("\n=== [MOB] COBERTURA DE PARCELAS (%) ==="); print(cobertura.to_string(index=False))
print(f"\nFIM -> tabelas + {FIG}/consignado_tabelas_mob_2023.png")
