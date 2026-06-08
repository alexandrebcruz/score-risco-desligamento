"""Gráfico-resumo do tempo-até-desligamento por categoria de risco: caixa Q1-Q3,
mediana e média (estatísticas Weibull MONOTONIZADAS por isotônico), com as 5 PERSONAS
(grupos de risco) marcadas visualmente por cor das caixas + faixas de fundo + rótulos.

Eixo Y em ESCALA LOG (meses), pois os valores vão de ~3 a ~2150 meses.

Entrada: outputs/tables/sobrevivencia_weibull_estatisticas_mono_2023.csv
Saída:   outputs/figures/estatisticas_tempo_categorias_2023.png
"""
import os, shutil
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

S = "outputs/tables/sobrevivencia_weibull_estatisticas_mono_2023.csv"
OUT = "outputs/figures/estatisticas_tempo_categorias_2023.png"
TMP = "/tmp/estatisticas_tempo_categorias_2023.png"

# grupos de risco (personas) — mesmos do deck / HTML
GROUPS = [
    ("Risco Mínimo",      [1, 2],                   "#1a9850"),
    ("Risco Baixo",       [3, 4, 5, 6],             "#86cb66"),
    ("Risco Médio-Baixo", [7, 8, 9, 10, 11],        "#c9a227"),
    ("Risco Médio",       [12, 13, 14, 15, 16, 17], "#fb8d3d"),
    ("Risco Alto",        [18, 19, 20, 21, 22, 23], "#d73027"),
]
cat2grp = {c: (nome, cor) for nome, cats, cor in GROUPS for c in cats}

s = pd.read_csv(S).sort_values("categoria").reset_index(drop=True)
q1 = s["q1_meses_mono"].to_numpy()
med = s["mediana_meses_mono"].to_numpy()
mean = s["media_meses_mono"].to_numpy()
q3 = s["q3_meses_mono"].to_numpy()
ks = s["categoria"].to_numpy()

fig, ax = plt.subplots(figsize=(12.5, 6.6))

# faixas de fundo por persona + rótulo no topo
ytop = q3.max() * 1.9
for nome, cats, cor in GROUPS:
    lo, hi = min(cats) - 0.5, max(cats) + 0.5
    ax.axvspan(lo, hi, color=cor, alpha=0.07, zorder=0)
    ax.text((lo + hi) / 2, ytop, nome, ha="center", va="top", fontsize=9.5,
            weight="bold", color=cor)
    ax.axvline(hi, color="#ccc", lw=0.8, zorder=0)

# caixa Q1-Q3 + mediana + média por categoria
for i, k in enumerate(ks):
    cor = cat2grp[k][1]
    # caixa interquartil
    ax.bar(k, q3[i] - q1[i], bottom=q1[i], width=0.66, color=cor, alpha=0.45,
           edgecolor=cor, linewidth=1.1, zorder=2)
    # mediana (linha horizontal grossa)
    ax.hlines(med[i], k - 0.33, k + 0.33, color="#222", lw=2.2, zorder=4)
    # média (losango)
    ax.plot(k, mean[i], marker="D", markersize=6, color="white",
            markeredgecolor="#222", markeredgewidth=1.3, zorder=5)

ax.set_yscale("log")
ax.set_ylim(q1.min() * 0.7, ytop * 1.05)
ax.set_xlim(0.3, 23.7); ax.set_xticks(ks)
ax.set_xlabel("categoria de risco (1 = menor risco · 23 = maior)")
ax.set_ylabel("tempo até desligamento (meses, escala log)")
ax.set_title("Tempo até o desligamento por categoria — caixa = IQR (Q1–Q3), linha = mediana, losango = média\n"
             "(Weibull monotonizado por isotônico; cores = personas / grupos de risco)", fontsize=11)
ax.grid(axis="y", which="both", alpha=0.22)

# linhas de referência (12, 24, 36 meses = horizontes)
for ymo, lab in [(12, "12m"), (24, "24m"), (36, "36m")]:
    ax.axhline(ymo, color="#888", lw=0.8, ls=":", zorder=1)
    ax.text(0.45, ymo, lab, fontsize=8, color="#666", va="bottom")

leg = [Patch(facecolor="#999", alpha=0.45, edgecolor="#555", label="IQR (Q1–Q3)"),
       Line2D([0], [0], color="#222", lw=2.2, label="mediana"),
       Line2D([0], [0], marker="D", color="white", markeredgecolor="#222",
              markersize=7, lw=0, label="média")]
ax.legend(handles=leg, loc="upper right", framealpha=0.9, fontsize=9)

fig.tight_layout()
fig.savefig(TMP, dpi=130)
os.makedirs("outputs/figures", exist_ok=True)
shutil.copy(TMP, OUT)
print(f"FIM -> {OUT}")
