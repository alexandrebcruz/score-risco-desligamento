"""Figuras de apoio do MODELO NOVO (retreino 2021–2024) para os decks:
- calibração no out-of-time 2025 (decis previsto × observado);
- importância das 21 features (média A,B);
- AUC/KS por ano 2016–2025 (barras, papel de cada ano).

Saídas: outputs/figures/{calibracao_ensemble_2124.png, importancia_ensemble_2124.png,
        metricas_ano_2124.png}
Uso: MPLCONFIGDIR=/tmp/mpl /tmp/consig_venv/bin/python fig_modelo_2124.py
"""
import os, shutil
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ART = "outputs/runpod_retreino_2124"
FIG = "outputs/figures"; os.makedirs(FIG, exist_ok=True)

def salva(fig, nome):
    tmp = f"/tmp/{nome}"
    fig.savefig(tmp, dpi=130)
    shutil.copy(tmp, f"{FIG}/{nome}"); plt.close(fig)
    print("ok", nome)

# --- calibração (2025, out-of-time) ---
cal = pd.read_csv(f"{ART}/calibracao_2025.csv")
fig, ax = plt.subplots(figsize=(6.2, 5.4))
lim = max(cal.prevista.max(), cal.observada.max()) * 1.08
ax.plot([0, lim], [0, lim], ls="--", color="#999", label="calibração perfeita")
ax.plot(cal.prevista, cal.observada, marker="o", color="#2c5f9e", label="ensemble 21–24")
for _, r in cal.iterrows():
    ax.annotate(f"{r.n/1e6:.1f}M", (r.prevista, r.observada), fontsize=7, color="#888",
                xytext=(4, -8), textcoords="offset points")
ax.set_xlabel("Risco previsto (média do decil)"); ax.set_ylabel("Risco observado (freq. real)")
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
ax.set_title("Calibração — ensemble 2021–24 (out-of-time 2025)")
ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); salva(fig, "calibracao_ensemble_2124.png")

# --- importância ---
imp = pd.read_csv(f"{ART}/importancia_ensemble.csv").sort_values("imp_ensemble")
fig, ax = plt.subplots(figsize=(7.2, 6.4))
ax.barh(imp.feature, imp.imp_ensemble, color="#2e9e5b")
for i, (f, v) in enumerate(zip(imp.feature, imp.imp_ensemble)):
    ax.text(v + .15, i, f"{v:.1f}%", va="center", fontsize=8, color="#444")
ax.set_xlabel("Importância (%)"); ax.set_title("Feature importance — ensemble 2021–24 (21 features)")
ax.grid(axis="x", alpha=.3)
fig.tight_layout(); salva(fig, "importancia_ensemble_2124.png")

# --- AUC/KS por ano ---
m = pd.read_csv(f"{ART}/metricas_por_ano.csv")
e = m[m.modelo == "ensemble"].sort_values("ano")
fig, ax = plt.subplots(figsize=(9.6, 5.2))
x = np.arange(len(e))
cores = ["#5b8db8" if p == "out_of_sample" else "#2c5f9e" for p in e.papel]
ax.bar(x - .2, e.AUC, width=.38, color=cores, label="AUC")
ax.bar(x + .2, e.KS, width=.38, color="#f4a722", label="KS")
for i, (a, k) in enumerate(zip(e.AUC, e.KS)):
    ax.text(i - .2, a + .008, f"{a:.3f}", ha="center", fontsize=7.5)
    ax.text(i + .2, k + .008, f"{k:.3f}", ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(e.ano)
ax.set_ylim(0, 1.0); ax.set_title("AUC e KS do ensemble por ano — azul-claro = fora do treino (out-of-sample)")
ax.legend(loc="upper right"); ax.grid(axis="y", alpha=.3)
for i, p in enumerate(e.papel):
    if p == "out_of_sample":
        ax.text(i, .02, "OOS", ha="center", fontsize=7, color="#333")
fig.tight_layout(); salva(fig, "metricas_ano_2124.png")
print("FIM_FIG_2124")
