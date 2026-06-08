"""Helpers de visualização para a EDA (notebook 04) e validação (notebook 06)."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def save_fig(fig, path: Path, dpi: int = 120) -> Path:
    """Salva uma figura criando o diretório se preciso."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_rate_distribution(rates_series, titulo: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(rates_series, bins=40, ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel("Taxa anual de desligamento")
    ax.set_ylabel("Nº de células")
    return fig


def plot_marginal(df, dim: str, rate_col: str, titulo: str):
    """Risco médio (ponderado por exposição) por categoria de uma dimensão.

    Agrega numerador (taxa*n) e denominador (n) por categoria — robusto a
    células com poucos vínculos (a ponderação cuida do suporte estatístico).
    """
    tmp = df.copy()
    tmp["_num"] = tmp[rate_col] * tmp["n"]
    g = tmp.groupby(dim, observed=True).agg(_num=("_num", "sum"), _den=("n", "sum"))
    serie = (g["_num"] / g["_den"]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    serie.plot(kind="barh", ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel("Taxa anual média (ponderada)")
    return fig


def plot_heatmap(pivot, titulo: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(pivot, cmap="rocket_r", ax=ax)
    ax.set_title(titulo)
    return fig
