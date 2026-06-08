"""Discretização (binning) das dimensões contínuas/ordinais.

CRÍTICO: as MESMAS funções de binning são usadas no cálculo das taxas
(notebook 03) e no scoring (src/scoring.py). Isso evita *train/serve skew* —
a pessoa consultada cai exatamente na mesma faixa usada no treino.

As bordas vêm de config.yaml (chave `binning`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config


def _faixas_labels(bordas: list[int], unidade: str) -> list[str]:
    """Gera rótulos legíveis a partir de bordas superiores.

    Ex.: bordas [3,6,12] -> ['<3{u}', '3-6{u}', '6-12{u}', '12+{u}'].
    """
    labels = [f"<{bordas[0]}{unidade}"]
    for a, b in zip(bordas[:-1], bordas[1:]):
        labels.append(f"{a}-{b}{unidade}")
    labels.append(f"{bordas[-1]}+{unidade}")
    return labels


def _cut(valores, bordas: list[int], unidade: str) -> pd.Categorical:
    """Aplica corte por bordas superiores, com faixa final aberta (+inf)."""
    edges = [-np.inf, *bordas, np.inf]
    labels = _faixas_labels(bordas, unidade)
    return pd.cut(pd.to_numeric(valores, errors="coerce"),
                  bins=edges, labels=labels, right=True, include_lowest=True)


def bin_tempo_vinculo(meses, cfg: dict | None = None) -> pd.Categorical:
    cfg = cfg or load_config()
    return _cut(meses, cfg["binning"]["tempo_vinculo_meses"], "m")


def bin_idade(anos, cfg: dict | None = None) -> pd.Categorical:
    cfg = cfg or load_config()
    return _cut(anos, cfg["binning"]["idade_anos"], "a")


def bin_tamanho(tamanho, cfg: dict | None = None) -> pd.Categorical:
    """Tamanho do estabelecimento.

    No schema sintético `tamanho_estab` já é um código de faixa (0..5); apenas
    o convertemos para rótulo estável. Para dados RAIS reais (nº de empregados),
    use as bordas de config `binning.tamanho_estab`.
    """
    cfg = cfg or load_config()
    s = pd.Series(pd.to_numeric(pd.Series(tamanho), errors="coerce"))
    # Heurística: se já parece código de faixa pequeno (<=10), usa direto.
    if s.dropna().le(10).mean() > 0.9:
        return s.astype("Int64").astype(str).radd("faixa_")
    return _cut(s, cfg["binning"]["tamanho_estab"], "emp")


def bin_escolaridade(escolaridade) -> pd.Series:
    """Escolaridade já vem canônica de cleaning.py; passa adiante como categoria."""
    return pd.Series(escolaridade).astype(str)


def add_bins(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Adiciona as colunas de faixa ao dataframe canônico.

    Cria: tempo_faixa, idade_faixa, escol_faixa, tamanho_faixa.
    """
    cfg = cfg or load_config()
    out = df.copy()
    out["tempo_faixa"] = bin_tempo_vinculo(out["tempo_vinculo_meses"], cfg).astype(str)
    out["idade_faixa"] = bin_idade(out["idade"], cfg).astype(str)
    out["escol_faixa"] = bin_escolaridade(out["escolaridade"]).values
    out["tamanho_faixa"] = bin_tamanho(out["tamanho_estab"], cfg).astype(str)
    return out
