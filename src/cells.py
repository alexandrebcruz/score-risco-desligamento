"""Construção das células de perfil e da hierarquia de backoff.

A célula mais específica combina CBO×CNAE (em níveis) × tempo × tamanho × UF
× idade × escolaridade. Quando uma célula é pouco populosa, recuamos
(backoff) para níveis progressivamente mais agregados.

A hierarquia é definida UMA vez aqui (BACKOFF_LEVELS) e reutilizada tanto no
cálculo das taxas (src/rates.py) quanto no scoring (src/scoring.py), garantindo
consistência total entre treino e uso.
"""
from __future__ import annotations

import pandas as pd

# Mapa UF -> grande região (usado no backoff geográfico).
UF_REGIAO = {
    "AC": "N", "AP": "N", "AM": "N", "PA": "N", "RO": "N", "RR": "N", "TO": "N",
    "AL": "NE", "BA": "NE", "CE": "NE", "MA": "NE", "PB": "NE", "PE": "NE",
    "PI": "NE", "RN": "NE", "SE": "NE",
    "DF": "CO", "GO": "CO", "MT": "CO", "MS": "CO",
    "ES": "SE", "MG": "SE", "RJ": "SE", "SP": "SE",
    "PR": "S", "RS": "S", "SC": "S",
}

# Ordem do MAIS GERAL para o MAIS ESPECÍFICO.
# O scoring percorre nesta ordem, aplicando shrinkage Empirical Bayes aninhado:
# começa na taxa global e, a cada nível encontrado, refina a estimativa.
# Cada nível é uma lista de colunas-chave de agrupamento.
BACKOFF_LEVELS: list[dict] = [
    {"name": "global",                      "cols": []},
    {"name": "cbo2",                        "cols": ["cbo2"]},
    {"name": "cbo2_cnae2_regiao",           "cols": ["cbo2", "cnae2", "regiao"]},
    {"name": "cbo2_cnae2_tempo_uf",         "cols": ["cbo2", "cnae2", "tempo_faixa", "uf"]},
    {"name": "cbo4_cnae2_tempo_tam_uf",     "cols": ["cbo4", "cnae2", "tempo_faixa", "tamanho_faixa", "uf"]},
    {"name": "cbo4_cnae4_tempo_tam_uf",     "cols": ["cbo4", "cnae4", "tempo_faixa", "tamanho_faixa", "uf"]},
    {"name": "completo",                    "cols": ["cbo4", "cnae4", "tempo_faixa", "tamanho_faixa",
                                                      "uf", "idade_faixa", "escol_faixa"]},
]

# Todas as colunas-chave que precisam existir no dataframe de células.
ALL_KEY_COLS = sorted({c for lvl in BACKOFF_LEVELS for c in lvl["cols"]})


def active_levels(cfg: dict | None = None) -> list[dict]:
    """Retorna os níveis de backoff ativos, excluindo os listados em config.

    `cfg['suavizacao']['niveis_excluidos']` permite remover níveis ultra-granulares
    (ex.: na escala nacional). O nível 'global' nunca é excluído (prior base do EB).
    """
    excl = set((cfg or {}).get("suavizacao", {}).get("niveis_excluidos", []))
    excl.discard("global")
    return [lvl for lvl in BACKOFF_LEVELS if lvl["name"] not in excl]


def add_cell_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva as colunas usadas como chaves de célula a partir do df com bins.

    Pré-requisito: df já passou por binning.add_bins (tem tempo_faixa, etc.).
    Cria: cbo4, cbo2, cnae4, cnae2, regiao.

    Harmoniza os códigos entre layouts de ano (o de 2023 perde o zero-padding e
    recodifica alguns campos) ANTES de fatiar — senão CBOs do grupo militar
    ('010305' vs '10305') gerariam cbo2 inconsistente entre anos.
    """
    from .cleaning import normalize_short_codes
    out = normalize_short_codes(df.copy())   # remap '999'->'99' + strip de zeros (defensivo)
    # zfill consistente e ESCRITO DE VOLTA na base (não só nas derivadas).
    out["cbo"] = out["cbo"].astype(str).str.zfill(6)
    out["cnae"] = out["cnae"].astype(str).str.zfill(7)
    cbo, cnae = out["cbo"], out["cnae"]
    out["cbo4"] = cbo.str[:4]
    out["cbo2"] = cbo.str[:2]
    out["cnae4"] = cnae.str[:4]
    out["cnae2"] = cnae.str[:2]
    out["regiao"] = out["uf"].map(UF_REGIAO).fillna("NA")
    return out


def person_keys(attrs: dict) -> dict:
    """Deriva as chaves de célula (cbo4, cbo2, cnae4, cnae2, regiao) de uma pessoa.

    `attrs` já deve conter as faixas (tempo_faixa, idade_faixa, escol_faixa,
    tamanho_faixa, uf) — produzidas pelo scoring a partir dos atributos brutos.
    """
    cbo = str(attrs["cbo"]).zfill(6)
    cnae = str(attrs["cnae"]).zfill(7)
    keys = dict(attrs)
    keys["cbo"], keys["cnae"] = cbo, cnae   # base normalizada de volta (simetria c/ add_cell_keys)
    keys["cbo4"] = cbo[:4]
    keys["cbo2"] = cbo[:2]
    keys["cnae4"] = cnae[:4]
    keys["cnae2"] = cnae[:2]
    keys["regiao"] = UF_REGIAO.get(str(attrs.get("uf", "")).upper(), "NA")
    return keys
