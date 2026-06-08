"""Componente conjuntural (recente) a partir do Novo CAGED.

A RAIS dá o risco *estrutural* (perfil detalhado, mas defasado/anual). O Novo
CAGED é um fluxo mensal recente — porém **sem estoque e sem tempo de vínculo**.
Por isso não substitui a célula completa da RAIS; ele **recalibra o nível geral**
do risco para a conjuntura mais recente.

Estratégia (fator de ajuste conjuntural):
1. RAIS (estrutural): hazard anual por nível agregado L = cbo2 × cnae2 × UF.
2. CAGED (recente): desligamentos involuntários por L nos últimos P meses;
   hazard mensal recente = desligamentos / (P × estoque_RAIS_L); anualizado ×12.
3. fator_L = hazard_recente_L / hazard_estrutural_L  (suavizado).
4. No scoring, risco_ajustado(célula) = risco_estrutural(célula) × fator_L.

Assim a granularidade fina vem da RAIS e o "termômetro" recente vem do CAGED.

Layout do Novo CAGED (CAGEDMOV*.txt): separador ';', **UTF-8**, decimal vírgula
(difere da RAIS, que é latin-1). O motivo do desligamento vem em
`tipomovimentação`; `saldomovimentação` = ±1.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .cleaning import _norm_col, _detect_sep, MAPA_UF_IBGE

# tipomovimentação (Novo CAGED) -> categoria unificada (apenas desligamentos).
# Admissões e transferências não interessam ao numerador de risco.
MAPA_TIPOMOV_CAGED = {
    31: "involuntario_sjc",   # desligamento por demissão sem justa causa
    32: "involuntario_cjc",   # demissão com justa causa
    33: "outros",             # culpa recíproca
    40: "pedido_demissao",    # a pedido
    43: "fim_contrato",       # término de contrato por prazo determinado
    45: "fim_contrato",       # término
    50: "aposent_morte",      # aposentadoria
    60: "aposent_morte",      # morte
    70: "outros",             # transferência (saída)
    80: "outros",
    90: "outros",             # acordo entre as partes
    98: "outros", 99: "outros",
}

# Casamento de colunas por tokens sobre o nome normalizado (robusto a acentos).
CAGED_FIELD_TOKENS = {
    "competencia":    [["competenciamov"]],
    "uf":             [["uf"]],
    "municipio":      [["municipio"]],
    "cbo":            [["cbo2002ocupacao"], ["cbo"]],
    "cnae":           [["subclasse"]],
    "idade":          [["idade"]],
    "grau_instrucao": [["graudeinstrucao"], ["grau", "instrucao"]],
    "tamanho_estab":  [["tamestabjan"], ["tamanho"]],
    "saldo":          [["saldomovimentacao"]],
    "tipomov":        [["tipomovimentacao"]],
}
CAGED_EXACT = {"uf": "uf", "idade": "idade", "municipio": "municipio",
               "saldo": "saldomovimentacao", "tipomov": "tipomovimentacao",
               "competencia": "competenciamov"}


def _resolve_caged_columns(header: list[str]) -> dict:
    norm = {c: _norm_col(c) for c in header}
    achado = {}
    for logico, predicados in CAGED_FIELD_TOKENS.items():
        alvo = CAGED_EXACT.get(logico)
        for real, n in norm.items():
            if alvo is not None:
                if n == alvo:
                    achado[logico] = real
                    break
                continue
            if any(all(tok in n for tok in pred) for pred in predicados):
                achado[logico] = real
                break
    faltam = set(CAGED_FIELD_TOKENS) - set(achado)
    if faltam:
        raise KeyError(f"Colunas CAGED não resolvidas: {faltam}")
    return {real: logico for logico, real in achado.items()}


def iter_caged_deslig_chunks(path, ufs_subset: list[str] | None = None,
                             chunksize: int = 1_000_000):
    """Itera um CAGEDMOV*.txt em chunks, devolvendo só DESLIGAMENTOS canônicos.

    Colunas: uf, cbo (6d), cnae (7d), motivo_unificado, competencia.
    """
    sep = _detect_sep(path)
    header = pd.read_csv(path, sep=sep, encoding="utf-8", nrows=0).columns.tolist()
    ren = _resolve_caged_columns(header)
    reader = pd.read_csv(path, sep=sep, encoding="utf-8", dtype=str,
                         usecols=list(ren.keys()), chunksize=chunksize)
    for chunk in reader:
        c = chunk.rename(columns=ren)
        saldo = pd.to_numeric(c["saldo"].str.replace(",", ".", regex=False), errors="coerce")
        c = c[saldo == -1].copy()                      # apenas desligamentos
        tipo = pd.to_numeric(c["tipomov"], errors="coerce").astype("Int64")
        c["motivo_unificado"] = tipo.map(MAPA_TIPOMOV_CAGED).fillna("outros")
        # UF: o CAGED traz o código numérico da UF; mapeia para sigla.
        c["uf"] = c["uf"].astype(str).str.zfill(2).map(MAPA_UF_IBGE).fillna("NI")
        if ufs_subset:
            c = c[c["uf"].isin(ufs_subset)]
        c["cbo"] = c["cbo"].astype(str).str.strip().str.zfill(6)
        c["cnae"] = c["cnae"].astype(str).str.strip().str.zfill(7)
        c["cbo2"] = c["cbo"].str[:2]
        c["cnae2"] = c["cnae"].str[:2]
        yield c[["uf", "cbo2", "cnae2", "motivo_unificado", "competencia"]]


def count_caged_deslig(paths, motivos: list[str], ufs_subset=None) -> tuple[pd.DataFrame, int]:
    """Conta desligamentos do CAGED por nível L=(cbo2,cnae2,uf) e motivo.

    Retorna (tabela, n_meses_distintos). Tabela: cbo2,cnae2,uf + k_<motivo>.
    """
    cols = ["cbo2", "cnae2", "uf"]
    acc = None
    meses = set()
    for p in paths:
        for ch in iter_caged_deslig_chunks(p, ufs_subset=ufs_subset):
            meses.update(ch["competencia"].unique().tolist())
            d = pd.get_dummies(ch["motivo_unificado"])
            for m in motivos:
                if m not in d.columns:
                    d[m] = 0
            g = pd.concat([ch[cols], d[motivos]], axis=1).groupby(cols, observed=True,
                                                                  as_index=False)[motivos].sum()
            g = g.rename(columns={m: f"k_{m}" for m in motivos})
            acc = g if acc is None else (pd.concat([acc, g], ignore_index=True)
                                         .groupby(cols, observed=True, as_index=False).sum())
    return (acc if acc is not None else pd.DataFrame(columns=cols)), len(meses)


def rais_estoque_por_L(level_tab: pd.DataFrame, motivos: list[str]) -> pd.DataFrame:
    """Agrega uma tabela RAIS de nível para L=(cbo2,cnae2,uf).

    Aceita qualquer nível que permita derivar (cbo2, cnae2, uf): usa as colunas
    cbo2/cnae2 se presentes, senão deriva de cbo4/cnae4. Soma `n` e `k_*`.
    """
    df = level_tab.copy()
    if "cbo2" not in df.columns:
        df["cbo2"] = df["cbo4"].astype(str).str[:2]
    if "cnae2" not in df.columns:
        src = "cnae4" if "cnae4" in df.columns else "cnae"
        df["cnae2"] = df[src].astype(str).str[:2]
    val = ["n"] + [f"k_{m}" for m in motivos]
    return df.groupby(["cbo2", "cnae2", "uf"], observed=True, as_index=False)[val].sum()


def fator_ajuste_conjuntural(caged_tab: pd.DataFrame, n_meses: int,
                             rais_estoque: pd.DataFrame, motivo: str,
                             m_suav: float = 200.0, n_anos_rais: int = 1) -> pd.DataFrame:
    """Calcula o fator conjuntural por L = (cbo2,cnae2,uf) para um motivo.

    `rais_estoque`: colunas cbo2,cnae2,uf, n (exposição RAIS SOMADA sobre os anos),
    k_<motivo>. `n_anos_rais` é o nº de anos somados em `n` — usado para converter a
    exposição acumulada em estoque ANUAL médio (denominador do hazard recente).
    Retorna L + colunas: hazard_estrut, hazard_recente, fator (suavizado p/ 1).
    """
    cols = ["cbo2", "cnae2", "uf"]
    df = rais_estoque.merge(caged_tab, on=cols, how="left", suffixes=("_rais", "_caged"))
    kc = df.get(f"k_{motivo}_caged", df.get(f"k_{motivo}"))
    df["desl_caged"] = pd.to_numeric(kc, errors="coerce").fillna(0.0)
    # hazard estrutural anual: k e n ambos somados sobre os anos -> razão é taxa anual.
    df["hazard_estrut"] = df[f"k_{motivo}_rais"] / df["n"]
    # hazard recente anualizado: fluxo CAGED anualizado / ESTOQUE ANUAL médio (n/anos).
    meses = max(n_meses, 1)
    estoque_anual = df["n"] / max(n_anos_rais, 1)
    df["hazard_recente"] = (df["desl_caged"] / meses) * 12.0 / estoque_anual
    # fator bruto e suavização: encolhe para 1 quando a célula tem pouca exposição
    bruto = df["hazard_recente"] / df["hazard_estrut"].replace(0, np.nan)
    w = df["n"] / (df["n"] + m_suav)
    df["fator"] = (1 - w) * 1.0 + w * bruto.fillna(1.0)
    df["fator"] = df["fator"].clip(lower=0.2, upper=5.0).fillna(1.0)
    return df[cols + ["hazard_estrut", "hazard_recente", "fator", "n"]]
