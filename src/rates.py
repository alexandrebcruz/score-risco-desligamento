"""Cálculo das taxas de desligamento por célula, suavização e horizonte.

Conceitos:
- Para cada nível da hierarquia de backoff (cells.BACKOFF_LEVELS) construímos
  uma tabela com a exposição `n` (denominador de risco) e, por motivo, a
  contagem de desligamentos `k_<motivo>`.
- A taxa anual (hazard) de um motivo numa célula é k/n.
- Suavização Empirical Bayes ANINHADA: a estimativa parte da taxa global e é
  refinada nível a nível com shrinkage Beta-Binomial:
        p_hat = (k + m * p_prior) / (n + m)
  onde p_prior é a estimativa do nível imediatamente mais geral e m é a força
  do shrinkage (config: suavizacao.shrinkage_m). Isso resolve células pouco
  populosas sem descartá-las.
- Conversão de horizonte: do hazard anual para janelas de H meses, assumindo
  hazard mensal aproximadamente constante:
        h_mensal = 1 - (1 - p_anual) ** (1/12)
        p_H      = 1 - (1 - h_mensal) ** H
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from .config import load_config
from .cells import BACKOFF_LEVELS


def build_level_tables(df_cells: pd.DataFrame, motivos: list[str],
                       levels: list[dict] | None = None) -> dict[str, pd.DataFrame]:
    """Constrói uma tabela de exposição+contagens por motivo para cada nível.

    `df_cells` é o dataframe canônico já com bins e chaves de célula, contendo
    uma linha por vínculo (exposição) e a coluna `motivo_unificado`.
    `levels` permite restringir os níveis calculados (default: todos). Útil para
    escala nacional, onde os níveis mais granulares são grandes demais.
    Retorna {nome_nivel: DataFrame} com colunas-chave + `n` + `k_<motivo>`.
    """
    levels = levels or BACKOFF_LEVELS
    tables: dict[str, pd.DataFrame] = {}
    # one-hot dos motivos para somar contagens por célula numa única passada.
    dummies = pd.get_dummies(df_cells["motivo_unificado"])
    for m in motivos:
        if m not in dummies.columns:
            dummies[m] = 0
    base = pd.concat([df_cells, dummies[motivos]], axis=1)

    for lvl in levels:
        cols = lvl["cols"]
        agg = {m: "sum" for m in motivos}
        if cols:
            g = base.groupby(cols, observed=True)
            tab = g.agg(n=("motivo_unificado", "size"), **{f"k_{m}": (m, "sum") for m in motivos})
            tab = tab.reset_index()
        else:  # nível global -> uma única linha
            row = {"n": len(base)}
            for m in motivos:
                row[f"k_{m}"] = int(base[m].sum())
            tab = pd.DataFrame([row])
        tables[lvl["name"]] = tab
    return tables


def _sum_two_tables(a: pd.DataFrame | None, b: pd.DataFrame | None,
                    cols: list[str], motivos: list[str]) -> pd.DataFrame:
    """Soma duas tabelas do MESMO nível (mesmo conjunto de colunas-chave).

    Concatena e reagrupa por chave somando exposição `n` e contagens `k_*`.
    Núcleo do map-reduce: permite agregar arquivos/chunks incrementalmente.
    """
    if a is None:
        return b
    if b is None:
        return a
    cat = pd.concat([a, b], ignore_index=True)
    val_cols = ["n"] + [f"k_{m}" for m in motivos]
    if cols:
        return cat.groupby(cols, observed=True, as_index=False)[val_cols].sum()
    return pd.DataFrame([{c: int(cat[c].sum()) for c in val_cols}])


class Accumulator:
    """Acumulador incremental de tabelas de taxa por nível de backoff.

    Uso: instanciar, chamar `.add(df_cells)` para cada chunk/arquivo (df já com
    bins e chaves de célula), e ao final `.tables()` devolve as tabelas somadas.
    Mantém memória baixa: o estado é sempre a agregação reduzida, nunca os
    microdados.
    """

    def __init__(self, motivos: list[str], levels: list[dict] | None = None):
        self.motivos = motivos
        self.active_levels = levels or BACKOFF_LEVELS
        self.levels: dict[str, pd.DataFrame | None] = {lvl["name"]: None for lvl in self.active_levels}

    def add(self, df_cells: pd.DataFrame) -> "Accumulator":
        parciais = build_level_tables(df_cells, self.motivos, self.active_levels)
        for lvl in self.active_levels:
            self.levels[lvl["name"]] = _sum_two_tables(
                self.levels[lvl["name"]], parciais[lvl["name"]], lvl["cols"], self.motivos)
        return self

    def tables(self) -> dict[str, pd.DataFrame]:
        # garante que níveis nunca vistos existam (vazios) — robustez
        out = {}
        for lvl in self.active_levels:
            t = self.levels[lvl["name"]]
            out[lvl["name"]] = t if t is not None else build_level_tables(
                pd.DataFrame(columns=["motivo_unificado"]), self.motivos, [lvl])[lvl["name"]]
        return out


def count_single_level(df_cells: pd.DataFrame, cols: list[str],
                       motivos: list[str]) -> pd.DataFrame:
    """Agrega UM único nível (lista de colunas) — usado p/ validação por ano."""
    dummies = pd.get_dummies(df_cells["motivo_unificado"])
    for m in motivos:
        if m not in dummies.columns:
            dummies[m] = 0
    base = pd.concat([df_cells[cols], dummies[motivos]], axis=1)
    agg = base.groupby(cols, observed=True, as_index=False).agg(
        n=(cols[0], "size"), **{f"k_{m}": (m, "sum") for m in motivos})
    return agg


def save_level_tables(tables: dict[str, pd.DataFrame], motivos: list[str],
                      out_dir: Path) -> None:
    """Persiste as tabelas de nível em parquet + um levels.json com metadados.

    O `levels.json` registra APENAS os níveis presentes em `tables` (na ordem
    canônica de BACKOFF_LEVELS), de modo que o scoring use exatamente os níveis
    materializados — permite excluir níveis ultra-granulares na escala nacional.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols_por_nivel = {lvl["name"]: lvl["cols"] for lvl in BACKOFF_LEVELS}
    order = [lvl["name"] for lvl in BACKOFF_LEVELS if lvl["name"] in tables]
    meta = {"order": order,
            "cols": {name: cols_por_nivel[name] for name in order},
            "motivos": motivos}
    for name, tab in tables.items():
        tab.to_parquet(out_dir / f"level_{name}.parquet", index=False)
    with open(out_dir / "levels.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)


def load_level_tables(out_dir: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Carrega as tabelas de nível e o levels.json (usado pelo scoring)."""
    out_dir = Path(out_dir)
    with open(out_dir / "levels.json", "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    tables = {name: pd.read_parquet(out_dir / f"level_{name}.parquet")
              for name in meta["order"]}
    return tables, meta


def _index_table(tab: pd.DataFrame, cols: list[str], motivos: list[str]) -> dict:
    """Indexa uma tabela de nível de forma COMPACTA (eficiente em memória).

    Em vez de um dict de dicts (custoso para milhões de células na escala
    nacional), guarda um keymap {chave_str -> posição} + arrays numpy de `n` e
    de `k_<motivo>`. A chave é a concatenação das colunas com '|'.
    """
    n_arr = tab["n"].to_numpy(dtype="float64")
    k_arr = {m: tab[f"k_{m}"].to_numpy(dtype="float64") for m in motivos}
    if not cols:
        keymap = {"": 0}
    else:
        keycols = [tab[c].astype(str).to_numpy() for c in cols]
        keymap = {"|".join(kc[i] for kc in keycols): i for i in range(len(tab))}
    return {"keymap": keymap, "n": n_arr, "k": k_arr}


def build_indexes(tables: dict[str, pd.DataFrame], meta: dict) -> dict[str, dict]:
    """Pré-indexa todos os níveis para lookup O(1) e leve no scoring."""
    return {name: _index_table(tables[name], meta["cols"][name], meta["motivos"])
            for name in meta["order"]}


def eb_annual_hazard(indexes: dict[str, dict], meta: dict, keys: dict,
                     motivo: str, m: float) -> tuple[float, str, int]:
    """Hazard anual suavizado (Empirical Bayes aninhado) para um motivo.

    Percorre os níveis do mais geral ao mais específico. A taxa global é o
    prior inicial; cada nível encontrado refina via shrinkage Beta-Binomial.
    Retorna (hazard_anual, nivel_efetivo, exposicao_no_nivel_efetivo).
    """
    p = 0.0
    nivel_efetivo = "global"
    exposicao = 0
    for name in meta["order"]:
        cols = meta["cols"][name]
        key = "|".join(str(keys[c]) for c in cols) if cols else ""
        idxobj = indexes[name]
        i = idxobj["keymap"].get(key)
        if i is None:
            continue
        n = idxobj["n"][i]
        k = idxobj["k"][motivo][i]
        if n <= 0:
            continue
        if name == "global":
            p = k / n            # prior base
        else:
            p = (k + m * p) / (n + m)  # refino EB usando o nível anterior como prior
        nivel_efetivo = name
        exposicao = int(n)
    return p, nivel_efetivo, exposicao


def horizon_risk(hazard_anual: float, horizonte_meses: int) -> float:
    """Converte hazard anual em risco numa janela de H meses."""
    hazard_anual = min(max(hazard_anual, 0.0), 0.999999)
    h_mensal = 1.0 - (1.0 - hazard_anual) ** (1.0 / 12.0)
    return 1.0 - (1.0 - h_mensal) ** horizonte_meses


def beta_ci(k: float, n: float, p: float, m: float, z: float = 1.96) -> tuple[float, float]:
    """Intervalo aproximado (Wald sobre a posterior Beta) para a taxa anual.

    Usa contagens efetivas pós-shrinkage (n+m) como tamanho de amostra.
    Apenas indicativo da incerteza — não é um IC exato.
    """
    n_eff = n + m
    if n_eff <= 0:
        return (0.0, 0.0)
    se = np.sqrt(max(p * (1 - p), 1e-9) / n_eff)
    return (max(0.0, p - z * se), min(1.0, p + z * se))
