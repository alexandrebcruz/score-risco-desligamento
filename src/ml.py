"""Modelo supervisionado (CatBoost) como ALTERNATIVA/benchmark ao score agregado.

Premissa: a RAIS é um painel implícito (cada vínculo tem o desfecho), então dá
para treinar um classificador individual. Aqui:
- amostra os microdados (cabe em memória),
- treina um CatBoost com split TEMPORAL (treino < holdout),
- compara com o score de células (rates) nas MESMAS métricas e conjunto de teste.

Features: categóricas de alta cardinalidade tratadas nativamente pelo CatBoost.
NUNCA usar como feature: vinculo_ativo, mes_deslig, motivo_desligamento, motivo_unificado
(são o desfecho — vazamento).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import load_config
from . import binning, cells, rates

# novas features (existem em todos os anos 2020-2023)
EXTRA_CAT = ["tipo_vinculo", "natureza_juridica", "natureza_setor",
             "intermitente", "simples"]
# Ordinais (a ORDEM do código tem significado) -> NUMÉRICAS; 99=ignorado -> -1.
ORD = ["escolaridade", "tamanho_estab", "faixa_remuneracao", "faixa_horas"]
# Colunas lidas direto dos microdados (interim)
COLS_RAW = (["cbo", "cnae", "uf",
             "idade", "tempo_vinculo_meses", "qtd_dias_afastamento"] + ORD + EXTRA_CAT)

# Prefixos hierárquicos derivados (agregadores naturais dos códigos):
#  CBO 2002 (6 díg): 1=grande grupo, 2=subgrupo principal, 4=família.
#  CNAE 2.0 (7 díg): 2=divisão, 3=grupo, 5=classe.
CBO_NIVEIS = {"cbo1": 1, "cbo2": 2, "cbo4": 4}
CNAE_NIVEIS = {"cnae2": 2, "cnae3": 3, "cnae5": 5}

# Features do modelo: código completo + prefixos agregadores + demais categóricas.
FEATURES_CAT = (["cbo", "cbo4", "cbo2", "cbo1", "cnae", "cnae5", "cnae3", "cnae2"]
                + ["uf"] + EXTRA_CAT)
FEATURES_NUM = ["idade", "tempo_vinculo_meses", "qtd_dias_afastamento"] + ORD
FEATURES = FEATURES_CAT + FEATURES_NUM


def add_hier_features(X: pd.DataFrame) -> pd.DataFrame:
    """Adiciona os prefixos hierárquicos de CBO e CNAE como features categóricas."""
    cbo = X["cbo"].astype(str).str.zfill(6)
    cnae = X["cnae"].astype(str).str.zfill(7)
    for nome, k in CBO_NIVEIS.items():
        X[nome] = cbo.str[:k]
    for nome, k in CNAE_NIVEIS.items():
        X[nome] = cnae.str[:k]
    return X


def sample_microdata(anos: list[int], frac: float = 0.10, seed: int = 42,
                     motivo: str = "involuntario_sjc", cfg: dict | None = None) -> pd.DataFrame:
    """Lê as partições interim dos anos pedidos e amostra uma fração de cada.

    Retorna um DataFrame com FEATURES + 'ano' + 'y' (target binário = teve o
    `motivo` de desligamento no ano). Amostragem por partição mantém memória baixa.
    """
    cfg = cfg or load_config()
    interim = cfg["abs"]["interim"]
    cols = COLS_RAW + ["motivo_unificado"]
    partes = []
    for ano in anos:
        for f in sorted((interim / "rais" / f"ano={ano}").glob("*.parquet")):
            df = pd.read_parquet(f, columns=cols)
            if frac < 1.0:
                df = df.sample(frac=frac, random_state=seed)
            df["ano"] = ano
            df["y"] = (df["motivo_unificado"] == motivo).astype(int)
            partes.append(df.drop(columns=["motivo_unificado"]))
    out = pd.concat(partes, ignore_index=True)
    return out


def prepare_xy(df: pd.DataFrame):
    """Separa X (features) e y, derivando os prefixos hierárquicos.

    Categóricas como string (sem NaN); numéricas com sentinela -1 para faltantes.
    """
    X = df[COLS_RAW].copy()
    X = add_hier_features(X)                 # cria cbo1/2/4 e cnae2/3/5
    X = X[FEATURES].copy()
    for c in FEATURES_CAT:
        X[c] = X[c].astype(str).fillna("NA")
    for c in FEATURES_NUM:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(-1.0)
    for c in ("faixa_remuneracao", "faixa_horas"):
        X.loc[X[c] == 99, c] = -1.0              # 99 = ignorado -> sentinela
    y = df["y"].astype(int).values
    return X, y


def train_catboost(X_tr, y_tr, X_val=None, y_val=None,
                   early_stopping_rounds=50, **kwargs):
    """Treina um CatBoostClassifier com as categóricas nativas.

    Se `X_val`/`y_val` forem dados, ativa EARLY STOPPING nesse conjunto de
    validação (que deve ser DISTINTO do holdout de avaliação, p/ não vazar):
    o nº de árvores passa a ser escolhido pela validação (use_best_model).
    """
    from catboost import CatBoostClassifier, Pool

    has_val = X_val is not None
    params = dict(iterations=3000 if has_val else 400, depth=8, learning_rate=0.1,
                  loss_function="Logloss", eval_metric="AUC",
                  random_seed=42, verbose=100,
                  # aceleradores p/ muitas categóricas de alta cardinalidade:
                  thread_count=-1,          # usa todos os cores
                  boosting_type="Plain",    # mais rápido que Ordered em datasets grandes
                  max_ctr_complexity=1,     # não combina pares de categóricas (caro)
                  bootstrap_type="Bernoulli", subsample=0.7)
    if has_val:                              # só faz sentido com conjunto de validação
        params.update(early_stopping_rounds=early_stopping_rounds,
                      use_best_model=True, od_type="Iter")
    params.update(kwargs)
    model = CatBoostClassifier(cat_features=FEATURES_CAT, **params)
    eval_set = Pool(X_val, y_val, cat_features=FEATURES_CAT) if has_val else None
    model.fit(X_tr, y_tr, eval_set=eval_set)
    return model


def eval_scores(y_true, p_pred) -> dict:
    """AUC, Brier e log loss (métricas comparáveis entre modelos)."""
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

    p = np.clip(np.asarray(p_pred, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "AUC": float(roc_auc_score(y_true, p)),
        "Brier": float(brier_score_loss(y_true, p)),
        "LogLoss": float(log_loss(y_true, p)),
    }


def build_train_only_rates(cfg, holdout_ano: int, out_name: str = "rates_treino"):
    """Constrói (ou reusa) as tabelas de células usando SÓ os anos de treino.

    Para uma AVALIAÇÃO honesta (out-of-time): exclui o `holdout_ano` das taxas,
    para o score de células não "ver" o ano que será usado como teste. Salva em
    `data/processed/<out_name>/` e reusa se já existir. Difere das tabelas de
    PRODUÇÃO (`data/processed/rates/`, todos os anos) — essas seguem corretas
    para pontuar pessoas de verdade.
    """
    import pyarrow.parquet as pq
    out = cfg["abs"]["processed"] / out_name
    if (out / "levels.json").exists():
        return out
    anos_treino = [a for a in cfg["anos"] if a != holdout_ano]
    niveis = cells.active_levels(cfg)
    acc = rates.Accumulator(cfg["motivos"], levels=niveis)
    files = [f for f in sorted((cfg["abs"]["interim"] / "rais").rglob("*.parquet"))
             if int(f.parent.name.split("=")[1]) in anos_treino]
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=3_000_000):
            d = cells.add_cell_keys(binning.add_bins(batch.to_pandas(), cfg))
            acc.add(d)
    rates.save_level_tables(acc.tables(), cfg["motivos"], out)
    return out


def score_celulas_array(df: pd.DataFrame, scorer, motivo: str,
                        cfg: dict | None = None) -> np.ndarray:
    """Aplica o score de CÉLULAS (hazard anual EB) a cada linha de `df`.

    Vetoriza o binning/chaves e percorre as linhas no lookup do backoff —
    para comparar o score agregado com o CatBoost no mesmo conjunto.
    """
    cfg = cfg or load_config()
    d = cells.add_cell_keys(binning.add_bins(df, cfg))
    keycols = sorted({c for lvl in scorer.meta["order"] for c in scorer.meta["cols"][lvl]})
    recs = d[keycols].astype(str).to_dict(orient="records")
    m = scorer.m
    out = np.empty(len(recs), dtype=float)
    for i, keys in enumerate(recs):
        haz, _, _ = rates.eb_annual_hazard(scorer.indexes, scorer.meta, keys, motivo, m)
        out[i] = haz
    return out


def reliability_table(y_true, p_pred, bins: int = 10) -> pd.DataFrame:
    """Curva de calibração: previsto médio vs observado, por decil de risco."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p_pred)})
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(prevista=("p", "mean"),
                                             observada=("y", "mean"),
                                             n=("y", "size")).reset_index(drop=True)
    return g
