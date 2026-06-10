"""Features de histórico agregado por categoria (target/count encoding TEMPORAL).

Ideia: para cada feature categórica, agregar por (valor, ano) o nº de vínculos
`n` e o nº de desligamentos involuntários `k_sjc`. Depois, no dataset de treino,
para cada vínculo do ano Y, trazer (n, k_sjc) da sua categoria nos 3 anos
ANTERIORES (Y-1, Y-2, Y-3) — lag1/lag2/lag3.

Por que NÃO vaza: usa apenas anos passados em relação ao vínculo, então é
informação disponível no momento da previsão. As features ficam completas só de
2019 em diante (precisam de 3 anos anteriores; dados desde 2016).

Inclui os NÍVEIS HIERÁRQUICOS de CBO/CNAE (cbo, cbo4, cbo2, cbo1, cnae, cnae5,
cnae3, cnae2) — agregados mais grossos são mais robustos para categorias raras.
"""
from __future__ import annotations

from pathlib import Path
import glob
import numpy as np
import pandas as pd

from .config import load_config

# Features categóricas a agregar (univariadas), incl. níveis hierárquicos.
LAG_FEATURES = [
    "cbo", "cbo4", "cbo2", "cbo1", "cnae", "cnae5", "cnae3", "cnae2",
    "uf", "escolaridade", "tamanho_estab", "tipo_vinculo", "faixa_remuneracao",
    "natureza_juridica", "natureza_setor", "intermitente", "simples",
    "faixa_horas",
]
# Colunas lidas do interim (cbo4/cbo2/... são derivadas; natureza_setor já existe).
_BASE_COLS = ["cbo", "cnae", "uf", "escolaridade", "tamanho_estab", "tipo_vinculo",
              "faixa_remuneracao", "natureza_juridica", "natureza_setor",
              "intermitente", "simples", "faixa_horas",
              "motivo_unificado"]
LAGS = (1, 2, 3)


def _derive_hier(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva os níveis hierárquicos de CBO/CNAE (como no treino).

    Sobrescreve as colunas base `cbo`/`cnae` já com zero-padding (6/7 dígitos):
    o layout de 2023 vem sem o zero à esquerda ('10105') enquanto 2019-22 vêm
    com ('010105'). Padronizar aqui garante que a categoria e a chave de join do
    lag casem entre anos (afeta CBOs/CNAEs do grupo iniciado em 0).
    """
    df["cbo"] = df["cbo"].astype(str).str.zfill(6)
    df["cnae"] = df["cnae"].astype(str).str.zfill(7)
    cbo, cnae = df["cbo"], df["cnae"]
    df["cbo4"], df["cbo2"], df["cbo1"] = cbo.str[:4], cbo.str[:2], cbo.str[:1]
    df["cnae5"], df["cnae3"], df["cnae2"] = cnae.str[:5], cnae.str[:3], cnae.str[:2]
    return df


def build_lag_aggs(cfg: dict | None = None, motivo: str = "involuntario_sjc",
                   out_dir: Path | None = None) -> Path:
    """Gera, do interim 2016-2023, as agregações (valor, ano) -> (n, k_sjc) por feature.

    Processa partição a partição (cada uma é de um ano) e acumula. Salva um
    `agg_<feature>.parquet` por feature em `data/processed/lags/`.
    """
    cfg = cfg or load_config()
    out_dir = Path(out_dir or (cfg["abs"]["processed"] / "lags"))
    out_dir.mkdir(parents=True, exist_ok=True)
    acc: dict[str, pd.DataFrame | None] = {f: None for f in LAG_FEATURES}

    import sys, time, gc
    from .cleaning import normalize_short_codes
    t0 = time.time()
    def _log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True); sys.stdout.flush()

    # --- RESUME: carrega agg já existentes e descobre anos já concluídos ---
    # (snapshots só ocorrem na virada de ano, então todo ano presente no agg está
    #  completo). Anos concluídos são pulados -> cada execução faz só o que falta,
    #  ficando abaixo do limite de tempo de processos longos do sandbox.
    done_anos: set[int] = set()
    for f in LAG_FEATURES:
        p = out_dir / f"agg_{f}.parquet"
        if p.exists():
            acc[f] = pd.read_parquet(p)
            done_anos |= set(int(x) for x in acc[f]["ano"].unique())
    if done_anos:
        _log(f"RESUME: anos já concluídos = {sorted(done_anos)} (serão pulados)")

    def _snapshot():
        for f in LAG_FEATURES:
            if acc[f] is not None:
                tmp = out_dir / f"agg_{f}.parquet.tmp"
                acc[f].to_parquet(tmp, index=False)
                tmp.replace(out_dir / f"agg_{f}.parquet")

    files = sorted(glob.glob(str(cfg["abs"]["interim"] / "rais" / "ano=*" / "*.parquet")))
    files = [fp for fp in files if int(fp.split("ano=")[1].split("/")[0]) not in done_anos]
    _log(f"{len(files)} partições a processar (após pular anos concluídos) | saída: {out_dir}")
    prev_ano = None
    for i, fp in enumerate(files, 1):
        ano = int(fp.split("ano=")[1].split("/")[0])
        # virada de ano -> snapshot (poucos writes; deixa acompanhar ano a ano sem estourar a janela)
        if prev_ano is not None and ano != prev_ano:
            _snapshot(); _log(f"  >> snapshot gravado (concluído ano {prev_ano})")
        prev_ano = ano
        d = pd.read_parquet(fp, columns=_BASE_COLS)
        d = normalize_short_codes(d)   # remap '999'->'99' + strip de zeros (in-place)
        d = _derive_hier(d)            # zfill cbo/cnae + substrings (in-place)
        d["k"] = (d["motivo_unificado"] == motivo).astype("int64")
        nrows = len(d)
        for f in LAG_FEATURES:
            g = (d.groupby(f, observed=True)
                   .agg(n=("k", "size"), k_sjc=("k", "sum")).reset_index())
            g["ano"] = ano
            g = g.rename(columns={f: "valor"})
            acc[f] = g if acc[f] is None else (
                pd.concat([acc[f], g], ignore_index=True)
                  .groupby(["valor", "ano"], as_index=False)[["n", "k_sjc"]].sum())
        del d; gc.collect()
        _log(f"partição {i}/{len(files)} (ano={ano}, {nrows:,} linhas)")
    _snapshot()
    _log("FIM build_lag_aggs (snapshot final gravado)")
    return out_dir


def load_lag_aggs(out_dir: Path | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    out_dir = Path(out_dir or (cfg["abs"]["processed"] / "lags"))
    return {f: pd.read_parquet(out_dir / f"agg_{f}.parquet") for f in LAG_FEATURES}


def add_lag_features(df: pd.DataFrame, aggs: dict, features=LAG_FEATURES,
                     lags=LAGS) -> pd.DataFrame:
    """Junta ao `df` (que tem as colunas das features + 'ano') as colunas de lag.

    Para cada feature F e lag L: traz n e k_sjc da categoria no ano (ano - L).
    Cria F_n_lagL e F_k_lagL. Categoria/ano ausente -> NaN (CatBoost trata).
    """
    df = _derive_hier(df.copy())
    for f in features:
        a = aggs[f]                          # colunas: valor, ano, n, k_sjc
        for L in lags:
            t = a.rename(columns={"valor": f, "n": f"{f}_n_lag{L}", "k_sjc": f"{f}_k_lag{L}"}).copy()
            t["ano"] = t["ano"] + L          # agg do ano X serve de lag L p/ vínculos de X+L
            df = df.merge(t[[f, "ano", f"{f}_n_lag{L}", f"{f}_k_lag{L}"]],
                          on=[f, "ano"], how="left")
    return df


def lag_feature_names(features=LAG_FEATURES, lags=LAGS) -> list[str]:
    """Nomes das colunas de lag geradas (todas numéricas)."""
    return [f"{f}_{kind}_lag{L}" for f in features for L in lags for kind in ("n", "k")]


if __name__ == "__main__":
    cfg = load_config()
    print("Gerando agregações de lag do interim 2016-2023...")
    out = build_lag_aggs(cfg)
    aggs = load_lag_aggs(out, cfg)
    for f in LAG_FEATURES:
        a = aggs[f]
        print(f"  agg_{f}: {a['valor'].nunique()} categorias x {sorted(a['ano'].unique())} anos "
              f"({len(a):,} linhas)")
    print("OK em", out)
