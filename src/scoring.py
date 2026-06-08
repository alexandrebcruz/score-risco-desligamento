"""Função de scoring: dos atributos de uma pessoa ao risco de desligamento.

Carrega as tabelas de taxas por nível (data/processed/rates/) uma única vez e
expõe:
- `score_pessoa(...)`: risco por horizonte e por motivo para um indivíduo.
- `score_lote(df)`: versão vetorizada (linha a linha) para um DataFrame.

O score NÃO é uma predição individual calibrada: é a taxa histórica da célula
a que a pessoa pertence, suavizada por Empirical Bayes e backoff hierárquico.
Pessoas com atributos idênticos recebem o mesmo score (ver limitações no README).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pandas as pd

from .config import load_config
from . import binning, cells, rates


class Scorer:
    """Encapsula as tabelas de taxas e a lógica de consulta."""

    def __init__(self, rates_dir: Path | None = None, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        rates_dir = Path(rates_dir or self.cfg["abs"]["rates"])
        tables, meta = rates.load_level_tables(rates_dir)
        self.meta = meta
        self.indexes = rates.build_indexes(tables, meta)
        self.motivos = meta["motivos"]
        self.m = float(self.cfg["suavizacao"]["shrinkage_m"])
        self.horizontes = list(self.cfg["horizontes_meses"])
        # Fatores de ajuste conjuntural (CAGED), se disponíveis: por motivo,
        # indexados por L = (cbo2, cnae2, uf). Opcional — ausência => sem ajuste.
        self.fatores = self._load_fatores(self.cfg["abs"]["processed"])

    @staticmethod
    def _load_fatores(processed_dir) -> dict:
        out = {}
        for f in Path(processed_dir).glob("caged_fator_*.parquet"):
            motivo = f.stem.replace("caged_fator_", "")
            df = pd.read_parquet(f)
            out[motivo] = {(r.cbo2, r.cnae2, r.uf): float(r.fator)
                           for r in df.itertuples(index=False)}
        return out

    def _fator(self, motivo: str, keys: dict) -> float:
        tab = self.fatores.get(motivo)
        if not tab:
            return 1.0
        return tab.get((keys["cbo2"], keys["cnae2"], str(keys["uf"]).upper()), 1.0)

    # -- preparação das chaves de célula a partir de atributos brutos --------
    def _keys_from_attrs(self, cbo, cnae, uf, idade, escolaridade,
                         tempo_vinculo_meses, tamanho_empresa) -> dict:
        """Aplica o MESMO binning do treino e deriva as chaves de célula."""
        attrs = {
            "cbo": cbo, "cnae": cnae, "uf": str(uf).upper(),
            "tempo_faixa": str(binning.bin_tempo_vinculo([tempo_vinculo_meses], self.cfg)[0]),
            "idade_faixa": str(binning.bin_idade([idade], self.cfg)[0]),
            "escol_faixa": str(binning.bin_escolaridade([escolaridade])[0]),
            "tamanho_faixa": str(binning.bin_tamanho([tamanho_empresa], self.cfg)[0]),
        }
        return cells.person_keys(attrs)

    def score_pessoa(self, *, cbo, cnae, uf, idade, escolaridade,
                     tempo_vinculo_meses, tamanho_empresa,
                     motivos: list[str] | None = None,
                     horizontes: list[int] | None = None,
                     ajuste_conjuntural: bool = False) -> dict:
        """Retorna o risco da pessoa por horizonte e motivo, com metadados.

        Estrutura de retorno:
            {
              "risco": {motivo: {horizonte: prob, ...}, ...},
              "hazard_anual": {motivo: prob, ...},
              "ic_anual": {motivo: (lo, hi), ...},
              "nivel_usado": {motivo: nome_nivel, ...},
              "exposicao": {motivo: n, ...},
              "confiavel": {motivo: bool, ...},   # exposição >= exposicao_minima
            }
        """
        motivos = motivos or self.cfg["motivo_default"]
        horizontes = horizontes or self.horizontes
        keys = self._keys_from_attrs(cbo, cnae, uf, idade, escolaridade,
                                     tempo_vinculo_meses, tamanho_empresa)

        out = {"risco": {}, "hazard_anual": {}, "ic_anual": {},
               "nivel_usado": {}, "exposicao": {}, "confiavel": {}, "fator_conjuntural": {}}
        exp_min = float(self.cfg["suavizacao"]["exposicao_minima"])
        for motivo in motivos:
            haz, nivel, n = rates.eb_annual_hazard(
                self.indexes, self.meta, keys, motivo, self.m)
            # Ajuste conjuntural opcional (CAGED): multiplica o hazard estrutural
            # pelo fator recente do nível L=(cbo2,cnae2,uf).
            fator = self._fator(motivo, keys) if ajuste_conjuntural else 1.0
            haz_aj = min(max(haz * fator, 0.0), 0.999999)
            out["hazard_anual"][motivo] = haz_aj
            out["fator_conjuntural"][motivo] = fator
            out["risco"][motivo] = {h: rates.horizon_risk(haz_aj, h) for h in horizontes}
            out["ic_anual"][motivo] = rates.beta_ci(0, n, haz_aj, self.m)
            out["nivel_usado"][motivo] = nivel
            out["exposicao"][motivo] = n
            out["confiavel"][motivo] = n >= exp_min
        return out


@lru_cache(maxsize=1)
def _default_scorer() -> Scorer:
    return Scorer()


def score_pessoa(**kwargs) -> dict:
    """Atalho de conveniência usando o Scorer padrão (cache singleton)."""
    return _default_scorer().score_pessoa(**kwargs)


def score_lote(df: pd.DataFrame, motivos: list[str] | None = None,
               horizontes: list[int] | None = None) -> pd.DataFrame:
    """Aplica o score a um DataFrame de pessoas.

    Espera as colunas: cbo, cnae, uf, idade, escolaridade,
    tempo_vinculo_meses, tamanho_empresa. Retorna o df original acrescido de
    colunas risco_<motivo>_<H>m, nivel_<motivo> e exposicao_<motivo>.
    """
    scorer = _default_scorer()
    motivos = motivos or scorer.cfg["motivo_default"]
    horizontes = horizontes or scorer.horizontes
    registros = []
    for _, row in df.iterrows():
        res = scorer.score_pessoa(
            cbo=row["cbo"], cnae=row["cnae"], uf=row["uf"], idade=row["idade"],
            escolaridade=row["escolaridade"],
            tempo_vinculo_meses=row["tempo_vinculo_meses"],
            tamanho_empresa=row["tamanho_empresa"],
            motivos=motivos, horizontes=horizontes)
        rec = {}
        for mtv in motivos:
            for h in horizontes:
                rec[f"risco_{mtv}_{h}m"] = res["risco"][mtv][h]
            rec[f"nivel_{mtv}"] = res["nivel_usado"][mtv]
            rec[f"exposicao_{mtv}"] = res["exposicao"][mtv]
        registros.append(rec)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(registros)], axis=1)
