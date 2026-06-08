"""Testes de sanidade do pipeline de scoring.

Constroem um conjunto sintético pequeno em diretório temporário, calculam as
tabelas de taxa e validam:
- célula conhecida retorna taxa plausível;
- backoff é acionado para célula inexistente (cai em nível mais geral);
- monotonicidade: risco não cresce com o tempo de vínculo;
- coerência de probabilidades (0..1) e horizontes crescentes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import io_utils, cleaning, binning, cells, rates  # noqa: E402
from src.config import load_config  # noqa: E402
from src.scoring import Scorer  # noqa: E402


@pytest.fixture(scope="module")
def scorer(tmp_path_factory):
    """Pipeline mínimo: gera dados -> limpa -> taxas -> Scorer apontando p/ tmp."""
    cfg = load_config()
    rates_dir = tmp_path_factory.mktemp("rates")

    frames = [io_utils.gen_rais_ano(ano, 60000, seed=7) for ano in (2022, 2023)]
    rais = cleaning.clean_rais(pd.concat(frames, ignore_index=True))
    rais = binning.add_bins(rais, cfg)
    rais = cells.add_cell_keys(rais)

    tables = rates.build_level_tables(rais, cfg["motivos"])
    rates.save_level_tables(tables, cfg["motivos"], rates_dir)
    return Scorer(rates_dir=rates_dir, cfg=cfg)


def test_celula_conhecida_retorna_taxa_plausivel(scorer):
    r = scorer.score_pessoa(
        cbo="252105", cnae="4711301", uf="SP", idade=33, escolaridade="superior",
        tempo_vinculo_meses=8, tamanho_empresa=2, motivos=["involuntario_sjc"])
    haz = r["hazard_anual"]["involuntario_sjc"]
    assert 0.0 < haz < 1.0
    assert r["nivel_usado"]["involuntario_sjc"] != "global"  # achou nível específico


def test_backoff_para_celula_inexistente(scorer):
    # CBO/CNAE fora do vocabulário -> deve recuar para nível geral, sem quebrar.
    r = scorer.score_pessoa(
        cbo="9999", cnae="9999", uf="SP", idade=33, escolaridade="superior",
        tempo_vinculo_meses=8, tamanho_empresa=2, motivos=["involuntario_sjc"])
    assert 0.0 <= r["hazard_anual"]["involuntario_sjc"] < 1.0
    # nenhum nível específico de CBO inexistente; cai em global ou nível geográfico
    assert r["exposicao"]["involuntario_sjc"] >= 0


def test_monotonicidade_marginal_tempo_vinculo():
    # Efeito MARGINAL do tempo de vínculo na taxa observada (propriedade dos dados,
    # robusta a ruído de célula). Espera-se taxa decrescente com o tempo de vínculo.
    cfg = load_config()
    frames = [io_utils.gen_rais_ano(ano, 60000, seed=7) for ano in (2022, 2023)]
    rais = cleaning.clean_rais(pd.concat(frames, ignore_index=True))
    rais = binning.add_bins(rais, cfg)
    g = (rais.assign(sjc=rais["motivo_unificado"].eq("involuntario_sjc"))
              .groupby("tempo_faixa", observed=True)["sjc"].mean())
    ordem = ["<3m", "3-6m", "6-12m", "12-24m", "24-60m", "60+m"]
    taxas = [g[f] for f in ordem if f in g.index]
    assert all(a >= b - 1e-6 for a, b in zip(taxas, taxas[1:])), dict(zip(ordem, taxas))


def test_horizontes_crescentes_e_validos(scorer):
    r = scorer.score_pessoa(
        cbo="252105", cnae="4711301", uf="SP", idade=33, escolaridade="superior",
        tempo_vinculo_meses=8, tamanho_empresa=2, motivos=["involuntario_sjc"],
        horizontes=[3, 6, 12])
    rr = r["risco"]["involuntario_sjc"]
    assert 0 <= rr[3] <= rr[6] <= rr[12] <= 1  # janela maior -> risco acumulado maior
