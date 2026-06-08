"""Testes da lógica de iteração/URL do downloader PGFN (sem rede)."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.pgfn import Trimestre, iter_trimestres, build_url, TIPOS_VALIDOS


def test_proximo_trimestre_vira_o_ano():
    assert Trimestre(2020, 4).proximo() == Trimestre(2021, 1)
    assert Trimestre(2020, 1).proximo() == Trimestre(2020, 2)


def test_iter_trimestres_inclusivo():
    tris = list(iter_trimestres(Trimestre(2020, 1), Trimestre(2021, 2)))
    assert tris[0] == Trimestre(2020, 1)
    assert tris[-1] == Trimestre(2021, 2)
    assert len(tris) == 6  # 2020T1..T4 + 2021T1..T2


def test_iter_vazio_quando_fim_antes_do_inicio():
    assert list(iter_trimestres(Trimestre(2021, 1), Trimestre(2020, 4))) == []


def test_build_url():
    url = build_url(Trimestre(2026, 1), "FGTS")
    assert url == "https://dadosabertos.pgfn.gov.br/2026_trimestre_01/Dados_abertos_FGTS.zip"


def test_tipos_validos():
    assert set(TIPOS_VALIDOS) == {"Previdenciario", "Nao_Previdenciario", "FGTS"}
