"""I/O e geração de amostra sintética.

Responsabilidades:
- `download()`: download idempotente com cache (pula se o arquivo já existe).
- Geradores de amostra sintética com o MESMO schema lógico dos microdados
  RAIS (vínculos) e Novo CAGED (movimentações), para destravar o pipeline
  ponta a ponta sem depender dos downloads reais (synthetic_mode: true).

IMPORTANTE: o gerador sintético embute, de propósito, relações conhecidas
(tempo de vínculo curto -> mais desligamento; empresa menor -> mais
desligamento; certos motivos por tipo de contrato). Isso permite que o
notebook 06 valide monotonicidades esperadas mesmo sem dados reais.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Vocabulário sintético (códigos plausíveis, NÃO são tabelas oficiais)
# ---------------------------------------------------------------------------
# CBO 2002 — ocupações de 6 dígitos plausíveis (grande grupo varia no 1º/2º díg).
_CBO = ["252105", "317110", "411005", "521110", "715210", "782310",
        "621005", "911205", "223505", "351405", "422105", "514315",
        "812105", "451305", "252110", "724210"]
# CNAE 2.0 — subclasses de 7 dígitos plausíveis (seção/divisão variam no início).
_CNAE = ["4711301", "1011201", "4120400", "5611201", "8610101", "8512100",
         "4930201", "6201501", "4520001", "7820500", "0111301", "3530100",
         "4781400", "9602501", "8211300", "4744001"]
_UFS = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "PE", "CE", "GO",
        "DF", "ES", "PA", "AM", "MT", "MS", "MA", "PB", "RN", "AL"]


def download(url: str, dest: Path, *, session=None, chunk: int = 1 << 16) -> Path:
    """Baixa `url` para `dest` se ainda não existir (idempotente).

    Retorna o caminho local. Lança em caso de falha de rede — o chamador
    decide se cai para o modo sintético. Mantido simples de propósito;
    para FTP/7z reais, trocar por urllib/ftplib + extração (ver notebook 01).
    """
    import requests

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest  # cache hit

    sess = session or requests.Session()
    with sess.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for block in resp.iter_content(chunk_size=chunk):
                fh.write(block)
        tmp.rename(dest)
    return dest


def download_ftp(url: str, dest: Path) -> Path:
    """Baixa um arquivo via FTP (idempotente). Usa urllib (suporta ftp://).

    Espaços na URL devem vir como %20 (ex.: 'NOVO%20CAGED').
    """
    import urllib.request

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest  # cache hit
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def extract_7z(path_7z: Path, dest_dir: Path) -> list[Path]:
    """Extrai um .7z para `dest_dir` usando py7zr; retorna os arquivos extraídos."""
    import py7zr

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(path_7z, "r") as z:
        nomes = z.getnames()
        # extrai apenas o que ainda não existe (idempotente)
        faltando = [n for n in nomes if not (dest_dir / n).exists()]
        if faltando:
            z.extract(path=dest_dir, targets=faltando)
    return [dest_dir / n for n in nomes]


def file_md5(path: Path, _bufsize: int = 1 << 20) -> str:
    """MD5 de um arquivo, para verificação de integridade do cache."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_bufsize), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Geração sintética
# ---------------------------------------------------------------------------
def _rng(seed: int, ano: int) -> np.random.Generator:
    """RNG determinístico por (seed, ano) — reprodutível e variando por ano."""
    return np.random.default_rng(seed + ano)


def gen_rais_ano(ano: int, n: int, seed: int = 42) -> pd.DataFrame:
    """Gera vínculos RAIS sintéticos com o schema *bruto* (códigos crus).

    Colunas refletem (de forma simplificada) o layout RAIS Vínculos:
    cbo, cnae, uf, idade, grau_instrucao, tamanho_estab, tempo_emprego_meses,
    vinculo_ativo_3112, mes_desligamento, motivo_desligamento.
    """
    rng = _rng(seed, ano)
    cbo = rng.choice(_CBO, n)
    cnae = rng.choice(_CNAE, n)
    uf = rng.choice(_UFS, n)
    idade = rng.integers(18, 70, n)
    grau = rng.integers(1, 12, n)            # 1..11 graus de instrução (estilo RAIS)
    tamanho = rng.integers(0, 6, n)          # faixa de tamanho do estabelecimento (0..5)
    tempo = rng.gamma(shape=2.0, scale=18.0, size=n).round(1)  # meses, cauda longa

    # --- Probabilidade de desligamento embute relações conhecidas ---
    base = 0.18
    # tempo curto aumenta risco (decai com o tempo de vínculo)
    f_tempo = np.clip(0.35 * np.exp(-tempo / 24.0), 0, 0.35)
    # empresa menor (tamanho baixo) aumenta risco
    f_tam = (5 - tamanho) * 0.012
    # ruído por célula CBO/CNAE
    f_cbo = (np.array([hash(c) % 7 for c in cbo]) - 3) * 0.01
    p_deslig = np.clip(base + f_tempo + f_tam + f_cbo, 0.02, 0.85)

    desligou = rng.random(n) < p_deslig
    mes_deslig = np.where(desligou, rng.integers(1, 13, n), 0)
    vinculo_ativo = np.where(desligou, 0, 1)

    # Motivo (códigos crus estilo RAIS) condicional a ter desligado.
    # 11=disp s/ justa causa, 12=disp c/ justa causa, 20/22=pedido,
    # 30/32=fim de contrato, 40/60=aposent/morte, 0=ativo.
    motivos_cod = np.zeros(n, dtype=int)
    idx = np.where(desligou)[0]
    # mais peso em dispensa s/ justa causa
    escolha = rng.choice([11, 12, 20, 30, 40, 90],
                         size=idx.size,
                         p=[0.50, 0.05, 0.20, 0.15, 0.05, 0.05])
    motivos_cod[idx] = escolha

    return pd.DataFrame({
        "ano": ano,
        "cbo": cbo,
        "cnae": cnae,
        "uf": uf,
        "idade": idade,
        "grau_instrucao": grau,
        "tamanho_estab": tamanho,
        "tempo_emprego_meses": tempo,
        "vinculo_ativo_3112": vinculo_ativo,
        "mes_desligamento": mes_deslig,
        "motivo_desligamento": motivos_cod,
    })


def gen_caged_ano(ano: int, n: int, seed: int = 42) -> pd.DataFrame:
    """Gera movimentações Novo CAGED sintéticas com schema *bruto*.

    Uma linha por movimentação. saldomovimentacao = +1 (admissão) / -1
    (desligamento). Para desligamentos há motivo. Demais campos espelham
    os principais do layout do Novo CAGED.
    """
    rng = _rng(seed + 1000, ano)
    competencia = ano * 100 + rng.integers(1, 13, n)  # AAAAMM
    cbo = rng.choice(_CBO, n)
    cnae = rng.choice(_CNAE, n)
    uf = rng.choice(_UFS, n)
    idade = rng.integers(18, 70, n)
    grau = rng.integers(1, 12, n)
    tamanho = rng.integers(0, 6, n)
    # ~52% admissões, 48% desligamentos
    saldo = rng.choice([1, -1], n, p=[0.52, 0.48])
    motivos_cod = np.zeros(n, dtype=int)
    idx = np.where(saldo == -1)[0]
    motivos_cod[idx] = rng.choice([11, 12, 20, 30, 40, 90], size=idx.size,
                                  p=[0.50, 0.05, 0.20, 0.15, 0.05, 0.05])
    return pd.DataFrame({
        "competenciamov": competencia,
        "cbo": cbo,
        "cnae": cnae,
        "uf": uf,
        "idade": idade,
        "grau_instrucao": grau,
        "tamanho_estab": tamanho,
        "saldomovimentacao": saldo,
        "motivo_desligamento": motivos_cod,
    })


def read_parquet_glob(folder: Path, pattern: str = "*.parquet") -> pd.DataFrame:
    """Concatena todos os parquet de uma pasta (usado para ler interim)."""
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Nenhum parquet em {folder}/{pattern}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
