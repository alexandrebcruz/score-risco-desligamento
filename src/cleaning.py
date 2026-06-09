"""Limpeza, padronização e harmonização RAIS x CAGED para um schema comum.

Schema canônico de saída (uma linha por vínculo/movimentação):
    ano, fonte, cbo, cnae, uf, idade, escolaridade, tamanho_estab,
    tempo_vinculo_meses, vinculo_ativo, mes_deslig,
    motivo_unificado, separado

`motivo_unificado` ∈ MOTIVOS (config) ou "ativo" quando não houve desligamento.
`separado` é booleano (houve desligamento no período).

Decisões:
- Códigos crus de motivo (RAIS e CAGED usam tabelas parecidas) são mapeados
  para categorias unificadas via MAPA_MOTIVO.
- Escolaridade (grau de instrução) é colapsada em faixas canônicas ordenadas.
- CBO/CNAE são normalizados como string e derivados em níveis (4/2 dígitos)
  em src/cells.py (não aqui), para manter responsabilidades separadas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Mapa de código cru de motivo -> categoria unificada.
# Cobre os principais códigos RAIS/CAGED (aproximação documentada).
MAPA_MOTIVO = {
    0:  "ativo",            # sem desligamento
    11: "involuntario_sjc", # dispensa sem justa causa
    12: "involuntario_cjc", # dispensa com justa causa
    20: "pedido_demissao",
    21: "pedido_demissao",
    22: "pedido_demissao",
    30: "fim_contrato",
    31: "fim_contrato",
    32: "fim_contrato",
    40: "aposent_morte",
    60: "aposent_morte",
    62: "aposent_morte",
    90: "outros",
}

# Grau de instrução RAIS "Escolaridade após 2005" (escala oficial 1..11):
#  1 Analfabeto | 2 até 5º incompleto | 3 5º completo | 4 6º-9º fundamental
#  5 Fundamental completo | 6 Médio incompleto | 7 Médio completo
#  8 Superior incompleto | 9 Superior completo | 10 Mestrado | 11 Doutorado
MAPA_ESCOLARIDADE = {
    1: "ate_fund_incompleto", 2: "ate_fund_incompleto", 3: "ate_fund_incompleto",
    4: "ate_fund_incompleto", 5: "fundamental", 6: "medio_incompleto",
    7: "medio_completo", 8: "superior_incompleto", 9: "superior",
    10: "superior", 11: "superior",
}


# ---------------------------------------------------------------------------
# Adaptador para microdados RAIS REAIS (arquivos .COMT do PDET/MTE)
# ---------------------------------------------------------------------------
# O layout da RAIS varia entre anos: o separador pode ser ',' (mais recente,
# extensão .COMT) ou ';' (formato antigo, decimal vírgula, extensão .txt); e os
# NOMES das colunas mudam (ex.: 'CBO 2002 Ocupação - Código' vs 'CBO Ocupação
# 2002'). Por isso casamos as colunas por TOKENS sobre o nome normalizado
# (minúsculo, sem acento, sem sufixo '- código'), e não por string exata.
#
# Cada campo lógico -> lista de predicados de tokens. Casa a 1ª coluna cujo
# nome normalizado satisfaça TODOS os tokens de algum predicado.
RAIS_FIELD_TOKENS = {
    "cbo":                 [["cbo", "ocupacao"]],
    "cnae":                [["cnae", "subclasse"]],
    "municipio":           [["municipio"]],          # casa exatamente 'municipio'
    "idade":               [["idade"]],
    "grau_instrucao":      [["escolaridade", "2005"]],
    "tamanho_estab":       [["tamanho", "estabelecimento"]],
    "tempo_emprego_meses": [["tempo", "emprego"]],    # 'tempo emprego' exato
    "vinculo_ativo_3112":  [["vinculo", "ativo"]],
    "mes_desligamento":    [["mes", "desligamento"]],
    "mes_admissao":        [["mes", "admissao"]],     # 0=admitido antes do ano; 1-12=mês de admissão no ano
    "motivo_desligamento": [["motivo", "desligamento"]],
    # --- variáveis adicionais (enriquecimento do modelo) ---
    "tipo_vinculo":        [["tipo", "vinculo"]],
    "categoria_trab":      [["categoria", "trabalhador"]],
    "faixa_remuneracao":   [["faixa", "rem", "media"]],
    "natureza_juridica":   [["natureza", "juridica"]],
    "intermitente":        [["intermitente"]],
    "parcial":             [["trabalho", "parcial"]],
    "qtd_dias_afastamento":[["dias", "afastamento"]],
    "simples":             [["simples"]],
    "faixa_horas":         [["faixa", "hora"]],
    "causa_afastamento":   [["causa", "afastamento", "1"]],
}
# Campos adicionais que são strings categóricas (não derivam UF/numérico).
RAIS_EXTRA_CAT = ["tipo_vinculo", "categoria_trab", "faixa_remuneracao",
                  "natureza_juridica", "intermitente", "parcial", "simples",
                  "faixa_horas", "causa_afastamento"]
# Colunas tratadas como float com decimal possivelmente em vírgula.
RAIS_FLOAT_FIELDS = {"tempo_emprego_meses"}
# Casamento por igualdade EXATA do nome normalizado (evita pegar 'município
# trab' ou 'faixa tempo emprego' em vez de 'município'/'tempo emprego').
RAIS_EXACT_TARGETS = {
    "municipio": "municipio",
    "idade": "idade",
    "tempo_emprego_meses": "tempo emprego",
}


def _norm_col(nome: str) -> str:
    """Normaliza um nome de coluna: minúsculo, sem acento, sem ' - código'."""
    import unicodedata
    s = unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode()
    s = s.lower().replace(" - codigo", "").replace("- codigo", "")
    s = s.replace("-", " ").replace(".", " ")
    return " ".join(s.split())


def _resolve_rais_columns(header: list[str]) -> dict:
    """Mapeia nome_real_no_arquivo -> nome_logico, casando por tokens.

    Campos ambíguos (município, idade, tempo emprego) exigem igualdade exata do
    nome normalizado; os demais casam pelo primeiro nome que contém os tokens.
    """
    norm = {c: _norm_col(c) for c in header}
    achado = {}
    for logico, predicados in RAIS_FIELD_TOKENS.items():
        alvo = RAIS_EXACT_TARGETS.get(logico)
        for real, n in norm.items():
            if alvo is not None:
                if n == alvo:
                    achado[logico] = real
                    break
                continue
            if any(all(tok in n for tok in pred) for pred in predicados):
                achado[logico] = real
                break
    faltam = set(RAIS_FIELD_TOKENS) - set(achado)
    # colunas adicionais são OPCIONAIS (formato antigo pode não tê-las) -> default depois.
    opcionais = set(RAIS_EXTRA_CAT) | {"qtd_dias_afastamento", "mes_admissao"}
    obrig = faltam - opcionais
    if obrig:
        raise KeyError(f"Colunas RAIS obrigatórias não resolvidas: {obrig} | header={header[:8]}...")
    if faltam:
        import logging
        logging.getLogger("hub").info("Colunas adicionais ausentes neste arquivo: %s", faltam)
    return {real: logico for logico, real in achado.items()}


def _detect_sep(path) -> str:
    """Detecta o separador (',' ou ';') pela 1ª linha do arquivo."""
    with open(path, encoding="latin-1") as fh:
        linha = fh.readline()
    return ";" if linha.count(";") > linha.count(",") else ","

# Código IBGE da UF (2 primeiros dígitos do código do município) -> sigla.
MAPA_UF_IBGE = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
    "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
    "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF",
}

# Tabela oficial de Motivo de Desligamento da RAIS -> categoria unificada.
# (Transferências 60-79 não são "perder o emprego" -> 'outros'; documentado.)
MAPA_MOTIVO_RAIS = {
    0: "ativo",
    10: "involuntario_cjc",   # rescisão COM justa causa por iniciativa do empregador
    11: "involuntario_sjc",   # rescisão SEM justa causa por iniciativa do empregador
    12: "fim_contrato",       # término do contrato de trabalho
    20: "outros",             # rescisão indireta (justa causa do empregado) — raro
    21: "pedido_demissao",    # rescisão sem justa causa por iniciativa do empregado
    22: "fim_contrato",       # término antecipado de contrato a termo
    30: "aposent_morte", 31: "aposent_morte", 32: "aposent_morte",
    33: "aposent_morte", 34: "aposent_morte", 35: "aposent_morte",
    40: "aposent_morte",      # falecimento
    50: "outros",
    60: "outros", 62: "outros", 63: "outros", 64: "outros",
    70: "outros", 72: "outros", 73: "outros", 74: "outros", 75: "outros",
    78: "outros", 79: "outros", 80: "outros", 81: "outros", 82: "outros",
    90: "outros",
}


def _to_num(s):
    """Converte para numérico tolerando decimal em vírgula (formato antigo)."""
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False),
                         errors="coerce")


def _rais_bruto_from_chunk(chunk: pd.DataFrame, ano: int,
                           ufs_subset: list[str] | None) -> pd.DataFrame:
    """Converte um chunk já renomeado (nomes lógicos) no schema BRUTO canônico."""
    c = chunk.copy()
    c["ano"] = ano
    c["uf"] = c["municipio"].str.strip().str[:2].map(MAPA_UF_IBGE).fillna("NI")
    if ufs_subset:
        c = c[c["uf"].isin(ufs_subset)]
    c["cbo"] = c["cbo"].astype(str).str.strip()
    c["cnae"] = c["cnae"].astype(str).str.strip()
    c["idade"] = _to_num(c["idade"])
    c["tempo_emprego_meses"] = _to_num(c["tempo_emprego_meses"])
    c["grau_instrucao"] = _to_num(c["grau_instrucao"]).astype("Int64")
    c["tamanho_estab"] = _to_num(c["tamanho_estab"]).astype("Int64")
    c["vinculo_ativo_3112"] = _to_num(c["vinculo_ativo_3112"]).fillna(0).astype(int)
    c["mes_desligamento"] = _to_num(c["mes_desligamento"]).fillna(0).astype(int)
    c["motivo_desligamento"] = _to_num(c["motivo_desligamento"]).fillna(0).astype(int)
    # adicionais (opcionais — criar com default se ausentes no formato antigo)
    if "qtd_dias_afastamento" not in c.columns:
        c["qtd_dias_afastamento"] = 0
    c["qtd_dias_afastamento"] = _to_num(c["qtd_dias_afastamento"]).fillna(0)
    for col in RAIS_EXTRA_CAT:
        if col not in c.columns:
            c[col] = "NI"
        c[col] = c[col].astype(str).str.strip()
    return c.drop(columns=["municipio"])


def iter_rais_clean_chunks(path, ano: int, ufs_subset: list[str] | None = None,
                           chunksize: int = 1_000_000):
    """Itera um arquivo de vínculos RAIS (.COMT/.txt) em chunks JÁ LIMPOS.

    Detecta separador e resolve nomes de coluna por tokens (robusto entre anos).
    Yields DataFrames no schema CANÔNICO — permite agregação incremental sem
    materializar o arquivo inteiro em memória.
    """
    sep = _detect_sep(path)
    header = pd.read_csv(path, sep=sep, encoding="latin-1", nrows=0).columns.tolist()
    ren = _resolve_rais_columns(header)          # nome_real -> nome_logico
    reader = pd.read_csv(path, sep=sep, encoding="latin-1", dtype=str,
                         usecols=list(ren.keys()), chunksize=chunksize)
    for chunk in reader:
        bruto = _rais_bruto_from_chunk(chunk.rename(columns=ren), ano, ufs_subset)
        yield clean_rais_real(bruto)


def read_rais_comt(path, ano: int, chunksize: int = 500_000,
                   ufs_subset: list[str] | None = None) -> pd.DataFrame:
    """Lê um arquivo de vínculos RAIS inteiro -> schema BRUTO canônico.

    Conveniência para inspeção/recortes pequenos. Para volumes grandes prefira
    `iter_rais_clean_chunks` (agregação incremental, memória baixa).
    """
    sep = _detect_sep(path)
    header = pd.read_csv(path, sep=sep, encoding="latin-1", nrows=0).columns.tolist()
    ren = _resolve_rais_columns(header)
    partes = []
    reader = pd.read_csv(path, sep=sep, encoding="latin-1", dtype=str,
                         usecols=list(ren.keys()), chunksize=chunksize)
    for chunk in reader:
        partes.append(_rais_bruto_from_chunk(chunk.rename(columns=ren), ano, ufs_subset))
    return pd.concat(partes, ignore_index=True)


# Códigos numéricos curtos cujo zero-padding VARIA entre layouts de ano
# (2019-2022 vêm zero-padded: '02','03'; 2023 vem sem padding: '2','3').
# Sem normalizar, '02' (treino) != '2' (holdout) -> categoria desconhecida em 2023
# e o join de lag por valor falha. Normalizamos removendo zeros à esquerda.
CODIGOS_A_NORMALIZAR = ["faixa_remuneracao", "faixa_horas", "causa_afastamento"]

# Remapeamentos de CONTEÚDO (não-formato) entre layouts de ano. Aplicados ANTES
# do strip de zeros. Ex.: em causa_afastamento a categoria default "sem afastamento"
# era '99' até 2022 (~83% dos vínculos) e virou '999' em 2023 (~84%) -> sem unificar,
# 84% do holdout 2023 cairia numa categoria nunca vista no treino.
CODE_REMAP = {
    "causa_afastamento": {"999": "99"},
}


def _strip_leading_zeros(s: pd.Series) -> pd.Series:
    """'02'->'2', '00'->'0'; mantém não-numéricos ('999','nao_informado') intactos."""
    s = s.astype(str).str.strip()
    num = s.str.fullmatch(r"0*\d+")
    norm = s.str.replace(r"^0+(?=\d)", "", regex=True)
    return s.where(~num.fillna(False), norm)


def normalize_short_codes(df: pd.DataFrame,
                          cols: list[str] = CODIGOS_A_NORMALIZAR) -> pd.DataFrame:
    """Harmoniza códigos curtos entre anos: remapeia conteúdo + remove zero-padding.

    Aplicar em TODO ponto de leitura (build de lags e leitura do treino) para
    garantir consistência train/serve quando o interim mistura formatos por ano.
    """
    for c, mapa in CODE_REMAP.items():
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace(mapa)
    for c in cols:
        if c in df.columns:
            df[c] = _strip_leading_zeros(df[c])
    return df


def clean_rais_real(df_bruto: pd.DataFrame) -> pd.DataFrame:
    """Padroniza o schema bruto da RAIS real para o schema canônico do projeto.

    Reutiliza a mesma estrutura de `clean_rais`, porém com o mapa de motivos
    OFICIAL da RAIS (MAPA_MOTIVO_RAIS) e a UF já derivada do município.
    """
    df = df_bruto.copy()
    separado = df["vinculo_ativo_3112"].astype(int) == 0
    out = pd.DataFrame({
        "ano": df["ano"].astype(int),
        "fonte": "RAIS",
        "cbo": df["cbo"].astype(str).str.strip(),
        "cnae": df["cnae"].astype(str).str.strip(),
        "uf": df["uf"].astype(str).str.upper().str.strip(),
        "idade": pd.to_numeric(df["idade"], errors="coerce"),
        "escolaridade": df["grau_instrucao"].map(MAPA_ESCOLARIDADE).fillna("nao_informado"),
        "tamanho_estab": pd.to_numeric(df["tamanho_estab"], errors="coerce"),
        "tempo_vinculo_meses": pd.to_numeric(df["tempo_emprego_meses"], errors="coerce"),
        "vinculo_ativo": df["vinculo_ativo_3112"].astype(int),
        "mes_deslig": pd.to_numeric(df["mes_desligamento"], errors="coerce").fillna(0).astype(int),
        # 0 = vínculo admitido em ano anterior (vigente no início do ano); 1-12 = mês de admissão no ano
        "mes_admissao": (pd.to_numeric(df["mes_admissao"], errors="coerce").fillna(0).astype(int)
                         if "mes_admissao" in df.columns else 0),
        "motivo_unificado": df["motivo_desligamento"].map(MAPA_MOTIVO_RAIS).fillna("outros"),
        "separado": separado,
        # --- variáveis adicionais (enriquecimento) ---
        "tipo_vinculo": df["tipo_vinculo"].astype(str).str.strip(),
        "categoria_trab": df["categoria_trab"].astype(str).str.strip(),
        "faixa_remuneracao": df["faixa_remuneracao"].astype(str).str.strip(),
        "natureza_juridica": df["natureza_juridica"].astype(str).str.strip(),
        # setor: 1º dígito da natureza jurídica (1=público, 2=privado, 3=outros)
        "natureza_setor": df["natureza_juridica"].astype(str).str.strip().str[:1],
        "intermitente": df["intermitente"].astype(str).str.strip(),
        "parcial": df["parcial"].astype(str).str.strip(),
        "simples": df["simples"].astype(str).str.strip(),
        "faixa_horas": df["faixa_horas"].astype(str).str.strip(),
        "causa_afastamento": df["causa_afastamento"].astype(str).str.strip(),
        "qtd_dias_afastamento": pd.to_numeric(df["qtd_dias_afastamento"], errors="coerce").fillna(0),
    })
    out.loc[~out["separado"], "motivo_unificado"] = "ativo"
    # Harmoniza zero-padding de códigos curtos entre layouts de ano (ver acima).
    out = normalize_short_codes(out)
    return out


def _map_motivo(cod: pd.Series) -> pd.Series:
    return cod.map(MAPA_MOTIVO).fillna("outros")


def _map_escolaridade(grau: pd.Series) -> pd.Series:
    return grau.map(MAPA_ESCOLARIDADE).fillna("nao_informado")


def clean_rais(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Padroniza um lote bruto de vínculos RAIS para o schema canônico."""
    df = df_raw.copy()
    separado = df["vinculo_ativo_3112"].astype(int) == 0
    out = pd.DataFrame({
        "ano": df["ano"].astype(int),
        "fonte": "RAIS",
        "cbo": df["cbo"].astype(str).str.strip(),
        "cnae": df["cnae"].astype(str).str.strip(),
        "uf": df["uf"].astype(str).str.upper().str.strip(),
        "idade": pd.to_numeric(df["idade"], errors="coerce"),
        "escolaridade": _map_escolaridade(df["grau_instrucao"]),
        "tamanho_estab": pd.to_numeric(df["tamanho_estab"], errors="coerce"),
        "tempo_vinculo_meses": pd.to_numeric(df["tempo_emprego_meses"], errors="coerce"),
        "vinculo_ativo": df["vinculo_ativo_3112"].astype(int),
        "mes_deslig": pd.to_numeric(df["mes_desligamento"], errors="coerce").fillna(0).astype(int),
        "motivo_unificado": _map_motivo(df["motivo_desligamento"]),
        "separado": separado,
    })
    # Quando ativo, força motivo "ativo" (consistência).
    out.loc[~out["separado"], "motivo_unificado"] = "ativo"
    return out


def clean_caged(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Padroniza movimentações Novo CAGED para o schema canônico.

    No CAGED cada linha é uma movimentação; aqui mantemos apenas a informação
    necessária para fluxos: admissões (saldo +1) e desligamentos (saldo -1).
    `tempo_vinculo_meses` não vem no CAGED de forma direta -> NaN (tratado no
    cálculo de taxas, que para o componente recente agrega sem essa dimensão).
    """
    df = df_raw.copy()
    desligamento = df["saldomovimentacao"].astype(int) == -1
    ano = (df["competenciamov"].astype(int) // 100)
    out = pd.DataFrame({
        "ano": ano.astype(int),
        "fonte": "CAGED",
        "competencia": df["competenciamov"].astype(int),
        "cbo": df["cbo"].astype(str).str.strip(),
        "cnae": df["cnae"].astype(str).str.strip(),
        "uf": df["uf"].astype(str).str.upper().str.strip(),
        "idade": pd.to_numeric(df["idade"], errors="coerce"),
        "escolaridade": _map_escolaridade(df["grau_instrucao"]),
        "tamanho_estab": pd.to_numeric(df["tamanho_estab"], errors="coerce"),
        "tempo_vinculo_meses": np.nan,
        "saldo": df["saldomovimentacao"].astype(int),
        "motivo_unificado": np.where(
            desligamento, _map_motivo(df["motivo_desligamento"]), "admissao"),
        "separado": desligamento,
    })
    return out


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Relatório simples de qualidade: % nulos e nº de categorias por coluna.

    Use para logar (não descartar silenciosamente) registros problemáticos.
    """
    rows = []
    n = len(df)
    for col in df.columns:
        rows.append({
            "coluna": col,
            "pct_nulo": round(100 * df[col].isna().mean(), 3) if n else 0.0,
            "n_unicos": int(df[col].nunique(dropna=True)),
        })
    return pd.DataFrame(rows)
