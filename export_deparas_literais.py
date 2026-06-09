"""Gera os de-paras LITERAIS (1:1, código -> rótulo OFICIAL) das categóricas, a partir
do RAIS_vinculos_layout2020.xls (versionado em data/dicts/). Complementam os de-paras
AGRUPADOS (que o pipeline usa). Re-executável.

Saídas em data/dicts/:
  depara_motivo_literal.csv, depara_tipo_vinculo_literal.csv,
  depara_natureza_juridica_literal.csv, depara_tamanho_estab_literal.csv,
  depara_causa_afastamento_literal.csv, depara_faixa_remuneracao_literal.csv,
  depara_faixa_horas_literal.csv, depara_cbo_literal.csv,
  depara_cnae_subclasse_literal.csv, depara_cnae_classe_literal.csv,
  depara_municipio_literal.csv
(escolaridade já tem depara_escolaridade_literal.csv; uf tem depara_uf.csv.)

  MPLCONFIGDIR=/tmp/mpl /tmp/consig_venv/bin/python export_deparas_literais.py
"""
import os, csv
import xlrd

XLS = "data/dicts/RAIS_vinculos_layout2020.xls"
OUT = "data/dicts"
wb = xlrd.open_workbook(XLS)
resumo = []

def _code(v):
    """'10.0'->'10'; '2062.0'->'2062'; '{ñ class}'/''/-1 -> None (ignora)."""
    s = str(v).strip()
    if not s or "class" in s.lower():
        return None
    try:
        i = int(float(s))
        return None if i < 0 else str(i)
    except ValueError:
        return None

def salva(nome, header, linhas):
    p = os.path.join(OUT, nome)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(linhas)
    resumo.append((nome, len(linhas)))

# ---------- 1) blocos inline da aba principal ----------
sh = wb.sheet_by_name("RAISD - layout")
# acha as linhas onde começa um campo (col0 curto e não-vazio)
starts = [r for r in range(sh.nrows)
          if str(sh.cell_value(r, 0)).strip() and len(str(sh.cell_value(r, 0)).strip()) < 60
          and str(sh.cell_value(r, 0)).strip().lower() != "nome"]

def bloco(r0, code_col=4, lab_col=3, desc_col=5, zpad=None):
    """Extrai (codigo, rotulo[, descricao]) do bloco que começa em r0 até o próximo campo."""
    fim = next((s for s in starts if s > r0), sh.nrows)
    out = []
    for r in range(r0, fim):
        cod = _code(sh.cell_value(r, code_col))
        if cod is None:
            continue
        if zpad:
            cod = cod.zfill(zpad)
        lab = str(sh.cell_value(r, lab_col)).strip()
        desc = str(sh.cell_value(r, desc_col)).strip() if desc_col is not None else ""
        out.append([cod, lab, desc] if desc_col is not None else [cod, lab])
    return out

# (start_row, nome_arquivo, header)
INLINE = [
    (7,   "depara_causa_afastamento_literal.csv", ["codigo", "rotulo"]),
    (43,  "depara_motivo_literal.csv",            ["codigo", "rotulo", "descricao"]),
    (156, "depara_natureza_juridica_literal.csv", ["codigo", "rotulo"]),
    (252, "depara_tamanho_estab_literal.csv",     ["codigo", "rotulo"]),
    (292, "depara_tipo_vinculo_literal.csv",      ["codigo", "rotulo"]),
]
for r0, nome, header in INLINE:
    desc_col = 5 if "descricao" in header else None
    salva(nome, header, bloco(r0, desc_col=desc_col))

# ---------- 2) aba FAIXAS: remuneração média (do ano) e hora contratual ----------
fx = wb.sheet_by_name("FAIXAS")
# remuneração média do ano: col6=código(00..11), col7=descrição; para no próximo cabeçalho
rem = []; capt = False
for r in range(fx.nrows):
    h = str(fx.cell_value(r, 6)).strip().upper()
    if h.startswith("FAIXA REMUNERA") and "ANO" in h:
        capt = True; continue
    if capt:
        if h.startswith("FAIXA "):              # próximo cabeçalho em col6 -> encerra
            break
        c = _code(fx.cell_value(r, 6)); d = str(fx.cell_value(r, 7)).strip()
        if c is not None and d:
            rem.append([c, d])
salva("depara_faixa_remuneracao_literal.csv", ["codigo", "faixa_sm"], rem)
# hora contratual: bloco col0 após "FAIXA HORA CONTRATUAL"
hh = []; capt = False
for r in range(fx.nrows):
    a = str(fx.cell_value(r, 0)).strip()
    if a.upper().startswith("FAIXA HORA"):
        capt = True; continue
    if capt:
        c = _code(a)
        if c is None:
            break
        hh.append([c, str(fx.cell_value(r, 1)).strip()])
salva("depara_faixa_horas_literal.csv", ["codigo", "faixa_horas"], hh)

# ---------- 3) abas-dicionário (formato "CODIGO:Descrição") ----------
def planilha_coddesc(aba, zpad=None):
    s = wb.sheet_by_name(aba); out = []
    for r in range(1, s.nrows):                 # linha 0 é título
        v = str(s.cell_value(r, 0))
        if ":" not in v:
            continue
        cod, desc = v.split(":", 1)
        cod = cod.strip()
        if zpad:
            cod = cod.zfill(zpad)
        out.append([cod, desc.strip()])
    return out

salva("depara_cbo_literal.csv",            ["cbo", "ocupacao"],   planilha_coddesc("ocupação", zpad=6))
salva("depara_cnae_subclasse_literal.csv", ["cnae", "atividade"], planilha_coddesc("subclasse 2.0", zpad=7))
salva("depara_cnae_classe_literal.csv",    ["cnae_classe", "atividade"], planilha_coddesc("classe 1.0 ou 95"))
salva("depara_municipio_literal.csv",      ["municipio", "nome"], planilha_coddesc("municipio"))

print("Gerados em data/dicts/:")
for nome, n in resumo:
    print(f"  {nome:42s} {n:>5} linhas")
