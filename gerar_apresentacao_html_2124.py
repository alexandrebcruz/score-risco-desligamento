"""Deck HTML autossuficiente do MODELO NOVO (esteira 2124) — adaptação do motor de
gerar_apresentacao_html_mob.py (NÃO substitui os decks antigos).

- Slides estáticos: gerar_apresentacao_2124.py renderizado em SVG vetorial (texto
  selecionável, fonte DejaVu embutida) e embutido inline (ids prefixados).
- Slides interativos: B1 (curvas KM MOB), B2 (extrapolação Weibull até 36),
  B3 (gráfico-caixa Q1/mediana/média/Q3) — dados _mob_2124, 14 categorias.
- Slide C1 substituído por tabelas HTML nativas (prazo máx. + cobertura, _2124).

Uso:  MPLCONFIGDIR=/tmp/mpl /tmp/consig_venv/bin/python gerar_apresentacao_html_2124.py
Saída: outputs/apresentacao_risco_2124.html
"""
import os, runpy, json, shutil, re, math
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import pandas as pd
import base64
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none"
from matplotlib import cm, colors

DUMP = "/tmp/apresentacao_svg_2124"
os.environ["DECK_DUMP_PNG"] = DUMP
os.environ["DECK_DUMP_FMT"] = "svg"
OUT = "outputs/apresentacao_risco_2124.html"
TMP = "/tmp/apresentacao_risco_2124.html"

# ---------- 1. roda o deck (gera PDF + dump SVG) ----------
print("renderizando slides via gerar_apresentacao_2124.py ...")
ns = runpy.run_path("gerar_apresentacao_2124.py")
NP = len(ns["pages"])
IDX = ns["IDX"]                         # índices robustos vindos do deck
B1, B2, B3 = IDX["B1"], IDX["B2"], IDX["B3"]
CTAB = IDX["CTAB"]                      # slide das tabelas de consignado (vira HTML nativo)
TABCAT = IDX["tabcat"]                  # slide 8 (tabela de categorias) -> ganha botão + modal
print(f"{NP} slides; B1={B1} B2={B2} B3={B3} CTAB={CTAB} tabcat={TABCAT}")

def inline_svg(path, pfx):
    s = open(path, encoding="utf-8").read()
    s = re.sub(r"<\?xml[^>]*\?>", "", s)
    s = re.sub(r"<!DOCTYPE.*?>", "", s, flags=re.S)
    s = re.sub(r"<style[^>]*>.*?</style>", "", s, flags=re.S)
    s = re.sub(r'\bid="([^"]+)"', lambda m: 'id="%s%s"' % (pfx, m.group(1)), s)
    s = re.sub(r'href="#([^"]+)"', lambda m: 'href="#%s%s"' % (pfx, m.group(1)), s)
    s = re.sub(r'url\(#([^)]+)\)', lambda m: 'url(#%s%s)' % (pfx, m.group(1)), s)
    s = re.sub(r'(<svg\b[^>]*?)\s+width="[^"]*"', r'\1', s, count=1)
    s = re.sub(r'(<svg\b[^>]*?)\s+height="[^"]*"', r'\1', s, count=1)
    s = re.sub(r'<svg\b', '<svg class="deckslide" preserveAspectRatio="xMidYMid meet"', s, count=1)
    return s.strip()

# ---------- 2. dados de sobrevivência (_mob_2124) ----------
km = pd.read_csv("outputs/tables/sobrevivencia_km_mob_2124.csv").rename(columns={"mob": "mes"})
ext = pd.read_csv("outputs/tables/sobrevivencia_weibull_extrap_mob_2124.csv")
mono = pd.read_csv("outputs/tables/sobrevivencia_weibull_estatisticas_mono_mob_2124.csv").set_index("categoria")
res = pd.read_csv("outputs/tables/sobrevivencia_resumo_mob_2124.csv").set_index("categoria")
ks = sorted(km["categoria"].unique())
cmap = matplotlib.colormaps["RdYlGn_r"]; norm = colors.Normalize(vmin=min(ks), vmax=max(ks))
cor = {k: colors.to_hex(cmap(norm(k))) for k in ks}
def _contraste(hx):
    r, g, b = (int(hx[i:i + 2], 16) for i in (1, 3, 5))
    return "#000000" if 0.299 * r + 0.587 * g + 0.114 * b > 150 else "#ffffff"

series = []
for k in ks:
    S = [round(float(v), 5) for v in km[km.categoria == k].sort_values("mes")["S"].tolist()]
    Wv = [round(float(v), 5) for v in ext[ext.categoria == k].sort_values("mes")["S_weibull"].tolist()]
    mo = mono.loc[k]
    series.append({"k": int(k), "cor": cor[k], "txt": _contraste(cor[k]), "S": S, "W": Wv,
                   "risco12": round(float(res.loc[k, "risco_deslig_12m_KM"]) * 100, 1),
                   "q1": round(float(mo["q1_meses_mono"]), 1),
                   "medm": round(float(mo["mediana_meses_mono"]), 1),
                   "media": round(float(mo["media_meses_mono"]), 1),
                   "q3": round(float(mo["q3_meses_mono"]), 1)})
DATA = json.dumps(series, ensure_ascii=False)
GROUPS = [("Mínimo", [1], "#1a9850"), ("Baixo", [2, 3, 4], "#86cb66"),
          ("Médio-Baixo", [5, 6, 7], "#c9a227"), ("Médio", [8, 9, 10], "#fb8d3d"),
          ("Alto", [11, 12, 13, 14], "#d73027")]
GROUPS_JSON = json.dumps([{"nome": n, "cats": c, "cor": col} for n, c, col in GROUPS], ensure_ascii=False)

# ---------- taxa de desligamento por (categoria, ano) 2016–2025 — modal do slide 8 ----------
_tx = pd.read_csv("outputs/tables/categoria_ano_taxa_2124.csv")
_anos_tx = [c for c in _tx.columns if c != "categoria"]
RATE = {"anos": [int(a) for a in _anos_tx],
        "treino": [2021, 2024],   # período de modelagem (faixa sombreada, como no AUC/KS)
        "series": [{"k": int(r.categoria), "cor": cor[int(r.categoria)],
                    "v": [round(float(r[a]) * 100, 3) for a in _anos_tx]}
                   for _, r in _tx.sort_values("categoria").iterrows()]}
RATE_JSON = json.dumps(RATE, ensure_ascii=False)

# ---------- dados do slide de FEATURES (importância clicável, layout do deck anterior) ----------
_imp = pd.read_csv("outputs/runpod_retreino_2124/importancia_ensemble.csv").sort_values("imp_ensemble", ascending=False)
IMP_JSON = json.dumps([{"f": r.feature, "imp": round(float(r.imp_ensemble), 2)} for r in _imp.itertuples()], ensure_ascii=False)
FEATINFO = {
 "tempo_vinculo_meses": {"curto": "Tempo de vínculo", "nome": "Tempo de vínculo na ENTRADA (meses)",
   "desc": "Antiguidade do vínculo medida na ENTRADA da janela de observação (leak-free): não usa a data do desligamento. Vínculos novos têm risco muito maior.",
   "ex": ["Numérico, em meses", "Ex.: 0 (recém-admitido), 12, 60, 120"]},
 "cbo": {"curto": "CBO-6", "nome": "CBO — ocupação (6 dígitos)", "desc": "Código completo da ocupação.",
   "ex": ["715210 = Pedreiro", "517330 = Porteiro", "521110 = Vendedor de comércio varejista"]},
 "cbo4": {"curto": "CBO-4", "nome": "CBO — família ocupacional (4 dígitos)", "desc": "Ocupação no nível de família.",
   "ex": ["7152 = pedreiros", "5173 = vigias/porteiros", "4110 = auxiliares de escritório"]},
 "cbo2": {"curto": "CBO-2", "nome": "CBO — subgrupo principal (2 dígitos)", "desc": "Ocupação no nível de subgrupo.",
   "ex": ["51 = serviços / 52 = vendas no comércio", "71 = construção civil e extração"]},
 "cbo1": {"curto": "CBO-1", "nome": "CBO — grande grupo (1 dígito)", "desc": "Ocupação no nível mais agregado.",
   "ex": ["2 = profissionais das ciências", "5 = serviços e vendas", "7 = produção de bens (indústria/obra)"]},
 "cnae": {"curto": "CNAE-7", "nome": "CNAE — subclasse (completo)", "desc": "Código completo da atividade econômica do empregador.",
   "ex": ["4120400 = Construção de edifícios", "4711301 = Hipermercados"]},
 "cnae5": {"curto": "CNAE-5", "nome": "CNAE — classe (5 dígitos)", "desc": "Atividade no nível de classe.",
   "ex": ["41204 = Construção de edifícios", "47113 = Comércio varejista"]},
 "cnae3": {"curto": "CNAE-3", "nome": "CNAE — grupo (3 dígitos)", "desc": "Atividade no nível de grupo.",
   "ex": ["412 = Construção de edifícios", "471 = Comércio varejista não especializado"]},
 "cnae2": {"curto": "CNAE-2", "nome": "CNAE — divisão (2 dígitos)", "desc": "Atividade no nível de divisão.",
   "ex": ["41/42 = Construção", "47 = Comércio varejista", "56 = Alimentação", "84 = Administração pública"]},
 "uf": {"curto": "UF", "nome": "UF (Unidade da Federação)", "desc": "Estado do vínculo (derivado do município).",
   "ex": ["SP, MG, RJ, BA, RS…"]},
 "tipo_vinculo": {"curto": "Tipo de vínculo", "nome": "Tipo de vínculo", "desc": "Natureza jurídica do contrato de trabalho (RAIS).",
   "ex": ["10/15/20/25 = CLT prazo indeterminado", "30/31/35 = estatutário (servidor)", "50/55/60/65 = temporário / determinado"]},
 "natureza_juridica": {"curto": "Nat. jurídica", "nome": "Natureza jurídica do empregador", "desc": "Tipo jurídico da empresa/órgão (CONCLA).",
   "ex": ["2062 = Sociedade Ltda.", "2135 = Empresário individual", "1031 = Órgão público municipal"]},
 "natureza_setor": {"curto": "Nat. setor", "nome": "Natureza do setor", "desc": "1º dígito da natureza jurídica: distingue público de privado.",
   "ex": ["1 = setor público", "2 = privado", "3 = sem fins lucrativos"]},
 "intermitente": {"curto": "Intermitente", "nome": "Contrato intermitente", "desc": "Se o vínculo é de trabalho intermitente.",
   "ex": ["1 = sim", "0 = não", "-1 = não existia (2016)"]},
 "simples": {"curto": "Simples", "nome": "Optante pelo Simples Nacional", "desc": "Se o empregador é optante do Simples.",
   "ex": ["1 = sim", "0 = não"]},
 "idade": {"curto": "Idade", "nome": "Idade (anos)", "desc": "Idade da pessoa.", "ex": ["Ex.: 18, 30, 45, 60"]},
 "qtd_dias_afastamento": {"curto": "Afast./mês", "nome": "Dias de afastamento POR MÊS observado",
   "desc": "Taxa de afastamento normalizada pela exposição (leak-free): dias afastado ÷ meses observados no ano.",
   "ex": ["Numérico (dias/mês)", "Ex.: 0; 0,5; 2,3"]},
 "escolaridade": {"curto": "Escolaridade", "nome": "Escolaridade (grau de instrução, NUMÉRICA)",
   "desc": "Código ordinal 1–11 tratado como numérico — a ordem carrega o sinal.",
   "ex": ["1 = analfabeto … 5 = fundamental", "7 = médio completo", "9 = superior completo", "10/11 = mestrado/doutorado"]},
 "tamanho_estab": {"curto": "Tamanho estab.", "nome": "Tamanho do estabelecimento (NUMÉRICA)",
   "desc": "Faixa ordinal 1–10 de nº de vínculos do estabelecimento (porte).",
   "ex": ["2 = 1 a 4", "5 = 20 a 49", "10 = 1000 ou mais"]},
 "faixa_remuneracao": {"curto": "Remuneração", "nome": "Faixa de remuneração média (SM, NUMÉRICA)",
   "desc": "Faixa ordinal 0–11 do salário em salários mínimos; 99 (ignorado) vira -1.",
   "ex": ["0–2 = até 1,5 SM", "3–7 = 1,5 a 7 SM", "8–11 = acima de 7 SM"]},
 "faixa_horas": {"curto": "Horas", "nome": "Faixa de horas contratadas (NUMÉRICA)",
   "desc": "Faixa ordinal 1–6 da jornada semanal; 99 (ignorado) vira -1.",
   "ex": ["6 = 41–44h (integral)", "4 = 21–30h (parcial)"]},
}
FEATINFO_JSON = json.dumps(FEATINFO, ensure_ascii=False)

# ranking dos principais fatores (top 5) p/ a coluna esquerda do slide de features
_imp_sorted = _imp.sort_values("imp_ensemble", ascending=False).reset_index(drop=True)
_impmax = float(_imp_sorted["imp_ensemble"].iloc[0])
def _feattop_html(n=5):
    out = []
    for i, r in _imp_sorted.head(n).iterrows():
        nome = FEATINFO.get(r.feature, {}).get("curto", r.feature)
        v = float(r.imp_ensemble); w = max(4, v / _impmax * 100)
        out.append('<div class="ftrow"><span class="ftk">%d</span>'
                   '<span class="ftn">%s</span>'
                   '<span class="ftbar"><span style="width:%.0f%%"></span></span>'
                   '<span class="ftv">%s%%</span></div>' % (i + 1, nome, w, f"{v:.1f}".replace(".", ",")))
    return "\n".join(out)
FEATTOP_HTML = _feattop_html()

# fonte DejaVu embutida
_TTF = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
def _face(fname, weight):
    b = base64.b64encode(open(os.path.join(_TTF, fname), "rb").read()).decode()
    return ("@font-face{font-family:'DejaVu Sans';font-style:normal;font-weight:%s;"
            "src:url(data:font/ttf;base64,%s) format('truetype');}" % (weight, b))
FONTS = _face("DejaVuSans.ttf", "400") + _face("DejaVuSans-Bold.ttf", "700")

# ---------- 3. slides interativos ----------
def bullets_html(items):
    out = []
    for b, t in items:
        out.append((f'<div class="b"><span class="bi">▸</span>{t}</div>') if b else f'<div class="bh">{t}</div>')
    return "\n".join(out)

B1_TXT = bullets_html([
    (False, "A ideia"),
    (True, "O modelo prevê QUEM/SE é desligado; a sobrevivência mede QUANDO."),
    (True, "S(t) = prob. de seguir empregado t meses após a ENTRADA (relógio MOB)."),
    (False, "Dos microdados (RAIS 2021–2024 agrupados)"),
    (True, "Evento = dispensa s/ justa causa; censura = ativo ou saída por outro motivo."),
    (True, "Pré-existente entra em janeiro; admitido no ano entra no mês de admissão."),
    (False, "Kaplan–Meier"),
    (True, "S(t) = Π (nₘ−dₘ)/nₘ — usa a censura sem viés, mês a mês."),
    (True, "4 safras agregadas → sazonalidade de calendário diluída."),
])
B2_TXT = bullets_html([
    (False, "O problema"),
    (True, "12 meses de dado não enxergam além de 12m (a curva ainda está alta)."),
    (False, "Solução: forma paramétrica de Weibull"),
    (True, "S(t) = exp(−(t/λ)ᵖ);  hazard ∝ t^(p−1)."),
    (True, "Ajuste por regressão pura: ln(−ln S) = p·ln t + ln α (OLS, 12 pts)."),
    (True, "R² médio ≈ 0,994 — extrapola até 36 MOB (tracejado)."),
    (False, "Qualidade do ajuste"),
    (True, "Q1/mediana/média/Q3 monotônicos sem ajuste (0 inversões); projeção >12m é suposição."),
])

def interactive_slide(kicker, title, txt, chart_id):
    return f'''<div class="slide cust">
  <div class="hb"><span class="kick">{kicker}</span><span class="ttl">{title}</span></div>
  <div class="txt">{txt}</div>
  <div class="chartwrap">
    <div class="ctrls"><div class="grp" id="grp-{chart_id}"></div><div class="chips" id="chips-{chart_id}"></div></div>
    <svg id="svg-{chart_id}" viewBox="0 0 760 470" preserveAspectRatio="xMidYMid meet"></svg>
  </div>
</div>'''

def features_slide():
    return ('<div class="slide cust">'
            '<div class="hb"><span class="kick">DESENVOLVIMENTO · FEATURES</span>'
            '<span class="ttl">As 21 variáveis do modelo — importância e significado</span></div>'
            '<div class="featleft">'
            '<div class="fl-block"><div class="fl-h">Como ler</div>'
            '<div class="fl-d">A importância mede quanto cada variável ajuda o modelo a separar '
            'quem é desligado de quem permanece. <b>Clique numa barra</b> para ver o significado e exemplos.</div></div>'
            '<div class="fl-block"><div class="fl-h">Principais fatores de risco</div>'
            '<div class="fttable">' + FEATTOP_HTML + '</div></div>'
            '<div class="fl-leak"><b>Base leak-free:</b> tempo de vínculo medido na ENTRADA da janela · '
            'afastamento em dias POR MÊS de exposição · causa de afastamento removida · '
            'ordinais (escolaridade, porte, faixas) tratadas como numéricas.</div>'
            '</div>'
            '<div class="impwrap"><svg id="svg-imp" viewBox="0 0 560 470" preserveAspectRatio="xMidYMid meet"></svg></div>'
            '<div class="featinfo" id="featinfo"><div class="fi-h">Significado da variável</div>'
            '<div class="fi-d">Clique numa barra do gráfico para ver o que é a variável e exemplos de valores.</div></div>'
            '</div>')

def box_slide():
    return '''<div class="slide cust">
  <div class="hb"><span class="kick">TEMPO ATÉ O DESLIGAMENTO · ESTATÍSTICAS</span><span class="ttl">Q1, mediana, média e Q3 por categoria — MOB, ref. 2021–2024 (meses)</span></div>
  <div class="boxwrap"><svg id="svg-box" viewBox="0 0 600 470" preserveAspectRatio="xMidYMid meet"></svg></div>
  <div class="boxtable" id="boxtable"></div>
  <div class="boxhint">Passe o mouse numa categoria (caixa ou linha) para ver os dados</div>
</div>'''

# ---------- C1: tabelas de política (HTML nativo) ----------
PRAZO = pd.read_csv("outputs/tables/consignado_prazo_max_2124.csv")
COB = pd.read_csv("outputs/tables/consignado_cobertura_parcelas_2124.csv")
def _heat(frac): return "hsl(%d,62%%,87%%)" % round(max(0.0, min(1.0, frac)) * 120)
def _catcell(k): return '<td class="ct" style="background:%s;color:%s">%d</td>' % (cor[k], _contraste(cor[k]), k)
def _termo_tbl():
    h = "<tr><th>cat</th><th>95%</th><th>90%</th><th>85%</th><th>80%</th></tr>"
    body = ""
    for _, r in PRAZO.iterrows():
        k = int(r.categoria); cells = ""
        for c in ("conf_95", "conf_90", "conf_85", "conf_80"):
            v = float(r[c]); disp = "120+" if v > 120 else f"{v:.0f}"
            cells += '<td style="background:%s">%s</td>' % (_heat(min(v, 36) / 36), disp)
        body += "<tr>" + _catcell(k) + cells + "</tr>"
    return '<table class="aptbl"><thead>%s</thead><tbody>%s</tbody></table>' % (h, body)
def _cov_tbl():
    TS = [6, 12, 18, 24, 36, 48, 60]
    h = "<tr><th>cat</th>" + "".join(f"<th>T={t}</th>" for t in TS) + "</tr>"
    body = ""
    for _, r in COB.iterrows():
        k = int(r.categoria)
        cells = "".join('<td style="background:%s">%.0f%%</td>' % (_heat(r[f"T_{t}"] / 100), r[f"T_{t}"]) for t in TS)
        body += "<tr>" + _catcell(k) + cells + "</tr>"
    return '<table class="aptbl"><thead>%s</thead><tbody>%s</tbody></table>' % (h, body)
TERMO_TBL, COV_TBL = _termo_tbl(), _cov_tbl()

# ---------- taxa de juros mínima (break-even): tabela nativa com toggle mês/ano ----------
TAXA = pd.read_csv("outputs/tables/consignado_taxa_breakeven_2124.csv").set_index("categoria")
_TS_TX = [6, 12, 18, 24, 36, 48, 60]
def _taxa_tbl(modo):    # modo: "m" (% a.m.) ou "a" (% a.a.)
    h = "<tr><th>cat</th>" + "".join(f"<th>T={t}</th>" for t in _TS_TX) + "</tr>"
    # normaliza cor pela posição relativa da taxa (log) p/ heat consistente entre modos
    vmax = TAXA["m_T60"].max()
    body = ""
    for k in TAXA.index:
        k = int(k); cells = ""
        for t in _TS_TX:
            v = float(TAXA.loc[k, f"{modo}_T{t}"])
            frac = 1 - min(1.0, (float(TAXA.loc[k, f"m_T{t}"]) / vmax) ** 0.5)  # verde=baixa, vermelho=alta
            dec = 2 if modo == "m" else 1
            cells += '<td style="background:%s">%s%%</td>' % (_heat(frac), f"{v:.{dec}f}".replace(".", ","))
        body += "<tr>" + _catcell(k) + cells + "</tr>"
    return '<table class="aptbl"><thead>%s</thead><tbody>%s</tbody></table>' % (h, body)
TAXA_TBL_M, TAXA_TBL_A = _taxa_tbl("m"), _taxa_tbl("a")

def taxa_table_slide():
    return ('<div class="slide cust"><div class="hb"><span class="kick">REFERÊNCIA · TAXA DE JUROS MÍNIMA</span>'
            '<span class="ttl">Taxa de equilíbrio (break-even) por categoria e prazo</span></div>'
            '<div class="taxwrap">'
            '<div class="taxhead"><div class="apt-h" id="tax-h">Taxa mínima para recuperar o principal — % ao mês</div>'
            '<button class="taxbtn" id="tax-btn" onclick="toggleTaxa()">Ver % ao ano →</button></div>'
            '<div id="tax-m">' + TAXA_TBL_M + '</div>'
            '<div id="tax-a" style="display:none">' + TAXA_TBL_A + '</div>'
            '<div class="apt-note">Piso de quebra-zero: recebido nominal esperado = A·Σ S(m) ≥ principal '
            '(A = parcela Price; hipótese conservadora de zero recuperação após o desligamento). '
            'Taxa praticada = este piso + custo de captação + custo operacional + margem.</div>'
            '</div></div>')

# ---------- taxa por NPV (pricing): 4 variantes (ROI 10/20 × mês/ano) ----------
TAXA_NPV = pd.read_csv("outputs/tables/consignado_taxa_npv_2124.csv").set_index("categoria")
def _npv_tbl(roi, modo):    # roi: "10"/"20" ; modo: "m"/"a"
    h = "<tr><th>cat</th>" + "".join(f"<th>T={t}</th>" for t in _TS_TX) + "</tr>"
    vmax = TAXA_NPV[f"m{roi}_T6"].max()
    body = ""
    for k in TAXA_NPV.index:
        k = int(k); cells = ""
        for t in _TS_TX:
            v = float(TAXA_NPV.loc[k, f"{modo}{roi}_T{t}"])
            frac = 1 - min(1.0, (float(TAXA_NPV.loc[k, f"m{roi}_T{t}"]) / vmax) ** 0.5)
            dec = 2 if modo == "m" else 1
            cells += '<td style="background:%s">%s%%</td>' % (_heat(frac), f"{v:.{dec}f}".replace(".", ","))
        body += "<tr>" + _catcell(k) + cells + "</tr>"
    return '<table class="aptbl"><thead>%s</thead><tbody>%s</tbody></table>' % (h, body)
NPV_TBL = {(roi, modo): _npv_tbl(roi, modo) for roi in ("10", "20") for modo in ("m", "a")}

def npv_table_slide():
    variantes = "".join(
        '<div id="npv-%s%s"%s>%s</div>' % (roi, modo, ("" if (roi, modo) == ("10", "m") else ' style="display:none"'),
                                           NPV_TBL[(roi, modo)])
        for roi in ("10", "20") for modo in ("m", "a"))
    return ('<div class="slide cust"><div class="hb"><span class="kick">REFERÊNCIA · PRICING POR NPV</span>'
            '<span class="ttl">Taxa de pricing por categoria e prazo (NPV · funding 1,2%/mês)</span></div>'
            '<div class="taxwrap">'
            '<div class="taxhead"><div class="apt-h" id="npv-h">Taxa de pricing — ROI 10% · % ao mês</div>'
            '<div style="display:flex;gap:.5em">'
            '<button class="taxbtn" id="npv-roi" onclick="toggleNpvRoi()">Ver ROI 20% →</button>'
            '<button class="taxbtn" id="npv-per" onclick="toggleNpvPer()">Ver % ao ano →</button></div></div>'
            + variantes +
            '<div class="apt-note"><b>Por causa do ROI fixo, no baixo risco a taxa CAI com o prazo</b> '
            '(a margem-alvo se dilui em mais meses de pagamento quase certo; no alto risco isso some). '
            'Pricing-alvo: i tal que NPV = ROI·P (lucro a valor presente). Premissas: captação 1,2%/mês; '
            'zero recuperação após o desligamento; falta somar custo operacional e perdas residuais.</div>'
            '</div></div>')

def consig_tables_slide():
    return ('<div class="slide cust"><div class="hb"><span class="kick">REFERÊNCIA PARA A POLÍTICA DE CONCESSÃO</span>'
            '<span class="ttl">Prazo máximo e cobertura de parcelas por categoria (14 faixas, ref. 2021–2024)</span></div>'
            '<div class="aptwrap-l"><div class="apt-h">Prazo máx. (meses) por confiança de seguir empregado</div>' + TERMO_TBL +
            '<div class="apt-note">t = λ·(−ln c)^(1/p) · inteiros · cap "120+"</div></div>'
            '<div class="aptwrap-r"><div class="apt-h">Cobertura esperada de parcelas (% pagas em folha) por prazo T</div>' + COV_TBL +
            '<div class="apt-note">Σ S(m)/T · S = KM MOB (≤12) + Weibull (&gt;12) · ref. 2021–2024 · &gt;12m é projeção</div></div></div>')

# ---------- 4. monta os slides ----------
PREPARO = 2          # slide leak-free do PDF -> no HTML vira feature importance interativo
slides = []
for i in range(NP):
    if i == PREPARO:
        slides.append(features_slide())
    elif i == B1:
        slides.append(interactive_slide("TEMPO ATÉ O DESLIGAMENTO · SOBREVIVÊNCIA",
                      "Curvas de sobrevivência por categoria — MOB, ref. 2021–2024 (Kaplan-Meier)", B1_TXT, "km"))
    elif i == B2:
        slides.append(interactive_slide("TEMPO ATÉ O DESLIGAMENTO · EXTRAPOLAÇÃO",
                      "Extrapolação Weibull das curvas (até 36 MOB)", B2_TXT, "weib"))
    elif i == B3:
        slides.append(box_slide())
    elif i == CTAB:
        slides.append(consig_tables_slide())
    elif i == IDX["TAXTAB"]:
        slides.append(taxa_table_slide())
    elif i == IDX["TAXNPV"]:
        slides.append(npv_table_slide())
    elif i == TABCAT:
        # slide 8 estático + botão que abre o modal do histórico de taxa de desligamento
        svg = inline_svg(f"{DUMP}/slide_{i:02d}.svg", f"s{i:02d}_")
        slides.append(f'<div class="slide">{svg}'
                      '<button class="ratebtn" onclick="openRate()">📈 Taxa de desligamento ao longo dos anos</button>'
                      '</div>')
    else:
        slides.append(f'<div class="slide">{inline_svg(f"{DUMP}/slide_{i:02d}.svg", f"s{i:02d}_")}</div>')
SLIDES = "\n".join(slides)

# ---------- 5. template ----------
HTML = r"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8">
<title>Risco de Desligamento — apresentação</title>
<style>
  __FONTS__
  :root{ --navy:#14233f; --ink:#1b2430; --grey:#5b6675; }
  *{box-sizing:border-box;} html,body{margin:0;height:100%;background:#0d1626;font-family:'DejaVu Sans',-apple-system,Segoe UI,Roboto,Arial,sans-serif;}
  .deck{height:100vh;height:100dvh;display:flex;align-items:center;justify-content:center;}
  .stage{position:relative;width:min(100vw,177.78vh);height:min(56.25vw,100vh);
         width:min(100vw,177.78dvh);height:min(56.25vw,100dvh);--u:min(1vw,1.7778vh);
         background:#fff;box-shadow:0 6px 30px rgba(0,0,0,.5);overflow:hidden;}
  .slide{position:absolute;inset:0;display:none;}
  .slide.active{display:block;}
  .deckslide{position:absolute;inset:0;width:100%;height:100%;}
  .deckslide *{stroke-linejoin:round;stroke-linecap:butt;}
  .cust{background:#fff;}
  .hb{position:absolute;top:0;left:0;right:0;height:14%;background:var(--navy);border-left:6px solid #f4a722;
      display:flex;flex-direction:column;justify-content:center;padding-left:2.6%;}
  .kick{color:#9fc0e8;font-weight:bold;letter-spacing:.04em;font-size:calc(var(--u)*1.20);}
  .ttl{color:#fff;font-weight:bold;font-size:calc(var(--u)*1.9);}
  .txt{position:absolute;left:3.5%;top:19%;width:38%;}
  .bh{font-weight:bold;color:var(--ink);font-size:calc(var(--u)*1.31);margin:calc(var(--u)*0.95) 0 calc(var(--u)*0.2);}
  .b{color:var(--ink);font-size:calc(var(--u)*1.31);margin:calc(var(--u)*0.28) 0;padding-left:1.5em;text-indent:-1.5em;line-height:1.3;}
  .bi{color:#f4a722;font-weight:bold;margin-right:.5em;}
  .chartwrap{position:absolute;left:43%;top:16%;width:55%;height:80%;display:flex;flex-direction:column;}
  .ctrls{flex:0 0 auto;margin-bottom:.4vh;}
  .grp{display:flex;flex-wrap:wrap;gap:calc(var(--u)*0.35);margin-bottom:calc(var(--u)*0.35);}
  .grp button{font-size:calc(var(--u)*0.92);padding:calc(var(--u)*0.12) calc(var(--u)*0.6);border-radius:5px;border:1px solid #bbb;background:#f7f7f7;cursor:pointer;font-weight:600;}
  .chips{display:flex;flex-wrap:wrap;gap:calc(var(--u)*0.2);}
  .chip{width:calc(var(--u)*1.9);height:calc(var(--u)*1.5);border-radius:4px;border:1.5px solid var(--c);
        background:var(--c);color:#fff;font-size:calc(var(--u)*0.82);font-weight:700;cursor:pointer;padding:0;line-height:1;}
  .chip.off{background:#fff;color:#bbb;border-color:#ddd;}
  .chartwrap svg{flex:1 1 auto;width:100%;min-height:0;}
  .boxwrap{position:absolute;left:2%;top:15.5%;width:60%;height:82%;}
  .boxwrap svg{width:100%;height:100%;}
  .box{cursor:pointer;}
  .boxtable{position:absolute;right:2.5%;top:15.5%;width:33%;height:80%;overflow:auto;
            display:flex;flex-direction:column;justify-content:center;}
  .boxtable table{border-collapse:collapse;width:100%;font-size:calc(var(--u)*1.0);font-variant-numeric:tabular-nums;}
  .boxtable th{background:var(--navy);color:#fff;padding:calc(var(--u)*0.18) calc(var(--u)*0.25);position:sticky;top:0;}
  .boxtable td{padding:calc(var(--u)*0.14) calc(var(--u)*0.25);text-align:center;border-bottom:1px solid #eee;}
  .boxtable td.ct{color:#fff;font-weight:700;}
  .boxtable tr.hl td{background:#fff3cf;}
  .boxtable tr.hl td.ct{filter:brightness(.85);}
  .boxhint{position:absolute;left:2%;bottom:2.5%;color:var(--grey);font-size:calc(var(--u)*0.9);}
  /* slide de features: importância clicável + box de detalhes (layout do deck anterior) */
  /* box descritivo da variável: canto INFERIOR DIREITO (sobre as barras), como no deck anterior */
  .featinfo{position:absolute;right:2.5%;bottom:4.5%;width:34%;max-height:56%;overflow:auto;z-index:6;
            background:rgba(255,255,255,.97);border:1px solid #cdd5df;border-radius:8px;
            padding:calc(var(--u)*0.9);box-shadow:0 4px 16px rgba(0,0,0,.22);}
  .featinfo .fi-h{font-weight:700;font-size:calc(var(--u)*1.22);color:var(--navy);margin-bottom:.3em;}
  .featinfo .fi-d{font-size:calc(var(--u)*1.0);color:var(--ink);line-height:1.4;}
  .featinfo .fi-ex{margin-top:.45em;font-size:calc(var(--u)*0.96);padding-left:1.2em;}
  .featinfo .fi-ex li{margin:.16em 0;color:#33404f;}
  /* coluna esquerda: como ler + ranking dos top fatores + nota leak-free */
  .featleft{position:absolute;left:2.5%;top:16.5%;width:33.5%;bottom:3.5%;display:flex;flex-direction:column;}
  .featleft .fl-block{margin-bottom:calc(var(--u)*1.5);}
  .featleft .fl-h{font-weight:700;font-size:calc(var(--u)*1.25);color:var(--navy);margin-bottom:.35em;}
  .featleft .fl-d{font-size:calc(var(--u)*1.05);color:var(--ink);line-height:1.45;}
  .fttable .ftrow{display:flex;align-items:center;gap:calc(var(--u)*0.5);margin:calc(var(--u)*0.42) 0;}
  .ftk{flex:0 0 auto;width:calc(var(--u)*1.7);height:calc(var(--u)*1.7);border-radius:5px;background:var(--navy);
       color:#fff;font-weight:700;font-size:calc(var(--u)*1.0);display:flex;align-items:center;justify-content:center;}
  .ftn{flex:0 0 38%;font-size:calc(var(--u)*1.02);color:var(--ink);}
  .ftbar{flex:1 1 auto;height:calc(var(--u)*0.85);background:#e6edf5;border-radius:4px;overflow:hidden;}
  .ftbar > span{display:block;height:100%;background:#2e9e5b;border-radius:4px;}
  .ftv{flex:0 0 auto;width:calc(var(--u)*3.2);text-align:right;font-weight:700;font-size:calc(var(--u)*1.0);
       color:var(--navy);font-variant-numeric:tabular-nums;}
  .fl-leak{margin-top:auto;font-size:calc(var(--u)*0.86);color:var(--grey);line-height:1.4;}
  .impwrap{position:absolute;left:38%;top:15%;width:60%;height:82%;}
  .impwrap svg{width:100%;height:100%;}
  .imp-bar{cursor:pointer;}
  .aptwrap-l{position:absolute;left:2%;top:15.5%;width:31%;height:80%;overflow:auto;display:flex;flex-direction:column;justify-content:center;}
  .aptwrap-r{position:absolute;left:35%;top:15.5%;width:63%;height:80%;overflow:auto;display:flex;flex-direction:column;justify-content:center;}
  .apt-h{font-weight:700;font-size:calc(var(--u)*1.0);color:var(--navy);margin-bottom:.35em;line-height:1.2;}
  .apt-note{font-size:calc(var(--u)*0.82);color:var(--grey);margin-top:.35em;}
  .aptbl{border-collapse:collapse;width:100%;font-size:calc(var(--u)*0.95);font-variant-numeric:tabular-nums;}
  .aptbl th{background:var(--navy);color:#fff;padding:2px 4px;position:sticky;top:0;font-weight:600;}
  .aptbl td{padding:2px 5px;text-align:center;border:1px solid #fff;}
  .aptbl td.ct{font-weight:700;}
  .grid{stroke:#e6e6e6;stroke-width:1;} .ax{stroke:#999;stroke-width:1;} .tk{fill:#666;font-size:11px;} .al{fill:#1b2430;font-size:12px;}
  .cv{fill:none;stroke-width:1.7;} .ext{fill:none;stroke-width:1.4;stroke-dasharray:5 4;} .dt{stroke:#fff;stroke-width:.5;}
  .bound{stroke:#999;stroke-width:1;stroke-dasharray:2 3;} .guide{stroke:#888;stroke-dasharray:4 3;stroke-width:1;visibility:hidden;}
  .nav{position:absolute;top:1.6%;right:1.6%;display:flex;align-items:center;gap:10px;z-index:50;
       background:rgba(40,60,95,.92);color:#fff;border:1px solid rgba(255,255,255,.35);border-radius:18px;padding:4px 12px;font-size:13px;}
  .nav button{background:none;border:none;color:#fff;font-size:19px;cursor:pointer;line-height:1;padding:0 5px;}
  .nav button:hover{color:#f4a722;}
  #tip{position:fixed;pointer-events:none;background:#111;color:#fff;font-size:11.5px;padding:8px 10px;border-radius:7px;
       max-width:250px;visibility:hidden;z-index:99;line-height:1.4;box-shadow:0 2px 10px rgba(0,0,0,.35);}
  #tip .tt-h{font-weight:700;margin-bottom:5px;padding-bottom:4px;border-bottom:1px solid #555;}
  #tip .tt-r{display:flex;justify-content:space-between;gap:18px;line-height:1.55;}
  #tip .tt-r span{color:#aab2c0;}
  #tip .tt-r b{font-variant-numeric:tabular-nums;}
  .ratebtn{position:absolute;left:3%;bottom:9%;z-index:8;background:#2c5f9e;color:#fff;border:none;
           border-radius:8px;padding:calc(var(--u)*0.55) calc(var(--u)*1.1);font-size:calc(var(--u)*1.05);
           font-weight:700;cursor:pointer;box-shadow:0 3px 10px rgba(0,0,0,.30);display:flex;align-items:center;gap:.45em;}
  .ratebtn:hover{background:#f4a722;color:#14233f;}
  /* modal ocupa só a área de CONTEÚDO do slide (abaixo do título, c/ margens) */
  .modal{position:absolute;left:0;right:0;top:14%;bottom:0;background:rgba(10,16,28,.42);
         display:none;align-items:center;justify-content:center;z-index:60;}
  .modal.on{display:flex;}
  .modalbox{position:relative;background:#fff;border-radius:12px;width:94%;height:94%;overflow:hidden;
            display:flex;flex-direction:column;
            box-shadow:0 10px 40px rgba(0,0,0,.45);padding:calc(var(--u)*1.4) calc(var(--u)*1.8) calc(var(--u)*1.0);}
  .modalbox h3{margin:0 0 2px;color:#14233f;font-size:calc(var(--u)*1.7);}
  .modalbox .sub{color:#5b6675;font-size:calc(var(--u)*1.05);margin-bottom:6px;}
  .modalbox .x{position:absolute;top:calc(var(--u)*0.6);right:calc(var(--u)*1.0);font-size:calc(var(--u)*2.2);
               color:#5b6675;cursor:pointer;background:none;border:none;line-height:1;}
  .modalbox .x:hover{color:#d73027;}
  .ratectrls{display:flex;flex-wrap:wrap;gap:5px;margin:6px 0 4px;}
  .ratectrls button{font-size:12px;padding:2px 9px;border-radius:5px;border:1px solid #bbb;background:#f7f7f7;cursor:pointer;font-weight:600;}
  .ratechips{display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px;}
  .ratechips .chip{width:26px;height:21px;border-radius:4px;border:1.5px solid var(--c);background:var(--c);color:#fff;font-size:11px;font-weight:700;cursor:pointer;padding:0;}
  .ratechips .chip.off{background:#fff;color:#bbb;border-color:#ddd;}
  #svg-rate{flex:1 1 auto;min-height:0;width:100%;}
  .taxwrap{position:absolute;left:3%;right:3%;top:15.5%;bottom:4%;overflow:auto;
           display:flex;flex-direction:column;justify-content:center;}
  .taxhead{display:flex;align-items:center;justify-content:space-between;margin-bottom:.55em;gap:1em;}
  .taxwrap .aptbl{margin:0 auto;}   /* tabela centralizada horizontalmente */
  .taxbtn{background:#2c5f9e;color:#fff;border:none;border-radius:7px;padding:calc(var(--u)*0.4) calc(var(--u)*0.9);
          font-size:calc(var(--u)*1.0);font-weight:700;cursor:pointer;white-space:nowrap;}
  .taxbtn:hover{background:#f4a722;color:#14233f;}
</style></head>
<body>
<div class="deck"><div class="stage" id="stage">
__SLIDES__
  <div class="modal" id="ratemodal" onclick="if(event.target===this)closeRate()">
    <div class="modalbox">
      <button class="x" onclick="closeRate()">×</button>
      <h3>Taxa de desligamento por categoria — 2016 a 2025</h3>
      <div class="sub">Dispensa sem justa causa observada a cada ano. Faixa sombreada = período de modelagem (treino 2021–2024).</div>
      <div class="ratectrls" id="rate-grp"></div>
      <div class="ratechips" id="rate-chips"></div>
      <svg id="svg-rate" viewBox="0 0 920 460" preserveAspectRatio="xMidYMid meet"></svg>
    </div>
  </div>
  <div class="nav"><button onclick="go(-1)" title="anterior (←)">‹</button><span id="counter"></span><button onclick="go(1)" title="próximo (→)">›</button></div>
</div></div>
<div id="tip"></div>
<script>
const DATA=__DATA__, GROUPS=__GROUPS__, RATE=__RATE__;
const IMP=__IMP__, FEATINFO=__FEATINFO__;
const slides=[...document.querySelectorAll('.slide')]; let cur=0;
const counter=document.getElementById('counter');
function show(n){ cur=Math.max(0,Math.min(slides.length-1,n));
  slides.forEach((s,i)=>s.classList.toggle('active',i===cur));
  counter.textContent=(cur+1)+' / '+slides.length; }
function go(d){ show(cur+d); }
document.addEventListener('keydown',e=>{ if(e.key==='ArrowRight'||e.key==='PageDown')go(1);
  else if(e.key==='ArrowLeft'||e.key==='PageUp')go(-1);
  else if(e.key==='Home')show(0); else if(e.key==='End')show(slides.length-1); });
show(0);

const tip=document.getElementById('tip');
function makeChart(svgId, chipsId, grpId, showExt, xmax){
  const svg=document.getElementById(svgId), chips=document.getElementById(chipsId), grp=document.getElementById(grpId);
  if(!svg) return;
  const NS='http://www.w3.org/2000/svg', W=760,HT=470,M={l:54,r:12,t:10,b:38},PW=W-M.l-M.r,PH=HT-M.t-M.b,H=12;
  let yMin=0,yMax=1;
  const xPix=m=>M.l+(m/xmax)*PW, yPix=s=>M.t+(1-(s-yMin)/(yMax-yMin))*PH;
  function el(t,a){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;}
  const visible=new Set(DATA.map(s=>s.k));
  const guide=el('line',{class:'guide',y1:M.t,y2:M.t+PH});
  function domain(){ if(!visible.size){yMin=0;yMax=1;return;} let lo=1;
    DATA.forEach(s=>{ if(!visible.has(s.k))return; for(const v of s.S)if(v<lo)lo=v;
      if(showExt)for(let m=H;m<=xmax;m++)if(s.W[m]<lo)lo=s.W[m]; });
    const pad=0.04*(1-lo)+0.005; yMin=Math.max(0,lo-pad); yMax=1; }
  function axes(){ const range=yMax-yMin,dec=range<0.04?3:2,NT=5;
    for(let i=0;i<=NT;i++){const s=yMin+range*i/NT,y=yPix(s);
      svg.appendChild(el('line',{class:'grid',x1:M.l,y1:y,x2:W-M.r,y2:y}));
      const t=el('text',{class:'tk',x:M.l-6,y:y+3,'text-anchor':'end'});t.textContent=s.toFixed(dec);svg.appendChild(t);}
    const step=xmax>12?3:1;
    for(let m=0;m<=xmax;m+=step){const x=xPix(m);
      svg.appendChild(el('line',{class:'grid',x1:x,y1:M.t,x2:x,y2:M.t+PH}));
      const t=el('text',{class:'tk',x:x,y:M.t+PH+15,'text-anchor':'middle'});t.textContent=m;svg.appendChild(t);}
    if(showExt){const xv=xPix(H);svg.appendChild(el('line',{class:'bound',x1:xv,y1:M.t,x2:xv,y2:M.t+PH}));}
    svg.appendChild(el('line',{class:'ax',x1:M.l,y1:M.t,x2:M.l,y2:M.t+PH}));
    svg.appendChild(el('line',{class:'ax',x1:M.l,y1:M.t+PH,x2:W-M.r,y2:M.t+PH}));
    const yl=el('text',{class:'al','text-anchor':'middle',transform:'translate(14,'+(M.t+PH/2)+') rotate(-90)'});yl.textContent='S(t) = P(seguir empregado)';svg.appendChild(yl);
    const xl=el('text',{class:'al',x:M.l+PW/2,y:HT-4,'text-anchor':'middle'});xl.textContent='MOB — meses desde a entrada';svg.appendChild(xl);
  }
  function lpath(arr,m0,m1){let d='M '+xPix(m0)+' '+yPix(arr[m0]);for(let m=m0+1;m<=m1;m++)d+=' L '+xPix(m)+' '+yPix(arr[m]);return d;}
  function curves(){ DATA.forEach(s=>{ if(!visible.has(s.k))return;
    if(showExt)svg.appendChild(el('path',{class:'ext',d:lpath(s.W,H,xmax),stroke:s.cor}));
    svg.appendChild(el('path',{class:'cv',d:lpath(s.S,0,H),stroke:s.cor}));
    for(let m=0;m<=H;m++)svg.appendChild(el('circle',{class:'dt',cx:xPix(m),cy:yPix(s.S[m]),r:2.4,fill:s.cor})); }); }
  function syncChips(){ chips.querySelectorAll('.chip').forEach(c=>c.classList.toggle('off',!visible.has(+c.dataset.k))); }
  function render(){ svg.innerHTML=''; domain(); axes(); curves(); svg.appendChild(guide); syncChips(); }
  DATA.forEach(s=>{ const b=document.createElement('button'); b.className='chip'; b.dataset.k=s.k; b.textContent=s.k;
    b.style.setProperty('--c',s.cor);
    b.onclick=()=>{ if(visible.has(s.k))visible.delete(s.k); else visible.add(s.k); render(); }; chips.appendChild(b); });
  function gbtn(label,fn,col){ const b=document.createElement('button'); b.textContent=label; if(col){b.style.borderColor=col;b.style.color=col;} b.onclick=fn; grp.appendChild(b); }
  gbtn('Todos',()=>{DATA.forEach(s=>visible.add(s.k));render();});
  gbtn('Nenhum',()=>{visible.clear();render();});
  GROUPS.forEach(g=>gbtn(g.nome,()=>{visible.clear();g.cats.forEach(k=>visible.add(k));render();},g.cor));
  svg.addEventListener('mousemove',ev=>{ const r=svg.getBoundingClientRect(); const sx=(ev.clientX-r.left)*(W/r.width);
    let m=Math.round((sx-M.l)/PW*xmax); m=Math.max(0,Math.min(xmax,m));
    if(sx<M.l-4||sx>W-M.r+4){tip.style.visibility='hidden';guide.style.visibility='hidden';return;}
    guide.setAttribute('x1',xPix(m));guide.setAttribute('x2',xPix(m));guide.style.visibility='visible';
    const val=s=>m<=H?s.S[m]:s.W[m];
    const vis=DATA.filter(s=>visible.has(s.k)).sort((a,b)=>val(b)-val(a));
    if(!vis.length){tip.style.visibility='hidden';return;}
    let html='<b>MOB '+m+'</b>'+(m>H?' (Weibull)':'')+'<br>';
    vis.slice(0,14).forEach(s=>{html+='<span style="color:'+s.cor+'">■</span> Cat '+s.k+': <b>'+(val(s)*100).toFixed(1)+'%</b><br>';});
    tip.innerHTML=html; tip.style.left=Math.min(ev.clientX+12,window.innerWidth-190)+'px'; tip.style.top=(ev.clientY+12)+'px'; tip.style.visibility='visible';
  });
  svg.addEventListener('mouseleave',()=>{tip.style.visibility='hidden';guide.style.visibility='hidden';});
  render();
}
makeChart('svg-km','chips-km','grp-km',false,12);
makeChart('svg-weib','chips-weib','grp-weib',true,36);

function makeBoxChart(){
  const svg=document.getElementById('svg-box'); if(!svg) return;
  const NS='http://www.w3.org/2000/svg', W=600,HT=470,M={l:40,r:10,t:14,b:26},PW=W-M.l-M.r,PH=HT-M.t-M.b;
  const n=DATA.length, fmt=v=>Math.round(v);
  const vmax=Math.max(...DATA.map(s=>s.q3)), vmin=Math.min(...DATA.map(s=>s.q1));
  const l0=Math.log10(Math.max(1,vmin*0.6)), l1=Math.log10(vmax*1.7);
  const yPix=v=>M.t+(1-(Math.log10(Math.max(v,1e-6))-l0)/(l1-l0))*PH;
  const xC=k=>M.l+((k-1)+0.5)/n*PW, bw=0.62*PW/n;
  function el(t,a){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;}
  GROUPS.forEach(g=>{ const x0=xC(g.cats[0])-bw/2-2, x1=xC(g.cats[g.cats.length-1])+bw/2+2;
    svg.appendChild(el('rect',{x:x0,y:M.t,width:x1-x0,height:PH,fill:g.cor,opacity:0.08}));
    const t=el('text',{x:(x0+x1)/2,y:M.t+9,'text-anchor':'middle','font-size':8,'font-weight':'bold',fill:g.cor}); t.textContent='Risco '+g.nome; svg.appendChild(t); });
  [10,100,1000].forEach(v=>{ if(Math.log10(v)<l0||Math.log10(v)>l1) return;
    const y=yPix(v); svg.appendChild(el('line',{x1:M.l,y1:y,x2:W-M.r,y2:y,stroke:'#e6e6e6'}));
    const t=el('text',{x:M.l-5,y:y+3,'text-anchor':'end','font-size':9,fill:'#666'}); t.textContent=v; svg.appendChild(t); });
  [[12,'12m'],[24,'24m'],[36,'36m']].forEach(a=>{ const y=yPix(a[0]); if(y<M.t||y>M.t+PH) return;
    svg.appendChild(el('line',{x1:M.l,y1:y,x2:W-M.r,y2:y,stroke:'#999','stroke-dasharray':'3 2','stroke-width':0.8}));
    const t=el('text',{x:M.l+2,y:y-2,'font-size':7.5,fill:'#777'}); t.textContent=a[1]; svg.appendChild(t); });
  svg.appendChild(el('line',{x1:M.l,y1:M.t,x2:M.l,y2:M.t+PH,stroke:'#999'}));
  svg.appendChild(el('line',{x1:M.l,y1:M.t+PH,x2:W-M.r,y2:M.t+PH,stroke:'#999'}));
  const yl=el('text',{'text-anchor':'middle','font-size':8.5,fill:'#1b2430',transform:'translate(11,'+(M.t+PH/2)+') rotate(-90)'}); yl.textContent='tempo até desligamento (meses, log)'; svg.appendChild(yl);
  const boxes={};
  DATA.forEach(s=>{ const x=xC(s.k), c=s.cor;
    const g=el('g',{class:'box','data-k':s.k});
    g.appendChild(el('rect',{x:x-bw/2,y:yPix(s.q3),width:bw,height:yPix(s.q1)-yPix(s.q3),fill:c,'fill-opacity':0.45,stroke:c,'stroke-width':1,class:'bx'}));
    g.appendChild(el('line',{x1:x-bw/2,y1:yPix(s.medm),x2:x+bw/2,y2:yPix(s.medm),stroke:'#222','stroke-width':1.6}));
    const my=yPix(s.media),d=3; g.appendChild(el('path',{d:'M '+x+' '+(my-d)+' L '+(x+d)+' '+my+' L '+x+' '+(my+d)+' L '+(x-d)+' '+my+' Z',fill:'#fff',stroke:'#222','stroke-width':1}));
    const t=el('text',{x:x,y:M.t+PH+9,'text-anchor':'middle','font-size':7.5,fill:'#666'}); t.textContent=s.k; svg.appendChild(t);
    g.appendChild(el('rect',{x:x-bw/2-1,y:M.t,width:bw+2,height:PH,fill:'transparent'}));
    g.addEventListener('mouseenter',()=>boxHov(s.k,true));
    g.addEventListener('mouseleave',()=>boxHov(s.k,false));
    boxes[s.k]=g; svg.appendChild(g);
  });
  const tb=document.getElementById('boxtable');
  let html='<table><thead><tr><th>cat</th><th>Q1</th><th>mediana</th><th>média</th><th>Q3</th></tr></thead><tbody>';
  DATA.forEach(s=>{ html+='<tr data-k="'+s.k+'"><td class="ct" style="background:'+s.cor+';color:'+s.txt+'">'+s.k+'</td><td>'+fmt(s.q1)+'</td><td>'+fmt(s.medm)+'</td><td>'+fmt(s.media)+'</td><td>'+fmt(s.q3)+'</td></tr>'; });
  tb.innerHTML=html+'</tbody></table>';
  tb.querySelectorAll('tr[data-k]').forEach(r=>{ const k=+r.dataset.k;
    r.addEventListener('mouseenter',()=>boxHov(k,true)); r.addEventListener('mouseleave',()=>boxHov(k,false)); });
  function boxHov(k,on){
    const g=boxes[k]; if(g){ const bx=g.querySelector('.bx'); bx.setAttribute('stroke-width',on?2.6:1); bx.setAttribute('fill-opacity',on?0.7:0.45); }
    const row=tb.querySelector('tr[data-k="'+k+'"]'); if(row) row.classList.toggle('hl',on);
    if(on){ const s=DATA.find(d=>d.k===k), r=svg.getBoundingClientRect(), gr=GROUPS.find(x=>x.cats.includes(k));
      const row2=(l,v)=>'<div class="tt-r"><span>'+l+'</span><b>'+v+'</b></div>';
      tip.innerHTML='<div class="tt-h">Categoria '+k+(gr?' · Risco '+gr.nome:'')+'</div>'+
        row2('Risco em 12 MOB', s.risco12+'%')+
        row2('Q1 (25%)', fmt(s.q1)+' meses')+
        row2('Mediana (50%)', fmt(s.medm)+' meses')+
        row2('Média', fmt(s.media)+' meses')+
        row2('Q3 (75%)', fmt(s.q3)+' meses');
      tip.style.left=Math.min(r.left+xC(k)/W*r.width+10,window.innerWidth-260)+'px';
      tip.style.top=(r.top+yPix(s.q3)/HT*r.height-8)+'px'; tip.style.visibility='visible';
    } else tip.style.visibility='hidden';
  }
}
makeBoxChart();

/* ---------- slide de features: importância clicável (layout do deck anterior) ---------- */
function makeImp(){
  const svg=document.getElementById('svg-imp'); if(!svg) return;
  const NS='http://www.w3.org/2000/svg', W=560,HT=470,M={l:96,r:44,t:8,b:10},PW=W-M.l-M.r,PH=HT-M.t-M.b;
  const n=IMP.length, maxImp=IMP[0].imp, barH=PH/n, bars={}; let sel=null;
  function el(t,a){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;}
  function selFeat(f){
    if(sel&&bars[sel]) bars[sel].querySelector('.ib').setAttribute('fill','#3b7dba');
    sel=f; if(bars[f]) bars[f].querySelector('.ib').setAttribute('fill','#14233f');
    const info=FEATINFO[f]||{}, box=document.getElementById('featinfo');
    const ex=(info.ex||[]).map(e=>'<li>'+e+'</li>').join('');
    box.innerHTML='<div class="fi-h">'+(info.nome||f)+'</div><div class="fi-d">'+(info.desc||'')+'</div>'+(ex?'<ul class="fi-ex">'+ex+'</ul>':'');
  }
  IMP.forEach((d,i)=>{ const y=M.t+i*barH, h=barH*0.74, len=d.imp/maxImp*PW, lab=(FEATINFO[d.f]||{}).curto||d.f;
    const g=el('g',{class:'imp-bar','data-f':d.f});
    g.appendChild(el('rect',{x:M.l,y:y,width:Math.max(len,0.5),height:h,fill:'#3b7dba',rx:2,class:'ib'}));
    const tl=el('text',{x:M.l-5,y:y+h/2+3,'text-anchor':'end','font-size':9.5,fill:'#1b2430'}); tl.textContent=lab; svg.appendChild(tl);
    const tv=el('text',{x:M.l+len+4,y:y+h/2+3,'font-size':9,fill:'#5b6675'}); tv.textContent=d.imp.toFixed(1)+'%'; svg.appendChild(tv);
    g.appendChild(el('rect',{x:M.l,y:y,width:PW,height:h,fill:'transparent'}));
    g.addEventListener('click',()=>selFeat(d.f));
    g.addEventListener('mouseenter',()=>{ if(sel!==d.f) g.querySelector('.ib').setAttribute('fill','#2c5f9e'); });
    g.addEventListener('mouseleave',()=>{ if(sel!==d.f) g.querySelector('.ib').setAttribute('fill','#3b7dba'); });
    bars[d.f]=g; svg.appendChild(g); });
  selFeat(IMP[0].f);
}
makeImp();

/* ---------- modal: taxa de desligamento ao longo dos anos (linhas por categoria) ---------- */
const _rmodal=document.getElementById('ratemodal');
function openRate(){ _rmodal.classList.add('on'); drawRate(); }
function closeRate(){ _rmodal.classList.remove('on'); }
document.addEventListener('keydown',e=>{ if(e.key==='Escape')closeRate(); });
let _rateVis=new Set(RATE.series.map(s=>s.k)), _rateInit=false;
function drawRate(){
  const svg=document.getElementById('svg-rate'); if(!svg) return;
  const NS='http://www.w3.org/2000/svg', W=920,HT=460,M={l:52,r:14,t:14,b:34},PW=W-M.l-M.r,PH=HT-M.t-M.b;
  const anos=RATE.anos, n=anos.length;
  let vmax=0; RATE.series.forEach(s=>{ if(_rateVis.has(s.k)) s.v.forEach(v=>{ if(v>vmax)vmax=v; }); });
  vmax=Math.max(vmax*1.08, 1);
  const xP=i=>M.l+(n<=1?0:i/(n-1))*PW, yP=v=>M.t+(1-v/vmax)*PH;
  function el(t,a){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;}
  svg.innerHTML='';
  // faixa de modelagem
  const i0=anos.indexOf(RATE.treino[0]), i1=anos.indexOf(RATE.treino[1]);
  if(i0>=0&&i1>=0){ const x0=xP(i0), x1=xP(i1);
    svg.appendChild(el('rect',{x:x0,y:M.t,width:x1-x0,height:PH,fill:'#dce6f2',opacity:.7}));
    const tt=el('text',{x:(x0+x1)/2,y:M.t+12,'text-anchor':'middle','font-size':11,'font-weight':'bold',fill:'#5b7da8'}); tt.textContent='treino (2021–24)'; svg.appendChild(tt); }
  // grades + eixos
  const NT=5;
  for(let g=0;g<=NT;g++){ const v=vmax*g/NT, y=yP(v);
    svg.appendChild(el('line',{x1:M.l,y1:y,x2:W-M.r,y2:y,stroke:'#eee'}));
    const t=el('text',{x:M.l-6,y:y+3,'text-anchor':'end','font-size':11,fill:'#666'}); t.textContent=v.toFixed(0)+'%'; svg.appendChild(t); }
  anos.forEach((a,i)=>{ const x=xP(i);
    const t=el('text',{x:x,y:M.t+PH+16,'text-anchor':'middle','font-size':10.5,fill:'#666'}); t.textContent=a; svg.appendChild(t); });
  svg.appendChild(el('line',{x1:M.l,y1:M.t,x2:M.l,y2:M.t+PH,stroke:'#999'}));
  svg.appendChild(el('line',{x1:M.l,y1:M.t+PH,x2:W-M.r,y2:M.t+PH,stroke:'#999'}));
  const yl=el('text',{'text-anchor':'middle','font-size':11,fill:'#1b2430',transform:'translate(14,'+(M.t+PH/2)+') rotate(-90)'}); yl.textContent='taxa de desligamento (%)'; svg.appendChild(yl);
  // linhas
  RATE.series.forEach(s=>{ if(!_rateVis.has(s.k))return;
    let d='M '+xP(0)+' '+yP(s.v[0]); for(let i=1;i<n;i++)d+=' L '+xP(i)+' '+yP(s.v[i]);
    svg.appendChild(el('path',{d:d,fill:'none',stroke:s.cor,'stroke-width':2}));
    for(let i=0;i<n;i++)svg.appendChild(el('circle',{cx:xP(i),cy:yP(s.v[i]),r:2.4,fill:s.cor})); });
  if(!_rateInit){
    const chips=document.getElementById('rate-chips'), grp=document.getElementById('rate-grp');
    RATE.series.forEach(s=>{ const b=document.createElement('button'); b.className='chip'; b.textContent=s.k; b.style.setProperty('--c',s.cor);
      b.onclick=()=>{ if(_rateVis.has(s.k))_rateVis.delete(s.k); else _rateVis.add(s.k); syncRchips(); drawRate(); }; chips.appendChild(b); });
    const gb=(lab,fn,col)=>{ const b=document.createElement('button'); b.textContent=lab; if(col){b.style.borderColor=col;b.style.color=col;} b.onclick=fn; grp.appendChild(b); };
    gb('Todas',()=>{RATE.series.forEach(s=>_rateVis.add(s.k));syncRchips();drawRate();});
    gb('Nenhuma',()=>{_rateVis.clear();syncRchips();drawRate();});
    GROUPS.forEach(g=>gb(g.nome,()=>{_rateVis.clear();g.cats.forEach(k=>_rateVis.add(k));syncRchips();drawRate();},g.cor));
    _rateInit=true; syncRchips();
  }
}
function syncRchips(){ document.querySelectorAll('#rate-chips .chip').forEach(c=>c.classList.toggle('off',!_rateVis.has(+c.textContent))); }

/* ---------- tabela de taxas: alterna entre % ao mês e % ao ano ---------- */
let _taxAnual=false;
function toggleTaxa(){
  _taxAnual=!_taxAnual;
  document.getElementById('tax-m').style.display=_taxAnual?'none':'';
  document.getElementById('tax-a').style.display=_taxAnual?'':'none';
  document.getElementById('tax-h').textContent='Taxa mínima para recuperar o principal — '+(_taxAnual?'% ao ano':'% ao mês');
  document.getElementById('tax-btn').textContent=_taxAnual?'← Ver % ao mês':'Ver % ao ano →';
}

/* ---------- tabela NPV: dois toggles (ROI 10/20 e mês/ano) ---------- */
let _npvRoi='10', _npvPer='m';
function _npvRender(){
  for(const r of ['10','20']) for(const p of ['m','a']){
    const el=document.getElementById('npv-'+r+p); if(el) el.style.display=(r===_npvRoi&&p===_npvPer)?'':'none';
  }
  document.getElementById('npv-h').textContent='Taxa de pricing — ROI '+_npvRoi+'% · '+(_npvPer==='a'?'% ao ano':'% ao mês');
  document.getElementById('npv-roi').textContent=_npvRoi==='10'?'Ver ROI 20% →':'← Ver ROI 10%';
  document.getElementById('npv-per').textContent=_npvPer==='a'?'← Ver % ao mês':'Ver % ao ano →';
}
function toggleNpvRoi(){ _npvRoi=_npvRoi==='10'?'20':'10'; _npvRender(); }
function toggleNpvPer(){ _npvPer=_npvPer==='m'?'a':'m'; _npvRender(); }
</script>
</body></html>"""

HTML = (HTML.replace("__FONTS__", FONTS).replace("__SLIDES__", SLIDES)
            .replace("__DATA__", DATA).replace("__GROUPS__", GROUPS_JSON)
            .replace("__RATE__", RATE_JSON)
            .replace("__IMP__", IMP_JSON).replace("__FEATINFO__", FEATINFO_JSON))
with open(TMP, "w", encoding="utf-8") as f:
    f.write(HTML)
shutil.copy(TMP, OUT)
print(f"FIM -> {OUT} ({len(HTML)/1024/1024:.1f} MB, {NP} slides, interativos B1/B2/B3 + tabela C1)")
