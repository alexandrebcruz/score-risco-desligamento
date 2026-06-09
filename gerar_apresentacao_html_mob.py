"""Converte a apresentação (gerar_apresentacao.py) para um DECK em HTML autossuficiente
(offline, sem CDN), com navegação por teclado/botões.

- Slides estáticos: renderizados pelo próprio deck (matplotlib) e embutidos como PNG base64.
- Slides B1 e B2 (curvas de sobrevivência e extrapolação Weibull): viram INTERATIVOS —
  o usuário escolhe quais categorias plotar (chips por categoria + botões por grupo de risco),
  com escala-Y dinâmica e tooltip, no mesmo espírito de outputs/sobrevivencia_interativa.html.

Uso:  MPLCONFIGDIR=/tmp/mpl /tmp/consig_venv/bin/python gerar_apresentacao_html.py
Saída: outputs/apresentacao_risco_desligamento.html
"""
import os, runpy, json, shutil, re
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import pandas as pd
import base64
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none"   # texto como <text> (SELECIONÁVEL); fonte embutida via @font-face
from matplotlib import cm, colors

DUMP = "/tmp/apresentacao_svg"
os.environ["DECK_DUMP_PNG"] = DUMP      # diretório do dump
os.environ["DECK_DUMP_FMT"] = "svg"     # exporta cada slide como SVG vetorial (não PNG)
OUT = "outputs/apresentacao_risco_desligamento_mob.html"
TMP = "/tmp/apresentacao_risco_desligamento_mob.html"

# ---------- 1. roda o deck (gera PDF + dump dos slides em SVG) ----------
print("renderizando slides via gerar_apresentacao.py ...")
ns = runpy.run_path("gerar_apresentacao.py")
NP = len(ns["pages"])
B1, B2, B3 = NP - 3, NP - 2, NP - 1   # slides interativos: surv_curva, surv_weibull, surv_estatisticas
DESEMP = 4                            # slide de desempenho (calibração + importância) — 5º slide
print(f"{NP} slides; interativos: desempenho={DESEMP}, B1={B1}, B2={B2}, B3={B3}")

def inline_svg(path, pfx):
    """Insere o SVG do matplotlib inline, escalado p/ preencher o slide. Prefixa todos os
    ids/refs com `pfx` (evita colisão entre os 26 SVGs no mesmo documento) e remove o
    <style>*{}</style> global (reposto escopado no CSS)."""
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

# ---------- 2. dados de sobrevivência (mesma fonte do HTML interativo) ----------
km = pd.read_csv("outputs/tables/sobrevivencia_km_mob_2023.csv").rename(columns={"mob": "mes"})
ext = pd.read_csv("outputs/tables/sobrevivencia_weibull_extrap_mob_2023.csv")
mono = pd.read_csv("outputs/tables/sobrevivencia_weibull_estatisticas_mono_mob_2023.csv").set_index("categoria")
res = pd.read_csv("outputs/tables/sobrevivencia_resumo_mob_2023.csv").set_index("categoria")
ks = sorted(km["categoria"].unique())
# cores idênticas à figura (RdYlGn_r normalizado em [min,max] das categorias)
cmap = cm.get_cmap("RdYlGn_r"); norm = colors.Normalize(vmin=min(ks), vmax=max(ks))
cor = {k: colors.to_hex(cmap(norm(k))) for k in ks}
def _contraste(hx):                      # cor de texto legível sobre qualquer fundo (luminância)
    r, g, b = (int(hx[i:i + 2], 16) for i in (1, 3, 5))
    return "#000000" if 0.299 * r + 0.587 * g + 0.114 * b > 150 else "#ffffff"

series = []
for k in ks:
    S = [round(float(v), 5) for v in km[km.categoria == k].sort_values("mes")["S"].tolist()]
    W = [round(float(v), 5) for v in ext[ext.categoria == k].sort_values("mes")["S_weibull"].tolist()]
    mo = mono.loc[k]
    series.append({"k": int(k), "cor": cor[k], "txt": _contraste(cor[k]), "S": S, "W": W,
                   "risco12": round(float(res.loc[k, "risco_deslig_12m_KM"]) * 100, 1),
                   "q1": round(float(mo["q1_meses_mono"]), 1),
                   "medm": round(float(mo["mediana_meses_mono"]), 1),
                   "media": round(float(mo["media_meses_mono"]), 1),
                   "q3": round(float(mo["q3_meses_mono"]), 1)})
DATA = json.dumps(series, ensure_ascii=False)
GROUPS = [("Mínimo", [1, 2], "#1a9850"), ("Baixo", [3, 4, 5, 6], "#86cb66"),
          ("Médio-Baixo", [7, 8, 9, 10, 11], "#c9a227"),
          ("Médio", [12, 13, 14, 15, 16, 17], "#fb8d3d"),
          ("Alto", [18, 19, 20, 21, 22, 23], "#d73027")]
GROUPS_JSON = json.dumps([{"nome": n, "cats": c, "cor": col} for n, c, col in GROUPS], ensure_ascii=False)

# ---------- dados do slide 5 (desempenho): calibração + importância + dicionário de features ----------
_cal = pd.read_csv("outputs/runpod_ensemble_base/calibracao_ensemble.csv")
CALIB_JSON = json.dumps([{"p": round(float(r.prevista), 5), "o": round(float(r.observada), 5)}
                         for r in _cal.itertuples()], ensure_ascii=False)
_imp = pd.read_csv("outputs/runpod_ensemble_base/importancia_ensemble.csv").sort_values("imp_ensemble", ascending=False)
IMP_JSON = json.dumps([{"f": r.feature, "imp": round(float(r.imp_ensemble), 2)} for r in _imp.itertuples()], ensure_ascii=False)

# explicação de cada variável + exemplos de valores (códigos traduzidos pelo de-para da RAIS)
FEATINFO = {
 "tempo_vinculo_meses": {"curto": "Tempo de vínculo", "nome": "Tempo de vínculo (meses)",
   "desc": "Há quantos meses a pessoa está no emprego atual. Variável mais importante — vínculos novos têm risco muito maior.",
   "ex": ["Numérico, em meses", "Ex.: 3, 12, 60, 120"]},
 "tipo_vinculo": {"curto": "Tipo de vínculo", "nome": "Tipo de vínculo",
   "desc": "Natureza jurídica do contrato de trabalho (RAIS).",
   "ex": ["10/15/20/25 = CLT prazo indeterminado", "30/31/35 = estatutário (servidor)",
          "50/55/60/65 = temporário / prazo determinado", "97 = contrato Verde e Amarelo"]},
 "faixa_remuneracao": {"curto": "Remuneração", "nome": "Faixa de remuneração (salários mínimos)",
   "desc": "Faixa do salário, em salários mínimos.",
   "ex": ["0–2 = até 1 SM", "3–7 = de 1 a 5 SM", "8–12 = acima de 5 SM"]},
 "tamanho_estab": {"curto": "Tamanho estab.", "nome": "Tamanho do estabelecimento",
   "desc": "Faixa de número de vínculos do estabelecimento (porte do empregador).",
   "ex": ["1 = zero / 2 = de 1 a 4", "5 = de 50 a 99", "10 = 1000 ou mais"]},
 "natureza_juridica": {"curto": "Nat. jurídica", "nome": "Natureza jurídica do empregador",
   "desc": "Tipo jurídico da empresa/órgão (tabela da Receita Federal).",
   "ex": ["2062 = Sociedade Empresária Ltda.", "2135 = Empresário individual (MEI)",
          "1023 = Órgão público municipal", "3999 = Associação privada"]},
 "uf": {"curto": "UF", "nome": "UF (Unidade da Federação)",
   "desc": "Estado onde está o vínculo.",
   "ex": ["SP, MG, RJ, BA, RS…", "(ou código IBGE: 35=SP, 31=MG, 33=RJ)"]},
 "qtd_dias_afastamento": {"curto": "Dias afast.", "nome": "Dias de afastamento",
   "desc": "Quantidade de dias afastado no ano.", "ex": ["Numérico, em dias", "Ex.: 0, 15, 120"]},
 "natureza_setor": {"curto": "Nat. setor", "nome": "Natureza do setor",
   "desc": "Distingue setor público de privado.", "ex": ["1 = setor público", "2/3/4 = setor privado / outros"]},
 "cnae2": {"curto": "CNAE-2", "nome": "CNAE — divisão (2 dígitos)",
   "desc": "Atividade econômica do empregador, no nível de divisão.",
   "ex": ["41 = Construção de edifícios", "47 = Comércio varejista",
          "84 = Administração pública", "86 = Saúde humana"]},
 "cbo1": {"curto": "CBO-1", "nome": "CBO — grande grupo (1 dígito)",
   "desc": "Ocupação da pessoa, no nível mais agregado.",
   "ex": ["3 = técnicos de nível médio", "5 = serviços e vendas",
          "7 = produção de bens (indústria)", "9 = manutenção e reparação"]},
 "cbo2": {"curto": "CBO-2", "nome": "CBO — subgrupo principal (2 dígitos)",
   "desc": "Ocupação no nível de subgrupo.",
   "ex": ["51 = serviços / 52 = vendas no comércio", "71 = construção civil e extração", "78 = condutores e operadores"]},
 "faixa_horas": {"curto": "Horas", "nome": "Faixa de horas contratadas",
   "desc": "Faixa de jornada semanal contratada.", "ex": ["41–44h = jornada integral", "≈ 20–30h = parcial"]},
 "cbo4": {"curto": "CBO-4", "nome": "CBO — família ocupacional (4 dígitos)",
   "desc": "Ocupação no nível de família.",
   "ex": ["7152 = pedreiros", "5173 = vigias/porteiros", "4110 = auxiliares de escritório"]},
 "cbo": {"curto": "CBO-6", "nome": "CBO — ocupação (6 dígitos)",
   "desc": "Código completo da ocupação.",
   "ex": ["715210 = Pedreiro", "517330 = Porteiro", "521110 = Vendedor de comércio varejista"]},
 "cnae3": {"curto": "CNAE-3", "nome": "CNAE — grupo (3 dígitos)",
   "desc": "Atividade econômica no nível de grupo.",
   "ex": ["412 = Construção de edifícios", "471 = Comércio varejista não especializado"]},
 "cnae": {"curto": "CNAE-6", "nome": "CNAE — classe/subclasse (completo)",
   "desc": "Código completo da atividade econômica.",
   "ex": ["4120400 = Construção de edifícios", "4711301 = Hipermercados"]},
 "idade": {"curto": "Idade", "nome": "Idade (anos)", "desc": "Idade da pessoa.",
   "ex": ["Numérico, em anos", "Ex.: 18, 30, 45, 60"]},
 "cnae5": {"curto": "CNAE-5", "nome": "CNAE — classe (5 dígitos)",
   "desc": "Atividade econômica no nível de classe.",
   "ex": ["41204 = Construção de edifícios", "47113 = Comércio varejista de mercadorias"]},
 "causa_afastamento": {"curto": "Causa afast.", "nome": "Causa de afastamento",
   "desc": "Motivo do afastamento registrado no ano.",
   "ex": ["99 = sem afastamento (≈84% dos casos)", "01 = acidente de trabalho", "03 = doença", "+ licenças diversas"]},
 "escolaridade": {"curto": "Escolaridade", "nome": "Escolaridade (grau de instrução)",
   "desc": "Grau de instrução da pessoa (RAIS, grau_instrucao 1–11).",
   "ex": ["1–4 = até fundamental incompleto", "8 = médio completo", "9 = superior incompleto", "11 = superior completo"]},
 "simples": {"curto": "Simples", "nome": "Optante pelo Simples Nacional",
   "desc": "Se o empregador é optante do Simples Nacional.", "ex": ["1 = sim (optante)", "0 = não"]},
 "intermitente": {"curto": "Intermitente", "nome": "Contrato intermitente",
   "desc": "Se o vínculo é de trabalho intermitente.", "ex": ["1 = sim", "0 = não"]},
}
FEATINFO_JSON = json.dumps(FEATINFO, ensure_ascii=False)

# fonte DejaVu embutida (texto fica SELECIONÁVEL e com métricas idênticas às do matplotlib)
_TTF = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
def _face(fname, weight):
    b = base64.b64encode(open(os.path.join(_TTF, fname), "rb").read()).decode()
    return ("@font-face{font-family:'DejaVu Sans';font-style:normal;font-weight:%s;"
            "src:url(data:font/ttf;base64,%s) format('truetype');}" % (weight, b))
FONTS = _face("DejaVuSans.ttf", "400") + _face("DejaVuSans-Bold.ttf", "700")

# ---------- 3. slides interativos (header + bullets + área do gráfico) ----------
NAVY = "#14233f"
def bullets_html(items):
    out = []
    for b, t in items:
        if b:
            out.append(f'<div class="b"><span class="bi">▸</span>{t}</div>')
        else:
            out.append(f'<div class="bh">{t}</div>')
    return "\n".join(out)

B1_TXT = bullets_html([
    (False, "A ideia"),
    (True, "O modelo prevê QUEM/SE é desligado; a sobrevivência mede QUANDO."),
    (True, "S(t) = probabilidade de continuar empregado após t meses."),
    (False, "Dos microdados (RAIS)"),
    (True, "Relógio = MOB (meses desde a ENTRADA na janela); evento = dispensa s/ justa causa."),
    (True, "Censura: quem fica ativo (ou sai por outro motivo) não é 'evento'."),
    (False, "Kaplan–Meier"),
    (True, "S(t) = Π (nₘ−dₘ)/nₘ — usa a censura sem viés, mês a mês."),
    (True, "RMST(12) = área sob S(t) = meses esperados de emprego no ano."),
])
B2_TXT = bullets_html([
    (False, "O problema"),
    (True, "12 meses de dado não enxergam além de 12m (a curva ainda está alta)."),
    (False, "Solução: forma paramétrica de Weibull"),
    (True, "S(t) = exp(−(t/λ)ᵖ);  hazard ∝ t^(p−1)  (p>1 sobe, p<1 cai)."),
    (True, "Ajuste por regressão pura: ln(−ln S) = p·ln t + ln α (OLS, 12 pts)."),
    (True, "R² médio ≈ 0,99 — extrapola a curva até 36 meses (tracejado)."),
    (False, "Ressalva"),
    (True, "Ignora a sazonalidade de dezembro; projeção >12m é suposição."),
])

def interactive_slide(kicker, title, txt, chart_id):
    return f'''<div class="slide cust">
  <div class="hb"><span class="kick">{kicker}</span><span class="ttl">{title}</span></div>
  <div class="txt">{txt}</div>
  <div class="chartwrap">
    <div class="ctrls">
      <div class="grp" id="grp-{chart_id}"></div>
      <div class="chips" id="chips-{chart_id}"></div>
    </div>
    <svg id="svg-{chart_id}" viewBox="0 0 760 470" preserveAspectRatio="xMidYMid meet"></svg>
  </div>
</div>'''

def box_slide():
    return '''<div class="slide cust">
  <div class="hb"><span class="kick">TEMPO ATÉ O DESLIGAMENTO · ESTATÍSTICAS</span><span class="ttl">Q1, mediana, média e Q3 por categoria — visão MOB (meses)</span></div>
  <div class="boxwrap"><svg id="svg-box" viewBox="0 0 600 470" preserveAspectRatio="xMidYMid meet"></svg></div>
  <div class="boxtable" id="boxtable"></div>
  <div class="boxhint">Passe o mouse numa categoria (caixa ou linha) para ver os dados</div>
</div>'''

def desempenho_slide():
    return '''<div class="slide cust">
  <div class="hb"><span class="kick">DESENVOLVIMENTO · RESULTADO</span><span class="ttl">Desempenho: discrimina bem e é bem calibrado</span></div>
  <div class="calibwrap"><svg id="svg-calib" viewBox="0 0 380 340" preserveAspectRatio="xMidYMid meet"></svg>
    <div class="capt">Calibração: risco previsto ≈ observado nos decis (erro &lt; 1 p.p.).</div></div>
  <div class="featinfo" id="featinfo"><div class="fi-h">Importância das variáveis</div><div class="fi-d">Clique numa barra ao lado para ver o que é a variável e exemplos de valores.</div></div>
  <div class="impwrap"><svg id="svg-imp" viewBox="0 0 560 470" preserveAspectRatio="xMidYMid meet"></svg></div>
</div>'''

# ---------- slides extra (HTML-only): aplicação ao crédito consignado ----------
import math
_kmS = {(int(r.categoria), int(r.mes)): float(r.S) for r in km.itertuples()}
def _S(c, m):
    if m <= 12: return _kmS[(c, m)]
    lam, pp = float(mono.loc[c, "escala_lambda_meses"]), float(mono.loc[c, "shape_p"])
    return math.exp(-(m / lam) ** pp)                       # >12m = extrapolação Weibull
def _termo(c, conf):                                        # prazo p/ confiança conf de seguir empregado
    lam, pp = float(mono.loc[c, "escala_lambda_meses"]), float(mono.loc[c, "shape_p"])
    return lam * (-math.log(conf)) ** (1.0 / pp)
def _cov(c, T): return sum(_S(c, m) for m in range(1, T + 1)) / T * 100   # % das T parcelas esperadas pagas
def _heat(frac): return "hsl(%d,62%%,87%%)" % round(max(0.0, min(1.0, frac)) * 120)  # vermelho->verde
CONFS = [("95%", 0.95), ("90%", 0.90), ("85%", 0.85), ("80%", 0.80)]
TS = [6, 12, 18, 24, 36, 48, 60]
def _catcell(k): return '<td class="ct" style="background:%s;color:%s">%d</td>' % (cor[k], _contraste(cor[k]), k)
def _termo_tbl():
    h = "<tr><th>cat</th>" + "".join("<th>%s</th>" % l for l, _ in CONFS) + "</tr>"
    body = ""
    for k in ks:
        cells = ""
        for _, c in CONFS:
            t = _termo(k, c); disp = "120+" if t > 120 else "%.0f" % t
            cells += '<td style="background:%s">%s</td>' % (_heat(min(t, 36) / 36), disp)
        body += "<tr>" + _catcell(k) + cells + "</tr>"
    return '<table class="aptbl"><thead>%s</thead><tbody>%s</tbody></table>' % (h, body)
def _cov_tbl():
    h = "<tr><th>cat</th>" + "".join("<th>T=%d</th>" % t for t in TS) + "</tr>"
    body = ""
    for k in ks:
        cells = "".join('<td style="background:%s">%.0f%%</td>' % (_heat(_cov(k, t) / 100), _cov(k, t)) for t in TS)
        body += "<tr>" + _catcell(k) + cells + "</tr>"
    return '<table class="aptbl"><thead>%s</thead><tbody>%s</tbody></table>' % (h, body)
TERMO_TBL, COV_TBL = _termo_tbl(), _cov_tbl()

_INTRO_A = bullets_html([
    (False, "A ligação com o consignado"),
    (True, "No consignado, a parcela é descontada direto da folha de pagamento."),
    (True, "Se a pessoa perde o emprego, o desconto para → risco de não pagamento."),
    (True, "S(t) = P(seguir empregado após t meses) = P(o desconto seguir ativo)."),
    (True, "Logo, o prazo deve caber na expectativa de permanência no emprego."),
])
_INTRO_B = bullets_html([
    (False, "Como definir o prazo máximo"),
    (True, "Confiança: prazo = t em que S(t)=c (ex.: 90%) — quantil do tempo até a dispensa."),
    (True, "Cobertura: parcelas esperadas pagas = Σ S(m); exija ≥ β% das T parcelas."),
    (True, "Risco-tier: faixas baixas → prazos longos; altas → curtos ou rejeição."),
    (False, "Ressalvas"),
    (True, ">12 meses é extrapolação; refine com LGD e saldo devedor decrescente."),
])
# ilustração vetorial (contracheque -> R$ -> banco -> relógio -> curva S(t)) p/ preencher o slide
CONSIG_ART = '''<svg viewBox="0 0 980 200" preserveAspectRatio="xMidYMid meet">
  <rect x="14" y="18" width="200" height="150" rx="10" fill="#fff" stroke="#cdd5df" stroke-width="2"/>
  <rect x="14" y="18" width="200" height="30" rx="10" fill="#14233f"/>
  <text x="114" y="39" text-anchor="middle" fill="#fff" font-size="14" font-weight="700">CONTRACHEQUE</text>
  <rect x="30" y="62" width="120" height="9" rx="4" fill="#e1e7ee"/>
  <rect x="30" y="80" width="155" height="9" rx="4" fill="#e1e7ee"/>
  <rect x="30" y="98" width="95" height="9" rx="4" fill="#e1e7ee"/>
  <rect x="24" y="120" width="180" height="34" rx="6" fill="#fdebc9" stroke="#f4a722" stroke-width="1.5"/>
  <text x="34" y="142" fill="#b9791a" font-size="12.5" font-weight="700">− parcela do consignado</text>
  <text x="114" y="192" text-anchor="middle" fill="#5b6675" font-size="13">Desconto direto na folha</text>
  <line x1="226" y1="92" x2="300" y2="92" stroke="#3b7dba" stroke-width="3"/>
  <path d="M300 92 l-11 -7 v14 z" fill="#3b7dba"/>
  <circle cx="263" cy="64" r="15" fill="#f4a722"/><text x="263" y="69" text-anchor="middle" font-size="13" font-weight="700" fill="#fff">R$</text>
  <path d="M320 60 L420 26 L520 60 Z" fill="#14233f"/>
  <rect x="326" y="60" width="188" height="10" fill="#14233f"/>
  <rect x="344" y="72" width="16" height="74" fill="#9fb0c6"/><rect x="376" y="72" width="16" height="74" fill="#9fb0c6"/>
  <rect x="408" y="72" width="16" height="74" fill="#9fb0c6"/><rect x="440" y="72" width="16" height="74" fill="#9fb0c6"/>
  <rect x="472" y="72" width="16" height="74" fill="#9fb0c6"/>
  <rect x="320" y="146" width="200" height="12" fill="#14233f"/>
  <text x="420" y="192" text-anchor="middle" fill="#5b6675" font-size="13">Crédito consignado</text>
  <line x1="540" y1="92" x2="612" y2="92" stroke="#3b7dba" stroke-width="3"/>
  <path d="M612 92 l-11 -7 v14 z" fill="#3b7dba"/>
  <circle cx="690" cy="92" r="50" fill="#fff" stroke="#3b7dba" stroke-width="4"/>
  <line x1="690" y1="92" x2="690" y2="58" stroke="#14233f" stroke-width="4" stroke-linecap="round"/>
  <line x1="690" y1="92" x2="715" y2="104" stroke="#14233f" stroke-width="4" stroke-linecap="round"/>
  <circle cx="690" cy="92" r="4" fill="#14233f"/>
  <text x="690" y="192" text-anchor="middle" fill="#5b6675" font-size="13">Prazo: até quando paga</text>
  <line x1="792" y1="42" x2="792" y2="150" stroke="#bbb" stroke-width="1.5"/>
  <line x1="792" y1="150" x2="956" y2="150" stroke="#bbb" stroke-width="1.5"/>
  <path d="M792 48 C 854 54, 902 112, 956 146" fill="none" stroke="#d73027" stroke-width="3.5"/>
  <text x="804" y="42" fill="#d73027" font-size="13" font-weight="700">S(t)</text>
  <text x="874" y="192" text-anchor="middle" fill="#5b6675" font-size="13">define o prazo máximo</text>
</svg>'''
def consig_intro_slide():
    return ('<div class="slide cust"><div class="hb"><span class="kick">APLICAÇÃO · CRÉDITO CONSIGNADO</span>'
            '<span class="ttl">Usando a sobrevivência para definir o prazo do consignado</span></div>'
            '<div class="consigbody">'
            '<div class="consigcols"><div>' + _INTRO_A + '</div><div>' + _INTRO_B + '</div></div>'
            '<div class="consigart">' + CONSIG_ART + '</div>'
            '</div></div>')
def consig_tables_slide():
    return ('<div class="slide cust"><div class="hb"><span class="kick">APLICAÇÃO · CRÉDITO CONSIGNADO</span>'
            '<span class="ttl">Prazo máximo e cobertura de parcelas por categoria</span></div>'
            '<div class="aptwrap-l"><div class="apt-h">Prazo máx. (meses) por confiança de seguir empregado</div>' + TERMO_TBL + '</div>'
            '<div class="aptwrap-r"><div class="apt-h">Cobertura esperada de parcelas (% pagas em folha) por prazo T</div>' + COV_TBL +
            '<div class="apt-note">T em meses · &gt;12m extrapolado (Weibull) · verde = melhor cobertura</div></div></div>')

# ---------- 4. monta todos os slides ----------
slides = []
for i in range(NP):
    if i == B1:
        slides.append(interactive_slide("TEMPO ATÉ O DESLIGAMENTO · SOBREVIVÊNCIA",
                      "Curvas de sobrevivência por categoria — visão MOB (Kaplan-Meier)", B1_TXT, "km"))
    elif i == B2:
        slides.append(interactive_slide("TEMPO ATÉ O DESLIGAMENTO · EXTRAPOLAÇÃO",
                      "Extrapolação Weibull das curvas MOB (até 36 MOB)", B2_TXT, "weib"))
    elif i == B3:
        slides.append(box_slide())
    elif i == DESEMP:
        slides.append(desempenho_slide())
    else:
        slides.append(f'<div class="slide">{inline_svg(f"{DUMP}/slide_{i:02d}.svg", f"s{i:02d}_")}</div>')
slides.append(consig_intro_slide())      # slides extra ao final (HTML-only)
slides.append(consig_tables_slide())
SLIDES = "\n".join(slides)

# ---------- 5. template HTML ----------
HTML = r"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8">
<title>Risco de Desligamento — apresentação (visão MOB)</title>
<style>
  __FONTS__
  :root{ --navy:#14233f; --ink:#1b2430; --grey:#5b6675; }
  *{box-sizing:border-box;} html,body{margin:0;height:100%;background:#0d1626;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;}
  .deck{height:100vh;height:100dvh;display:flex;align-items:center;justify-content:center;}
  /* --u = 1% da LARGURA do palco -> fontes proporcionais ao slide, iguais ao matplotlib */
  .stage{position:relative;width:min(100vw,177.78vh);height:min(56.25vw,100vh);
         width:min(100vw,177.78dvh);height:min(56.25vw,100dvh);--u:min(1vw,1.7778vh);
         background:#fff;box-shadow:0 6px 30px rgba(0,0,0,.5);overflow:hidden;}
  .slide{position:absolute;inset:0;display:none;}
  .slide.active{display:block;}
  .full{width:100%;height:100%;object-fit:contain;}
  /* SVG vetorial de cada slide estático (substitui os PNGs): preenche o slide, layout idêntico */
  .deckslide{position:absolute;inset:0;width:100%;height:100%;}
  .deckslide *{stroke-linejoin:round;stroke-linecap:butt;}
  /* slide interativo */
  .cust{background:#fff;}
  .hb{position:absolute;top:0;left:0;right:0;height:14%;background:var(--navy);border-left:6px solid #f4a722;
      display:flex;flex-direction:column;justify-content:center;padding-left:2.6%;}
  /* tamanhos = mesmos pontos do matplotlib (fig 13,33in): título 20pt, kicker 11,5pt, bullets 12,6pt */
  .kick{color:#9fc0e8;font-weight:bold;letter-spacing:.04em;font-size:calc(var(--u)*1.20);}
  .ttl{color:#fff;font-weight:bold;font-size:calc(var(--u)*2.08);}
  .txt{position:absolute;left:3.5%;top:19%;width:38%;}
  .bh{font-weight:bold;color:var(--ink);font-size:calc(var(--u)*1.31);margin:calc(var(--u)*0.95) 0 calc(var(--u)*0.2);}
  .b{color:var(--ink);font-size:calc(var(--u)*1.31);margin:calc(var(--u)*0.28) 0;padding-left:1.5em;text-indent:-1.5em;line-height:1.3;}
  .bi{color:#f4a722;font-weight:bold;margin-right:.5em;}
  .chartwrap{position:absolute;left:43%;top:16%;width:55%;height:80%;display:flex;flex-direction:column;}
  .ctrls{flex:0 0 auto;margin-bottom:.4vh;}
  /* controles em unidades --u (1% da largura do palco) -> escalam junto com o slide */
  .grp{display:flex;flex-wrap:wrap;gap:calc(var(--u)*0.35);margin-bottom:calc(var(--u)*0.35);}
  .grp button{font-size:calc(var(--u)*0.92);padding:calc(var(--u)*0.12) calc(var(--u)*0.6);border-radius:5px;border:1px solid #bbb;background:#f7f7f7;cursor:pointer;font-weight:600;}
  .chips{display:flex;flex-wrap:wrap;gap:calc(var(--u)*0.2);}
  .chip{width:calc(var(--u)*1.5);height:calc(var(--u)*1.5);border-radius:4px;border:1.5px solid var(--c);
        background:var(--c);color:#fff;font-size:calc(var(--u)*0.82);font-weight:700;cursor:pointer;padding:0;line-height:1;}
  .chip.off{background:#fff;color:#bbb;border-color:#ddd;}
  .chartwrap svg{flex:1 1 auto;width:100%;min-height:0;}
  /* slide B3: gráfico-caixa interativo + tabela */
  .boxwrap{position:absolute;left:2%;top:15.5%;width:60%;height:82%;}
  .boxwrap svg{width:100%;height:100%;}
  .box{cursor:pointer;}
  .boxtable{position:absolute;right:2.5%;top:15.5%;width:33%;height:80%;overflow:auto;
            display:flex;flex-direction:column;justify-content:center;}
  .boxtable table{border-collapse:collapse;width:100%;font-size:calc(var(--u)*0.95);font-variant-numeric:tabular-nums;}
  .boxtable th{background:var(--navy);color:#fff;padding:calc(var(--u)*0.15) calc(var(--u)*0.25);position:sticky;top:0;}
  .boxtable td{padding:calc(var(--u)*0.1) calc(var(--u)*0.25);text-align:center;border-bottom:1px solid #eee;}
  .boxtable td.ct{color:#fff;font-weight:700;}
  .boxtable tr.hl td{background:#fff3cf;}
  .boxtable tr.hl td.ct{filter:brightness(.85);}
  .boxhint{position:absolute;left:2%;bottom:2.5%;color:var(--grey);font-size:calc(var(--u)*0.9);}
  /* slide 5: desempenho (calibração SVG + importância interativa) */
  .calibwrap{position:absolute;left:2.5%;top:15.5%;width:37%;height:82%;}
  .calibwrap svg{width:100%;height:92%;}
  .capt{font-size:calc(var(--u)*0.82);color:var(--grey);margin-top:.3em;}
  /* box de detalhes flutua POR CIMA do gráfico de importância (economiza espaço) */
  .featinfo{position:absolute;right:2.5%;bottom:4%;width:32%;max-height:60%;overflow:auto;z-index:6;
            background:rgba(255,255,255,.97);border:1px solid #cdd5df;border-radius:8px;
            padding:calc(var(--u)*0.7);box-shadow:0 3px 14px rgba(0,0,0,.20);}
  .featinfo .fi-h{font-weight:700;font-size:calc(var(--u)*1.05);color:var(--navy);margin-bottom:.3em;}
  .featinfo .fi-d{font-size:calc(var(--u)*0.92);color:var(--ink);line-height:1.4;}
  .featinfo .fi-ex{margin-top:.5em;font-size:calc(var(--u)*0.9);}
  .featinfo .fi-ex li{margin:.15em 0;color:#33404f;}
  .impwrap{position:absolute;left:41%;top:14.5%;width:57%;height:83%;}
  .impwrap svg{width:100%;height:100%;}
  .imp-bar{cursor:pointer;}
  /* slides de aplicação ao consignado */
  /* corpo do slide centralizado verticalmente na área de conteúdo */
  .consigbody{position:absolute;left:4%;right:4%;top:14%;bottom:3%;display:flex;flex-direction:column;
              justify-content:center;gap:4.5%;}
  .consigcols{display:flex;gap:5%;align-items:flex-start;}
  .consigcols > div{flex:1 1 0;}
  .consigart{width:100%;}
  .consigart svg{display:block;width:88%;height:auto;margin:0 auto;}
  .aptwrap-l{position:absolute;left:2%;top:15.5%;width:31%;height:82%;overflow:auto;display:flex;flex-direction:column;justify-content:center;}
  .aptwrap-r{position:absolute;left:35%;top:15.5%;width:63%;height:82%;overflow:auto;display:flex;flex-direction:column;justify-content:center;}
  .apt-h{font-weight:700;font-size:calc(var(--u)*0.92);color:var(--navy);margin-bottom:.35em;line-height:1.2;}
  .apt-note{font-size:calc(var(--u)*0.8);color:var(--grey);margin-top:.35em;}
  .aptbl{border-collapse:collapse;width:100%;font-size:calc(var(--u)*0.86);font-variant-numeric:tabular-nums;}
  .aptbl th{background:var(--navy);color:#fff;padding:1px 3px;position:sticky;top:0;font-weight:600;}
  .aptbl td{padding:1px 4px;text-align:center;border:1px solid #fff;}
  .aptbl td.ct{font-weight:700;}
  .grid{stroke:#e6e6e6;stroke-width:1;} .ax{stroke:#999;stroke-width:1;} .tk{fill:#666;font-size:11px;} .al{fill:#1b2430;font-size:12px;}
  .cv{fill:none;stroke-width:1.7;} .ext{fill:none;stroke-width:1.4;stroke-dasharray:5 4;} .dt{stroke:#fff;stroke-width:.5;}
  .bound{stroke:#999;stroke-width:1;stroke-dasharray:2 3;} .guide{stroke:#888;stroke-dasharray:4 3;stroke-width:1;visibility:hidden;}
  /* navegação */
  /* navegação presa ao PALCO (canto sup. direito do slide), nunca na barra preta do viewport */
  .nav{position:absolute;top:1.6%;right:1.6%;
       display:flex;align-items:center;gap:10px;z-index:50;
       background:rgba(40,60,95,.92);color:#fff;border:1px solid rgba(255,255,255,.35);border-radius:18px;padding:4px 12px;font-size:13px;}
  .nav button{background:none;border:none;color:#fff;font-size:19px;cursor:pointer;line-height:1;padding:0 5px;}
  .nav button:hover{color:#f4a722;}
  #tip{position:fixed;pointer-events:none;background:#111;color:#fff;font-size:11.5px;padding:8px 10px;border-radius:7px;
       max-width:250px;visibility:hidden;z-index:99;line-height:1.4;box-shadow:0 2px 10px rgba(0,0,0,.35);}
  #tip .tt-h{font-weight:700;margin-bottom:5px;padding-bottom:4px;border-bottom:1px solid #555;}
  #tip .tt-r{display:flex;justify-content:space-between;gap:18px;line-height:1.55;}
  #tip .tt-r span{color:#aab2c0;}
  #tip .tt-r b{font-variant-numeric:tabular-nums;}
</style></head>
<body>
<div class="deck"><div class="stage" id="stage">
__SLIDES__
  <div class="nav"><button onclick="go(-1)" title="anterior (←)">‹</button><span id="counter"></span><button onclick="go(1)" title="próximo (→)">›</button></div>
</div></div>
<div id="tip"></div>
<script>
const DATA=__DATA__, GROUPS=__GROUPS__;
const CALIB=__CALIB__, IMP=__IMP__, FEATINFO=__FEATINFO__;
/* ---------- navegação do deck ---------- */
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

/* ---------- fábrica de gráfico de sobrevivência interativo ---------- */
const tip=document.getElementById('tip');
function makeChart(svgId, chipsId, grpId, showExt, xmax){
  const svg=document.getElementById(svgId), chips=document.getElementById(chipsId), grp=document.getElementById(grpId);
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
    const yl=el('text',{class:'al','text-anchor':'middle',transform:'translate(14,'+(M.t+PH/2)+') rotate(-90)'});yl.textContent='S(t) = P(continuar empregado)';svg.appendChild(yl);
    const xl=el('text',{class:'al',x:M.l+PW/2,y:HT-4,'text-anchor':'middle'});xl.textContent='MOB — meses desde a entrada';svg.appendChild(xl);
  }
  function lpath(arr,m0,m1){let d='M '+xPix(m0)+' '+yPix(arr[m0]);for(let m=m0+1;m<=m1;m++)d+=' L '+xPix(m)+' '+yPix(arr[m]);return d;}
  function curves(){ DATA.forEach(s=>{ if(!visible.has(s.k))return;
    if(showExt)svg.appendChild(el('path',{class:'ext',d:lpath(s.W,H,xmax),stroke:s.cor}));
    svg.appendChild(el('path',{class:'cv',d:lpath(s.S,0,H),stroke:s.cor}));
    for(let m=0;m<=H;m++)svg.appendChild(el('circle',{class:'dt',cx:xPix(m),cy:yPix(s.S[m]),r:2.4,fill:s.cor})); }); }
  function syncChips(){ chips.querySelectorAll('.chip').forEach(c=>c.classList.toggle('off',!visible.has(+c.dataset.k))); }
  function render(){ svg.innerHTML=''; domain(); axes(); curves(); svg.appendChild(guide); syncChips(); }
  // chips por categoria
  DATA.forEach(s=>{ const b=document.createElement('button'); b.className='chip'; b.dataset.k=s.k; b.textContent=s.k;
    b.style.setProperty('--c',s.cor);
    b.onclick=()=>{ if(visible.has(s.k))visible.delete(s.k); else visible.add(s.k); render(); }; chips.appendChild(b); });
  // botões de grupo + todos/nenhum
  function gbtn(label,fn,col){ const b=document.createElement('button'); b.textContent=label; if(col){b.style.borderColor=col;b.style.color=col;} b.onclick=fn; grp.appendChild(b); }
  gbtn('Todos',()=>{DATA.forEach(s=>visible.add(s.k));render();});
  gbtn('Nenhum',()=>{visible.clear();render();});
  GROUPS.forEach(g=>gbtn(g.nome,()=>{visible.clear();g.cats.forEach(k=>visible.add(k));render();},g.cor));
  // tooltip
  svg.addEventListener('mousemove',ev=>{ const r=svg.getBoundingClientRect(); const sx=(ev.clientX-r.left)*(W/r.width);
    let m=Math.round((sx-M.l)/PW*xmax); m=Math.max(0,Math.min(xmax,m));
    if(sx<M.l-4||sx>W-M.r+4){tip.style.visibility='hidden';guide.style.visibility='hidden';return;}
    guide.setAttribute('x1',xPix(m));guide.setAttribute('x2',xPix(m));guide.style.visibility='visible';
    const val=s=>m<=H?s.S[m]:s.W[m];
    const vis=DATA.filter(s=>visible.has(s.k)).sort((a,b)=>val(b)-val(a));
    if(!vis.length){tip.style.visibility='hidden';return;}
    let html='<b>MOB '+m+'</b>'+(m>H?' (Weibull)':'')+'<br>';
    vis.slice(0,12).forEach(s=>{html+='<span style="color:'+s.cor+'">■</span> Cat '+s.k+': <b>'+(val(s)*100).toFixed(1)+'%</b><br>';});
    if(vis.length>12)html+='… +'+(vis.length-12)+' categorias';
    tip.innerHTML=html; tip.style.left=Math.min(ev.clientX+12,window.innerWidth-190)+'px'; tip.style.top=(ev.clientY+12)+'px'; tip.style.visibility='visible';
  });
  svg.addEventListener('mouseleave',()=>{tip.style.visibility='hidden';guide.style.visibility='hidden';});
  render();
}
makeChart('svg-km','chips-km','grp-km',false,12);
makeChart('svg-weib','chips-weib','grp-weib',true,36);

/* ---------- slide B3: gráfico-caixa (Q1/mediana/média/Q3) interativo ---------- */
function makeBoxChart(){
  const svg=document.getElementById('svg-box'); if(!svg) return;
  const NS='http://www.w3.org/2000/svg', W=600,HT=470,M={l:40,r:10,t:14,b:26},PW=W-M.l-M.r,PH=HT-M.t-M.b;
  const n=DATA.length, fmt=v=>Math.round(v);
  const l0=Math.log10(2), l1=Math.log10(3000);           // escala Y log (2..3000 meses)
  const yPix=v=>M.t+(1-(Math.log10(v)-l0)/(l1-l0))*PH;
  const xC=k=>M.l+((k-1)+0.5)/n*PW, bw=0.62*PW/n;
  function el(t,a){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;}
  // bandas de fundo por persona + rótulo
  GROUPS.forEach(g=>{ const x0=xC(g.cats[0])-bw/2-2, x1=xC(g.cats[g.cats.length-1])+bw/2+2;
    svg.appendChild(el('rect',{x:x0,y:M.t,width:x1-x0,height:PH,fill:g.cor,opacity:0.08}));
    const t=el('text',{x:(x0+x1)/2,y:M.t+9,'text-anchor':'middle','font-size':8,'font-weight':'bold',fill:g.cor}); t.textContent='Risco '+g.nome; svg.appendChild(t); });
  // grade log + linhas de referência 12/24/36
  [10,100,1000].forEach(v=>{ const y=yPix(v); svg.appendChild(el('line',{x1:M.l,y1:y,x2:W-M.r,y2:y,stroke:'#e6e6e6'}));
    const t=el('text',{x:M.l-5,y:y+3,'text-anchor':'end','font-size':9,fill:'#666'}); t.textContent=v; svg.appendChild(t); });
  [[12,'12m'],[24,'24m'],[36,'36m']].forEach(a=>{ const y=yPix(a[0]);
    svg.appendChild(el('line',{x1:M.l,y1:y,x2:W-M.r,y2:y,stroke:'#999','stroke-dasharray':'3 2','stroke-width':0.8}));
    const t=el('text',{x:M.l+2,y:y-2,'font-size':7.5,fill:'#777'}); t.textContent=a[1]; svg.appendChild(t); });
  svg.appendChild(el('line',{x1:M.l,y1:M.t,x2:M.l,y2:M.t+PH,stroke:'#999'}));
  svg.appendChild(el('line',{x1:M.l,y1:M.t+PH,x2:W-M.r,y2:M.t+PH,stroke:'#999'}));
  const yl=el('text',{'text-anchor':'middle','font-size':8.5,fill:'#1b2430',transform:'translate(11,'+(M.t+PH/2)+') rotate(-90)'}); yl.textContent='tempo até desligamento (meses, log)'; svg.appendChild(yl);
  // caixas: IQR (Q1–Q3) + mediana (linha) + média (losango)
  const boxes={};
  DATA.forEach(s=>{ const x=xC(s.k), cor=s.cor;
    const g=el('g',{class:'box','data-k':s.k});
    g.appendChild(el('rect',{x:x-bw/2,y:yPix(s.q3),width:bw,height:yPix(s.q1)-yPix(s.q3),fill:cor,'fill-opacity':0.45,stroke:cor,'stroke-width':1,class:'bx'}));
    g.appendChild(el('line',{x1:x-bw/2,y1:yPix(s.medm),x2:x+bw/2,y2:yPix(s.medm),stroke:'#222','stroke-width':1.6}));
    const my=yPix(s.media),d=3; g.appendChild(el('path',{d:'M '+x+' '+(my-d)+' L '+(x+d)+' '+my+' L '+x+' '+(my+d)+' L '+(x-d)+' '+my+' Z',fill:'#fff',stroke:'#222','stroke-width':1}));
    const t=el('text',{x:x,y:M.t+PH+9,'text-anchor':'middle','font-size':7,fill:'#666'}); t.textContent=s.k; svg.appendChild(t);
    g.appendChild(el('rect',{x:x-bw/2-1,y:M.t,width:bw+2,height:PH,fill:'transparent'}));
    g.addEventListener('mouseenter',()=>boxHov(s.k,true));
    g.addEventListener('mouseleave',()=>boxHov(s.k,false));
    boxes[s.k]=g; svg.appendChild(g);
  });
  // tabela
  const tb=document.getElementById('boxtable');
  let html='<table><thead><tr><th>cat</th><th>Q1</th><th>mediana</th><th>média</th><th>Q3</th></tr></thead><tbody>';
  DATA.forEach(s=>{ html+='<tr data-k="'+s.k+'"><td class="ct" style="background:'+s.cor+';color:'+s.txt+'">'+s.k+'</td><td>'+fmt(s.q1)+'</td><td>'+fmt(s.medm)+'</td><td>'+fmt(s.media)+'</td><td>'+fmt(s.q3)+'</td></tr>'; });
  tb.innerHTML=html+'</tbody></table>';
  tb.querySelectorAll('tr[data-k]').forEach(r=>{ const k=+r.dataset.k;
    r.addEventListener('mouseenter',()=>boxHov(k,true)); r.addEventListener('mouseleave',()=>boxHov(k,false)); });
  // realce sincronizado (caixa <-> linha) + box de dados
  function boxHov(k,on){
    const g=boxes[k]; if(g){ const bx=g.querySelector('.bx'); bx.setAttribute('stroke-width',on?2.6:1); bx.setAttribute('fill-opacity',on?0.7:0.45); }
    const row=tb.querySelector('tr[data-k="'+k+'"]'); if(row) row.classList.toggle('hl',on);
    if(on){ const s=DATA.find(d=>d.k===k), r=svg.getBoundingClientRect(), gr=GROUPS.find(x=>x.cats.includes(k));
      const row=(l,v)=>'<div class="tt-r"><span>'+l+'</span><b>'+v+'</b></div>';
      tip.innerHTML='<div class="tt-h">Categoria '+k+(gr?' · Risco '+gr.nome:'')+'</div>'+
        row('Risco em 12 meses', s.risco12+'%')+
        row('Q1 (25%)', fmt(s.q1)+' meses')+
        row('Mediana (50%)', fmt(s.medm)+' meses')+
        row('Média', fmt(s.media)+' meses')+
        row('Q3 (75%)', fmt(s.q3)+' meses')+
        row('IQR (Q1–Q3)', fmt(s.q1)+'–'+fmt(s.q3)+' meses');
      tip.style.left=Math.min(r.left+xC(k)/W*r.width+10,window.innerWidth-260)+'px';
      tip.style.top=(r.top+yPix(s.q3)/HT*r.height-8)+'px'; tip.style.visibility='visible';
    } else tip.style.visibility='hidden';
  }
}
makeBoxChart();

/* ---------- slide 5: calibração (SVG) ---------- */
function makeCalib(){
  const svg=document.getElementById('svg-calib'); if(!svg) return;
  const NS='http://www.w3.org/2000/svg', W=380,HT=340,M={l:46,r:12,t:10,b:32},PW=W-M.l-M.r,PH=HT-M.t-M.b;
  let mx=0; CALIB.forEach(d=>{ mx=Math.max(mx,d.p,d.o); }); const top=Math.ceil(mx*10)/10||0.1;
  function el(t,a){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;}
  const xP=v=>M.l+v/top*PW, yP=v=>M.t+(1-v/top)*PH;
  [0,top/2,top].forEach(v=>{ const y=yP(v),x=xP(v);
    svg.appendChild(el('line',{x1:M.l,y1:y,x2:W-M.r,y2:y,stroke:'#eee'}));
    const ty=el('text',{x:M.l-4,y:y+3,'text-anchor':'end','font-size':8.5,fill:'#666'}); ty.textContent=(v*100).toFixed(0)+'%'; svg.appendChild(ty);
    const tx=el('text',{x:x,y:M.t+PH+11,'text-anchor':'middle','font-size':8.5,fill:'#666'}); tx.textContent=(v*100).toFixed(0)+'%'; svg.appendChild(tx); });
  svg.appendChild(el('line',{x1:xP(0),y1:yP(0),x2:xP(top),y2:yP(top),stroke:'#bbb','stroke-dasharray':'4 3'}));   // diagonal ideal
  svg.appendChild(el('line',{x1:M.l,y1:M.t,x2:M.l,y2:M.t+PH,stroke:'#999'}));
  svg.appendChild(el('line',{x1:M.l,y1:M.t+PH,x2:W-M.r,y2:M.t+PH,stroke:'#999'}));
  let dd=''; CALIB.forEach((d,i)=>{ dd+=(i?' L ':'M ')+xP(d.p)+' '+yP(d.o); });
  svg.appendChild(el('path',{d:dd,fill:'none',stroke:'#2c5f9e','stroke-width':1.6}));
  CALIB.forEach(d=>svg.appendChild(el('circle',{cx:xP(d.p),cy:yP(d.o),r:3,fill:'#2c5f9e',stroke:'#fff','stroke-width':0.6})));
  const xl=el('text',{x:M.l+PW/2,y:HT-2,'text-anchor':'middle','font-size':9,fill:'#1b2430'}); xl.textContent='risco previsto'; svg.appendChild(xl);
  const yl=el('text',{'text-anchor':'middle','font-size':9,fill:'#1b2430',transform:'translate(11,'+(M.t+PH/2)+') rotate(-90)'}); yl.textContent='risco observado'; svg.appendChild(yl);
}
makeCalib();

/* ---------- slide 5: importância das variáveis (SVG, clicável) ---------- */
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
</script>
</body></html>"""

HTML = (HTML.replace("__FONTS__", FONTS)
            .replace("__SLIDES__", SLIDES)
            .replace("__DATA__", DATA)
            .replace("__GROUPS__", GROUPS_JSON)
            .replace("__CALIB__", CALIB_JSON)
            .replace("__IMP__", IMP_JSON)
            .replace("__FEATINFO__", FEATINFO_JSON))
with open(TMP, "w", encoding="utf-8") as f:
    f.write(HTML)
shutil.copy(TMP, OUT)
print(f"FIM -> {OUT} ({len(HTML)/1024/1024:.1f} MB, {NP} slides, 3 interativos: B1/B2/B3)")
