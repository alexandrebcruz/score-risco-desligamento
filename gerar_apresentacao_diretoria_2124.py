"""[DIRETORIA · ESTEIRA 2124] Versão executiva da apresentação para Diretoria e
Presidência — modelo NOVO (retreino 2021–2024, leak-free, 14 categorias, MOB ref.
2021–2024). NÃO substitui gerar_apresentacao_consignado_diretoria.py (modelo antigo).

8 slides: Capa → Racional → Base analítica → Curva de sobrevivência interativa →
Do tempo ao prazo → TABELA DE REFERÊNCIA da política → Do risco à taxa (pricing
NPV, ROI 10%) → Síntese/proposta (números computados dos CSVs, nada hardcoded).

Uso:  MPLCONFIGDIR=/tmp/mpl /tmp/consig_venv/bin/python gerar_apresentacao_diretoria_2124.py
Saída: outputs/apresentacao_consignado_diretoria_2124.html
"""
import os, json, base64, re, shutil
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#c8d0db"})
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib import colors

DUMP = "/tmp/apresentacao_dir2124_svg"; os.makedirs(DUMP, exist_ok=True)
OUT = "outputs/apresentacao_consignado_diretoria_2124.html"
TMP = "/tmp/apresentacao_consignado_diretoria_2124.html"

# ---------- métricas do modelo ----------
MET = pd.read_csv("outputs/runpod_retreino_2124/metricas_por_ano.csv")
ME = MET[MET.modelo == "ensemble"].set_index("ano")
AUC25, KS25 = ME.loc[2025, "AUC"], ME.loc[2025, "KS"]
def pct(v, nd=1): return f"{v*100:.{nd}f}%".replace(".", ",")
def _br(v, nd=1): return f"{v:.{nd}f}".replace(".", ",")

# ======================= ESTILO =======================
W, H = 13.33, 7.5
NAVY = "#14233f"; BLUE = "#2c5f9e"; INK = "#1b2430"; GREY = "#5b6675"; LIGHT = "#eef2f7"

def new_slide():
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white"); return fig

def header(fig, kicker, title, band=NAVY):
    ax = fig.add_axes([0, 0.86, 1, 0.14]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0, 0), 1, 1, color=band))
    ax.add_patch(Rectangle((0, 0), 0.012, 1, color="#f4a722"))
    ax.text(0.035, 0.66, kicker, color="#9fc0e8", fontsize=11.5, weight="bold", va="center")
    ax.text(0.035, 0.30, title, color="white", fontsize=20, weight="bold", va="center")

def footer(fig, n):
    fig.text(0.035, 0.03, "Crédito Consignado · risco de permanência no emprego · RAIS 2016–2025 · treino 2021–2024",
             fontsize=8, color=GREY)
    fig.text(0.965, 0.03, f"{n}", fontsize=9, color=GREY, ha="right")

def bullet(fig, x, y, lines, fs=13, dy=0.062, color=INK):
    for i, (b, t) in enumerate(lines):
        yy = y - i * dy
        if b is True:
            fig.text(x, yy + 0.006, "▸", fontsize=fs, color="#f4a722", va="center")
            fig.text(x + 0.022, yy, t, fontsize=fs, color=color, va="center")
        elif b is None:
            fig.text(x + 0.022, yy, t, fontsize=fs, color=color, va="center")
        else:
            fig.text(x, yy, t, fontsize=fs, color=color, va="center", weight="bold")

pages = []

# ======================= SLIDE 1 — CAPA =======================
def capa():
    fig = new_slide()
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0, 0), 1, 1, color=NAVY))
    ax.add_patch(Rectangle((0, 0.595), 1, 0.012, color="#f4a722"))
    ax.text(0.06, 0.82, "POLÍTICA DE CRÉDITO CONSIGNADO", color="#9fc0e8", fontsize=15, weight="bold")
    ax.text(0.06, 0.725, "Risco de permanência no emprego", color="white", fontsize=31, weight="bold")
    ax.text(0.06, 0.648, "como base para a definição de prazo e elegibilidade", color="white", fontsize=22, weight="bold")
    ax.text(0.06, 0.545, "Modelo sobre dados públicos (RAIS) · treinado em 2021–2024 · validado em 10 anos (743 mi de vínculos)",
            color="#c9d6e8", fontsize=13.5)
    cards = [("Faixas de risco", "14", "ordenadas em 10 safras"),
             ("Discriminação", f"AUC {pct(AUC25)}", "2025 = futuro puro"),
             ("Separação", f"KS {pct(KS25)}", "padrão de crédito"),
             ("Horizonte", "6–60 m", "prazo de referência")]
    for i, (k, v, s) in enumerate(cards):
        x = 0.06 + i * 0.225
        ax.add_patch(FancyBboxPatch((x, 0.28), 0.20, 0.18, boxstyle="round,pad=0.012", linewidth=0, facecolor="#1e3357"))
        ax.text(x + 0.10, 0.425, k, color="#9fc0e8", fontsize=11.5, ha="center", weight="bold")
        ax.text(x + 0.10, 0.365, v, color="white", fontsize=22, ha="center", weight="bold")
        ax.text(x + 0.10, 0.305, s, color="#9fc0e8", fontsize=9, ha="center")
    ax.text(0.06, 0.12, "Material para Diretoria e Presidência  ·  objetivo: tabelas de referência para a concessão — prazo, cobertura e taxa (slides finais)",
            color="#c9d6e8", fontsize=12.5)
    pages.append(fig)
capa()

# ======================= SLIDE 2 — RACIONAL =======================
def racional():
    fig = new_slide(); header(fig, "POR QUE ESTE MODELO", "O risco central do consignado é a permanência no emprego")
    bullet(fig, 0.05, 0.75, [
        (False, "O mecanismo do produto"),
        (True, "No consignado, a parcela é descontada diretamente da folha de pagamento."),
        (True, "Enquanto há vínculo formal, o pagamento é altamente seguro (baixa inadimplência)."),
        (False, "Onde mora o risco"),
        (True, "Se o tomador é desligado, o desconto em folha cessa e o saldo vira crédito comum —"),
        (None, "sem a garantia da folha e com inadimplência muito maior."),
        (True, "Logo, o prazo do contrato deveria caber na expectativa de permanência no emprego."),
        (False, "A proposta"),
        (True, "Estimar, por perfil do tomador, a probabilidade de seguir empregado mês a mês —"),
        (None, "e traduzir isso em prazo máximo e cobertura esperada de parcelas."),
    ], fs=12.8, dy=0.067)
    ax = fig.add_axes([0.66, 0.18, 0.30, 0.58]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02", facecolor=LIGHT, linewidth=0))
    ax.text(0.5, 0.90, "Em uma frase", ha="center", fontsize=13, weight="bold", color=NAVY)
    ax.text(0.5, 0.66, "O consignado é seguro\nenquanto durar o emprego.", ha="center", va="center",
            fontsize=13.5, color=INK)
    ax.add_patch(Rectangle((0.1, 0.5), 0.8, 0.005, color="#f4a722"))
    ax.text(0.5, 0.36, "Medir QUANTO TEMPO\no emprego tende a durar", ha="center", va="center",
            fontsize=12.5, color=INK, weight="bold")
    ax.text(0.5, 0.14, "define o prazo seguro\ndo contrato, por faixa de risco.", ha="center", va="center",
            fontsize=11.5, color=GREY)
    footer(fig, "2"); pages.append(fig)
racional()

# ======================= SLIDE 3 — BASE ANALÍTICA =======================
def base_analitica():
    fig = new_slide(); header(fig, "A BASE ANALÍTICA", "Um modelo validado em 10 anos, sem atalhos")
    bullet(fig, 0.05, 0.75, [
        (False, "Dados públicos e auditáveis"),
        (True, "RAIS — registro oficial de todos os vínculos formais do país."),
        (True, "743 milhões de vínculos avaliados (2016–2025)."),
        (False, "Método"),
        (True, "Machine learning (CatBoost) treinado em 2021–2024, SEM vazamento"),
        (None, "de informação do desfecho (auditoria de features)."),
        (True, "Avaliado em TODOS os anos 2016–2025; em 2025 — o futuro"),
        (None, f"nunca visto — AUC {pct(AUC25)} e KS {pct(KS25)}."),
        (False, "Qualidade comprovada"),
        (True, "Desempenho estável em 10 safras (inclusive na pandemia)"),
        (None, "e calibrado: risco previsto bate com o observado →"),
    ], fs=12.6, dy=0.060)
    # calibração 2025 desenhada NATIVA no slide (vetorial no SVG — sem PNG raster)
    cal = pd.read_csv("outputs/runpod_retreino_2124/calibracao_2025.csv")
    ax = fig.add_axes([0.60, 0.17, 0.345, 0.555])
    lim = max(cal.prevista.max(), cal.observada.max()) * 1.08
    ax.plot([0, lim], [0, lim], ls="--", lw=1.1, color="#999999", label="calibração perfeita")
    ax.plot(cal.prevista, cal.observada, marker="o", ms=5, lw=1.8, color=BLUE,
            label="ensemble 2021–24")
    for _, r in cal.iterrows():
        ax.annotate(f"{r.n/1e6:.1f}M".replace(".", ","), (r.prevista, r.observada),
                    fontsize=6.5, color="#8a93a1", xytext=(5, -9), textcoords="offset points")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Risco previsto (média do decil)", fontsize=9.5, color=INK)
    ax.set_ylabel("Risco observado (freq. real)", fontsize=9.5, color=INK)
    ax.set_title("Calibração — out-of-time 2025", fontsize=11, color=NAVY, weight="bold")
    ax.tick_params(labelsize=8, colors=GREY)
    from matplotlib.ticker import FuncFormatter
    for axis in (ax.xaxis, ax.yaxis):          # frações → porcentagem (legibilidade)
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.grid(alpha=.3, lw=.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.55, 0.085, "Calibração no out-of-time 2025: risco previsto ≈ risco observado em todas as faixas.",
             fontsize=9.5, color=GREY)
    footer(fig, "3"); pages.append(fig)
base_analitica()

# salva estáticos em SVG (+PNG p/ inspeção via DIR_DUMP_PNG)
_PNG = os.environ.get("DIR_DUMP_PNG")
if _PNG: os.makedirs(_PNG, exist_ok=True)
for i, f in enumerate(pages):
    f.savefig(f"{DUMP}/slide_{i:02d}.svg", facecolor="white")
    if _PNG: f.savefig(f"{_PNG}/slide_{i:02d}.png", dpi=96, facecolor="white")
    plt.close(f)
NSTATIC = len(pages)

# ======================= MOTOR HTML =======================
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

km = pd.read_csv("outputs/tables/sobrevivencia_km_mob_2124.csv").rename(columns={"mob": "mes"})
ext = pd.read_csv("outputs/tables/sobrevivencia_weibull_extrap_mob_2124.csv")
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
    series.append({"k": int(k), "cor": cor[k], "txt": _contraste(cor[k]), "S": S, "W": Wv,
                   "risco12": round(float(res.loc[k, "risco_deslig_12m_KM"]) * 100, 1)})
DATA = json.dumps(series, ensure_ascii=False)
GROUPS = [("Mínimo", [1], "#1a9850"), ("Baixo", [2, 3, 4], "#86cb66"),
          ("Médio-Baixo", [5, 6, 7], "#c9a227"), ("Médio", [8, 9, 10], "#fb8d3d"),
          ("Alto", [11, 12, 13, 14], "#d73027")]
GROUPS_JSON = json.dumps([{"nome": n, "cats": c, "cor": col} for n, c, col in GROUPS], ensure_ascii=False)

_TTF = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
def _face(fname, weight):
    b = base64.b64encode(open(os.path.join(_TTF, fname), "rb").read()).decode()
    return ("@font-face{font-family:'DejaVu Sans';font-style:normal;font-weight:%s;"
            "src:url(data:font/ttf;base64,%s) format('truetype');}" % (weight, b))
FONTS = _face("DejaVuSans.ttf", "400") + _face("DejaVuSans-Bold.ttf", "700")

def bullets_html(items):
    # b=True → bullet ▸ · b=None → continuação da linha anterior (indentada, sem ▸) ·
    # b=False → cabeçalho de seção (mesma semântica do bullet() dos slides estáticos)
    out = []
    for b, t in items:
        if b is True:
            out.append(f'<div class="b"><span class="bi">▸</span>{t}</div>')
        elif b is None:
            out.append(f'<div class="b bn">{t}</div>')
        else:
            out.append(f'<div class="bh">{t}</div>')
    return "\n".join(out)

SURV_TXT = bullets_html([
    (False, "O que a curva mostra"),
    (True, "S(t) = probabilidade de o tomador seguir empregado t meses após a concessão."),
    (True, "Cada curva é uma faixa de risco; selecione faixas ou grupos para comparar."),
    (False, "Como ler"),
    (True, "Cai devagar (faixas baixas) = emprego estável → suporta prazos longos."),
    (True, "Cai rápido (faixas altas) = alta rotatividade → exige prazos curtos."),
    (False, "Base"),
    (True, "Kaplan-Meier sobre 4 safras da RAIS (2021–2024); ≤12m observado (sólido),"),
    (None, ">12m extrapolado por Weibull (tracejado)."),
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

# ---------- tabelas da política (dos CSVs _2124) ----------
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
            # truncado p/ baixo (floor): prazo MÁXIMO com confiança ≥ c — nunca
            # arredondar (round superestimaria o prazo seguro). Convenção dos decks 2124.
            v = float(r[c]); disp = "120+" if v > 120 else ("<1" if v < 1 else f"{int(v)}")
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

_INTRO_A = bullets_html([
    (False, "Do tempo de emprego ao prazo do contrato"),
    (True, "S(t) = probabilidade de o desconto em folha seguir ativo no mês t."),
    (True, "Prazo por confiança: maior prazo t em que S(t) ≥ c (ex.: 90% de seguir empregado)."),
    (True, "Cobertura: Σ S(m) = parcelas esperadas pagas em folha; exija ≥ um piso de cobertura."),
    (True, "Taxa: a mesma S(t) precifica o juro por faixa — piso de quebra-zero e taxa de"),
    (None, "pricing com retorno-alvo (slide 7)."),
])
_INTRO_B = bullets_html([
    (False, "Como usar na política"),
    (True, "Faixa de risco vira tier: prazo máximo (coluna de confiança), piso de cobertura"),
    (None, "e taxa calculada por faixa — baixas → prazos longos; altas → curtos ou rejeição."),
    (True, "Refinar a taxa calculada com LGD e mitigantes (rescisão/FGTS, portabilidade)."),
    (False, "Ressalvas (governança / LGPD)"),
    (True, ">12 meses é projeção (Weibull) — recalibrar com cada nova RAIS anual."),
    (True, "Análise agregada; decisão individual exige revisão humana e cuidado com viés."),
])
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
    return ('<div class="slide cust"><div class="hb"><span class="kick">APLICAÇÃO · POLÍTICA DE CONCESSÃO</span>'
            '<span class="ttl">Do tempo de emprego ao prazo do consignado</span></div>'
            '<div class="consigbody">'
            '<div class="consigcols"><div>' + _INTRO_A + '</div><div>' + _INTRO_B + '</div></div>'
            '<div class="consigart">' + CONSIG_ART + '</div>'
            '</div></div>')
def consig_tables_slide():
    return ('<div class="slide cust"><div class="hb"><span class="kick">REFERÊNCIA PARA A POLÍTICA DE CONCESSÃO</span>'
            '<span class="ttl">Prazo máximo e cobertura de parcelas por faixa de risco (14 faixas)</span></div>'
            '<div class="aptwrap-l"><div class="apt-h">Prazo máx. (meses) por confiança de seguir empregado</div>' + TERMO_TBL +
            '<div class="apt-note">Faixas baixas suportam 60 m+; altas → prazos curtos ou rejeição.</div></div>'
            '<div class="aptwrap-r"><div class="apt-h">Cobertura esperada de parcelas (% pagas em folha) por prazo T</div>' + COV_TBL +
            '<div class="apt-note">T em meses (MOB) · &gt;12 extrapolado (Weibull) · referência 2021–2024 · verde = melhor</div></div></div>')

# ---------- slide DO RISCO À TAXA (pricing NPV) ----------
TAXA_BK = pd.read_csv("outputs/tables/consignado_taxa_breakeven_2124.csv")
TAXA_NPV = pd.read_csv("outputs/tables/consignado_taxa_npv_2124.csv")
_TS_TAX = [6, 12, 18, 24, 36, 48, 60]
# valores citados nos bullets/síntese — sempre COMPUTADOS dos CSVs (convenção 2124)
_bk1, _bk14 = (float(TAXA_BK.loc[TAXA_BK.categoria == c, "m_T24"].iloc[0]) for c in (1, 14))
_np1, _np14 = (float(TAXA_NPV.loc[TAXA_NPV.categoria == c, "m10_T24"].iloc[0]) for c in (1, 14))

def _tax_tbl():
    vmax = max(float(TAXA_NPV[f"m10_T{t}"].max()) for t in _TS_TAX)
    h = "<tr><th>cat</th>" + "".join(f"<th>T={t}</th>" for t in _TS_TAX) + "</tr>"
    body = ""
    for _, r in TAXA_NPV.iterrows():
        k = int(r.categoria)
        cells = "".join('<td style="background:%s">%s</td>'
                        % (_heat(1 - float(r[f"m10_T{t}"]) / vmax), _br(float(r[f"m10_T{t}"]), 2))
                        for t in _TS_TAX)
        body += "<tr>" + _catcell(k) + cells + "</tr>"
    return '<table class="aptbl"><thead>%s</thead><tbody>%s</tbody></table>' % (h, body)

_TAXA_TXT = bullets_html([
    (False, "Como o risco vira taxa"),
    (True, "Taxa de pricing (NPV): o juro que entrega o retorno-alvo já descontando os"),
    (None, "desligamentos esperados de cada faixa e o custo de captação."),
    (True, f"Piso (quebra-zero, T=24): de {_br(_bk1, 2)}%/mês (faixa 1) a {_br(_bk14)}%/mês (faixa 14)."),
    (False, "Como ler a tabela"),
    (True, "A taxa cresce com a faixa de risco — o crédito continua viável nas faixas altas,"),
    (None, "desde que precificado (ou mitigado por prazo curto)."),
    (True, "Nas faixas baixas a taxa CAI com o prazo: parcelas longas seguem seguras."),
    (False, "Premissas (conservadoras)"),
    (True, "Captação 1,2%/mês · retorno-alvo (ROI) 10% a valor presente."),
    (True, "ZERO recuperação após o desligamento — com mitigantes (rescisão/FGTS,"),
    (None, "portabilidade), a taxa necessária é MENOR que a mostrada."),
])
def taxa_slide():
    return ('<div class="slide cust"><div class="hb"><span class="kick">APLICAÇÃO · PRICING POR FAIXA</span>'
            '<span class="ttl">Do risco à taxa: juros de pricing por faixa de risco</span></div>'
            '<div class="txt">' + _TAXA_TXT + '</div>'
            '<div class="aptwrap-tx"><div class="apt-h">Taxa de pricing (NPV, ROI 10%) — % ao mês, por prazo T</div>' + _tax_tbl() +
            '<div class="apt-note">T em meses · referência 2021–2024 · verde = taxa menor · premissa: 0% de recuperação pós-desligamento</div></div></div>')

# ---------- slide de SÍNTESE / PROPOSTA (fecho executivo) ----------
def _pz(c, col="conf_90"):
    v = float(PRAZO.loc[PRAZO.categoria == c, col].iloc[0])
    return "120+" if v > 120 else ("<1" if v < 1 else f"{int(v)}")
def sintese_slide():
    pts = [
        f"Modelo validado em 10 safras da RAIS (743 mi de vínculos); no futuro puro (2025): AUC {pct(AUC25)} e KS {pct(KS25)}.",
        "14 faixas de risco ordenadas em todos os anos → política objetiva de prazo, cobertura e taxa.",
        f"Prazo máximo (90% de confiança de seguir empregado): de {_pz(14)} mês(es) na faixa 14 a {_pz(1)} meses na faixa 1.",
        f"Taxa mínima (quebra-zero, T=24): de {_br(_bk1, 2)}%/mês (faixa 1) a {_br(_bk14)}%/mês (faixa 14).",
        f"Pricing com ROI 10% (T=24): de {_br(_np1)}%/mês a {_br(_np14)}%/mês — com premissa de ZERO recuperação pós-desligamento.",
        "Proposta: adotar a tabela de referência na concessão (prazo por faixa + piso de cobertura + taxa por faixa).",
        "Governança: recalibrar a cada nova RAIS anual; &gt;12 meses é projeção (Weibull).",
    ]
    lis = "".join(f'<div class="fb">• {t}</div>' for t in pts)
    return ('<div class="slide fin"><div class="finbody">'
            '<div class="finttl">Síntese e proposta</div><div class="finbar"></div>'
            + lis +
            '<div class="finlgpd">⚠ Uso ético/LGPD: análise agregada e descritiva; decisões sobre indivíduos '
            'exigem revisão humana e cuidado com vieses (setor, escolaridade, região).</div>'
            '</div></div>')

# ======================= MONTAGEM =======================
slides = []
for i in range(NSTATIC):
    slides.append(f'<div class="slide">{inline_svg(f"{DUMP}/slide_{i:02d}.svg", f"s{i:02d}_")}</div>')
slides.append(interactive_slide("PERMANÊNCIA NO EMPREGO",
              "Probabilidade de seguir empregado, por faixa de risco (ref. 2021–2024)", SURV_TXT, "km"))
slides.append(consig_intro_slide())
slides.append(consig_tables_slide())
slides.append(taxa_slide())
slides.append(sintese_slide())
SLIDES = "\n".join(slides)
NTOTAL = len(slides)

HTML = r"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8">
<title>Crédito Consignado — política de concessão (Diretoria)</title>
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
  .b.bn{text-indent:0;margin-top:calc(var(--u)*-0.14);}
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
  .consigbody{position:absolute;left:4%;right:4%;top:14%;bottom:3%;display:flex;flex-direction:column;
              justify-content:center;gap:4.5%;}
  .consigcols{display:flex;gap:5%;align-items:flex-start;}
  .consigcols > div{flex:1 1 0;}
  .consigart{width:100%;}
  .consigart svg{display:block;width:88%;height:auto;margin:0 auto;}
  .aptwrap-l{position:absolute;left:2%;top:15.5%;width:31%;height:82%;overflow:auto;display:flex;flex-direction:column;justify-content:center;}
  .aptwrap-r{position:absolute;left:35%;top:15.5%;width:63%;height:82%;overflow:auto;display:flex;flex-direction:column;justify-content:center;}
  .apt-h{font-weight:700;font-size:calc(var(--u)*1.0);color:var(--navy);margin-bottom:.35em;line-height:1.2;}
  .apt-note{font-size:calc(var(--u)*0.82);color:var(--grey);margin-top:.35em;}
  .aptbl{border-collapse:collapse;width:100%;font-size:calc(var(--u)*0.95);font-variant-numeric:tabular-nums;}
  .aptbl th{background:var(--navy);color:#fff;padding:2px 4px;position:sticky;top:0;font-weight:600;}
  .aptbl td{padding:2px 5px;text-align:center;border:1px solid #fff;}
  .aptbl td.ct{font-weight:700;}
  .aptwrap-tx{position:absolute;left:43%;top:15.5%;width:55%;height:82%;overflow:auto;display:flex;flex-direction:column;justify-content:center;}
  .fin{background:var(--navy);}
  .finbody{position:absolute;left:6%;right:6%;top:12%;bottom:6%;display:flex;flex-direction:column;justify-content:center;}
  .finttl{color:#fff;font-size:calc(var(--u)*3.0);font-weight:bold;margin-bottom:calc(var(--u)*0.6);}
  .finbar{width:18%;height:calc(var(--u)*0.16);background:#f4a722;margin-bottom:calc(var(--u)*1.6);}
  .fb{color:#dbe6f3;font-size:calc(var(--u)*1.32);margin:calc(var(--u)*0.55) 0;line-height:1.35;}
  .finlgpd{color:#9fc0e8;font-size:calc(var(--u)*1.05);margin-top:calc(var(--u)*2.0);}
  .grid{stroke:#e6e6e6;stroke-width:1;} .ax{stroke:#999;stroke-width:1;} .tk{fill:#666;font-size:11px;} .al{fill:#1b2430;font-size:12px;}
  .cv{fill:none;stroke-width:1.7;} .ext{fill:none;stroke-width:1.4;stroke-dasharray:5 4;} .dt{stroke:#fff;stroke-width:.5;}
  .bound{stroke:#999;stroke-width:1;stroke-dasharray:2 3;} .guide{stroke:#888;stroke-dasharray:4 3;stroke-width:1;visibility:hidden;}
  .nav{position:absolute;top:1.6%;right:1.6%;display:flex;align-items:center;gap:10px;z-index:50;
       background:rgba(40,60,95,.92);color:#fff;border:1px solid rgba(255,255,255,.35);border-radius:18px;padding:4px 12px;font-size:13px;}
  .nav button{background:none;border:none;color:#fff;font-size:19px;cursor:pointer;line-height:1;padding:0 5px;}
  .nav button:hover{color:#f4a722;}
  #tip{position:fixed;pointer-events:none;background:#111;color:#fff;font-size:11.5px;padding:8px 10px;border-radius:7px;
       max-width:250px;visibility:hidden;z-index:99;line-height:1.4;box-shadow:0 2px 10px rgba(0,0,0,.35);}
</style></head>
<body>
<div class="deck"><div class="stage" id="stage">
__SLIDES__
  <div class="nav"><button onclick="go(-1)" title="anterior (←)">‹</button><span id="counter"></span><button onclick="go(1)" title="próximo (→)">›</button></div>
</div></div>
<div id="tip"></div>
<script>
const DATA=__DATA__, GROUPS=__GROUPS__;
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
  function axes(){ const range=yMax-yMin,dec=range<0.04?1:0,NT=5;
    for(let i=0;i<=NT;i++){const s=yMin+range*i/NT,y=yPix(s);
      svg.appendChild(el('line',{class:'grid',x1:M.l,y1:y,x2:W-M.r,y2:y}));
      const t=el('text',{class:'tk',x:M.l-6,y:y+3,'text-anchor':'end'});t.textContent=(s*100).toFixed(dec)+'%';svg.appendChild(t);}
    const step=xmax>12?3:1;
    for(let m=0;m<=xmax;m+=step){const x=xPix(m);
      svg.appendChild(el('line',{class:'grid',x1:x,y1:M.t,x2:x,y2:M.t+PH}));
      const t=el('text',{class:'tk',x:x,y:M.t+PH+15,'text-anchor':'middle'});t.textContent=m;svg.appendChild(t);}
    if(showExt){const xv=xPix(H);svg.appendChild(el('line',{class:'bound',x1:xv,y1:M.t,x2:xv,y2:M.t+PH}));}
    svg.appendChild(el('line',{class:'ax',x1:M.l,y1:M.t,x2:M.l,y2:M.t+PH}));
    svg.appendChild(el('line',{class:'ax',x1:M.l,y1:M.t+PH,x2:W-M.r,y2:M.t+PH}));
    const yl=el('text',{class:'al','text-anchor':'middle',transform:'translate(14,'+(M.t+PH/2)+') rotate(-90)'});yl.textContent='S(t) = P(seguir empregado)';svg.appendChild(yl);
    const xl=el('text',{class:'al',x:M.l+PW/2,y:HT-4,'text-anchor':'middle'});xl.textContent='meses desde a concessão (MOB)';svg.appendChild(xl);
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
makeChart('svg-km','chips-km','grp-km',true,36);
</script>
</body></html>"""

HTML = (HTML.replace("__FONTS__", FONTS).replace("__SLIDES__", SLIDES)
            .replace("__DATA__", DATA).replace("__GROUPS__", GROUPS_JSON))
with open(TMP, "w", encoding="utf-8") as f:
    f.write(HTML)
shutil.copy(TMP, OUT)
print(f"FIM -> {OUT} ({len(HTML)/1024/1024:.1f} MB, {NTOTAL} slides; 1 interativo)")
