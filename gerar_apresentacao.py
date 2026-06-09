"""Gera a apresentação em PDF (16:9) do projeto de risco de desligamento.

Cobre: (1) desenvolvimento/treino do ensemble base, (2) categorização por ganho de
informação, (3) personas por grupo de risco. Usa matplotlib PdfPages (sem deps externas).
Saída: outputs/apresentacao_risco_desligamento.pdf
"""
import os, json
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyBboxPatch

# ---------- dados ----------
MET = json.load(open("outputs/runpod_ensemble_base/metrics_ensemble.json"))
SWEEP = pd.read_csv("outputs/tables/binning_infogain_sweep.csv")
CATS = pd.read_csv("outputs/tables/binning_infogain_escolhido.csv")
PERS = pd.read_csv("outputs/tables/persona_categorias.csv")
IMP = pd.read_csv("outputs/runpod_ensemble_base/importancia_ensemble.csv").sort_values("imp_ensemble", ascending=False)

# ---------- estilo ----------
W, H = 13.33, 7.5
NAVY = "#14233f"; BLUE = "#2c5f9e"; STEEL = "#3b7dba"; INK = "#1b2430"; GREY = "#5b6675"; LIGHT = "#eef2f7"
GROUPS = [
    ("Risco Mínimo",     [1, 2],              "#1a9850"),
    ("Risco Baixo",      [3, 4, 5, 6],        "#86cb66"),
    ("Risco Médio-Baixo",[7, 8, 9, 10, 11],   "#f6c544"),
    ("Risco Médio",      [12, 13, 14, 15, 16, 17], "#fb8d3d"),
    ("Risco Alto",       [18, 19, 20, 21, 22, 23], "#d73027"),
]
def gcolor(cat):
    for _, cs, c in GROUPS:
        if cat in cs: return c
    return GREY

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#c8d0db"})

def new_slide():
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    return fig

def header(fig, kicker, title, band=NAVY):
    ax = fig.add_axes([0, 0.86, 1, 0.14]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0, 0), 1, 1, color=band))
    ax.add_patch(Rectangle((0, 0), 0.012, 1, color="#f4a722"))
    ax.text(0.035, 0.66, kicker, color="#9fc0e8", fontsize=11.5, weight="bold", va="center")
    ax.text(0.035, 0.30, title, color="white", fontsize=20, weight="bold", va="center")

def footer(fig, n):
    fig.text(0.035, 0.03, "Risco de Desligamento · RAIS 2019–2023 · holdout 2023",
             fontsize=8, color=GREY)
    fig.text(0.965, 0.03, f"{n}", fontsize=9, color=GREY, ha="right")

def bullet(fig, x, y, lines, fs=13, dy=0.062, color=INK, gap_color="#f4a722"):
    for i, (b, t) in enumerate(lines):
        yy = y - i * dy
        if b:
            fig.text(x, yy + 0.006, "▸", fontsize=fs, color=gap_color, va="center")
            fig.text(x + 0.022, yy, t, fontsize=fs, color=color, va="center")
        else:
            fig.text(x, yy, t, fontsize=fs, color=color, va="center", weight="bold")

PDF = "outputs/apresentacao_risco_desligamento.pdf"
pages = []

# ======================= 1. CAPA =======================
def capa():
    fig = new_slide()
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0, 0), 1, 1, color=NAVY))
    ax.add_patch(Rectangle((0, 0.66), 1, 0.012, color="#f4a722"))
    ax.text(0.06, 0.80, "MODELO DE RISCO DE DESLIGAMENTO", color="#9fc0e8", fontsize=15, weight="bold")
    ax.text(0.06, 0.70, "Probabilidade de dispensa sem justa causa", color="white", fontsize=30, weight="bold")
    ax.text(0.06, 0.625, "Ensemble CatBoost · dados públicos RAIS · 82,96 milhões de vínculos (2023)",
            color="#c9d6e8", fontsize=14)
    # cartões de destaque
    cards = [("AUC", "0,741", "holdout 2023"), ("LogLoss", "0,348", "bem calibrado"),
             ("Categorias", "23", "ganho de informação"), ("Personas", "5 grupos", "do servidor à obra")]
    for i, (k, v, s) in enumerate(cards):
        x = 0.06 + i * 0.225
        ax.add_patch(FancyBboxPatch((x, 0.30), 0.20, 0.18, boxstyle="round,pad=0.012",
                                    linewidth=0, facecolor="#1e3357"))
        ax.text(x + 0.10, 0.445, k, color="#9fc0e8", fontsize=12, ha="center", weight="bold")
        ax.text(x + 0.10, 0.385, v, color="white", fontsize=26, ha="center", weight="bold")
        ax.text(x + 0.10, 0.325, s, color="#9fc0e8", fontsize=9.5, ha="center")
    ax.text(0.06, 0.12, "Conteúdo:  1) Desenvolvimento e treino do modelo    "
            "2) Categorização do risco    3) Personas por grupo de risco",
            color="#c9d6e8", fontsize=12.5)
    pages.append(fig)
capa()

# ======================= 2. CONTEXTO =======================
def contexto():
    fig = new_slide(); header(fig, "VISÃO GERAL", "O problema e a abordagem")
    bullet(fig, 0.05, 0.74, [
        (False, "Objetivo"),
        (True, "Estimar a probabilidade de um vínculo formal ser encerrado por"),
        (True, "dispensa sem justa causa nos meses seguintes."),
        (False, "Dados"),
        (True, "RAIS (Relação Anual de Informações Sociais) — base pública nacional"),
        (True, "de todos os vínculos formais. Anos 2019–2023, ~280 milhões de vínculos."),
        (False, "Validação honesta (out-of-time)"),
        (True, "Treina em ≤2022 e testa em 2023, ano nunca visto — mede a capacidade"),
        (True, "real de prever o futuro, não de decorar o passado."),
    ], fs=13.2, dy=0.072)
    # caixa lateral
    ax = fig.add_axes([0.66, 0.16, 0.30, 0.62]); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.add_patch(FancyBboxPatch((0,0),1,1, boxstyle="round,pad=0.02", facecolor=LIGHT, linewidth=0))
    ax.text(0.5, 0.91, "Em números", ha="center", fontsize=14, weight="bold", color=NAVY)
    rows = [("Vínculos no teste (2023)", "82,96 mi"), ("Taxa real de dispensa", "13,3%"),
            ("Features do modelo", "22"), ("AUC (holdout)", "0,741"),
            ("Erro de calibração", "< 1 p.p."), ("Categorias de risco", "23")]
    for i, (k, v) in enumerate(rows):
        y = 0.78 - i * 0.115
        ax.text(0.07, y, k, fontsize=11, color=INK, va="center")
        ax.text(0.93, y, v, fontsize=12.5, color=BLUE, va="center", ha="right", weight="bold")
    footer(fig, "2"); pages.append(fig)
contexto()

# ======================= 3. PREPARAÇÃO DOS DADOS =======================
def preparo():
    fig = new_slide(); header(fig, "DESENVOLVIMENTO · DADOS", "Preparação: features e harmonização de códigos")
    bullet(fig, 0.05, 0.75, [
        (False, "22 features por vínculo (todas disponíveis na RAIS)"),
        (True, "Ocupação (CBO) e setor (CNAE) — em níveis hierárquicos (6→1 dígito)"),
        (True, "Tempo de vínculo, idade, tipo de contrato, faixa de remuneração,"),
        (True, "tamanho e natureza da empresa, jornada, afastamentos, UF."),
        (False, "Harmonização de códigos entre anos (pré-processamento)"),
        (True, "Alguns códigos mudaram de formato entre 2022 e 2023; sem"),
        (True, "padronizar, o modelo veria categorias 'novas' (desconhecidas)"),
        (True, "no teste de 2023. A normalização garante consistência."),
    ], fs=12.8, dy=0.067)
    # mini-tabela ilustrativa da normalização (parte do pré-processamento do modelo)
    ax = fig.add_axes([0.58, 0.22, 0.38, 0.40]); ax.axis("off")
    cell = [["faixa salarial", "02", "2", "2"],
            ["afastamento", "99", "999", "99"],
            ["CBO (militar)", "010105", "10105", "010105"]]
    col = ["campo", "≤ 2022", "em 2023", "→ normalizado"]
    tbl = ax.table(cellText=cell, colLabels=col, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.8)
    for j in range(len(col)):
        c = tbl[0, j]; c.set_facecolor(NAVY); c.set_text_props(color="white", weight="bold")
    for i in range(1, 4):
        tbl[i, 3].set_text_props(color="#1a9850", weight="bold")
    fig.text(0.58, 0.135, "Ex.: a categoria majoritária de afastamento ('99', ~84% dos\n"
             "vínculos) virava '999' em 2023 — vista como desconhecida sem o ajuste.",
             fontsize=10, color=GREY)
    footer(fig, "3"); pages.append(fig)
preparo()

# ======================= 4. DESENHO DO ENSEMBLE =======================
def ensemble():
    fig = new_slide(); header(fig, "DESENVOLVIMENTO · MODELO", "Como o ensemble base foi treinado")
    ax = fig.add_axes([0.04, 0.12, 0.55, 0.70]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    def box(x, y, w, h, txt, fc, tc="white", fs=11, weight="bold"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", facecolor=fc, linewidth=0))
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", color=tc, fontsize=fs, weight=weight)
    # modelo A
    box(0.02, 0.74, 0.46, 0.13, "Modelo A", BLUE, fs=13)
    box(0.02, 0.58, 0.46, 0.13, "treina: 2019–2020\nvalida: 2021–2022", "#dce6f2", INK, 10.5, "normal")
    # modelo B
    box(0.52, 0.74, 0.46, 0.13, "Modelo B", STEEL, fs=13)
    box(0.52, 0.58, 0.46, 0.13, "treina: 2021–2022\nvalida: 2019–2020", "#dce6f2", INK, 10.5, "normal")
    # seta para média
    ax.annotate("", (0.35, 0.40), (0.25, 0.57), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2))
    ax.annotate("", (0.65, 0.40), (0.75, 0.57), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2))
    box(0.27, 0.27, 0.46, 0.13, "Ensemble = média (A + B) / 2", "#1a9850", fs=12.5)
    ax.annotate("", (0.5, 0.14), (0.5, 0.26), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2))
    box(0.27, 0.02, 0.46, 0.11, "prob_desligamento", NAVY, fs=12)
    bullet(fig, 0.62, 0.74, [
        (False, "Por que cross-temporal?"),
        (True, "Cada modelo aprende de um par de anos"),
        (True, "e valida no outro; a média reduz variância."),
        (False, "Algoritmo"),
        (True, "CatBoost (gradient boosting) em GPU"),
        (True, "Perda e early-stopping em LogLoss"),
        (True, "Trata categóricas nativamente"),
        (False, "Resultado (holdout 2023)"),
        (True, "AUC 0,741  ·  LogLoss 0,348"),
    ], fs=12.2, dy=0.066)
    footer(fig, "4"); pages.append(fig)
ensemble()

# ======================= 5. DESEMPENHO =======================
def desempenho():
    fig = new_slide(); header(fig, "DESENVOLVIMENTO · RESULTADO", "Desempenho: discrimina bem e é bem calibrado")
    ax1 = fig.add_axes([0.04, 0.10, 0.44, 0.70]); ax1.imshow(plt.imread("outputs/figures/calibracao_ensemble_base.png")); ax1.axis("off")
    ax2 = fig.add_axes([0.52, 0.10, 0.45, 0.70]); ax2.imshow(plt.imread("outputs/figures/importancia_ensemble_base.png")); ax2.axis("off")
    fig.text(0.04, 0.045, "Calibração: risco previsto ≈ observado em todos os decis (erro < 1 p.p.).",
             fontsize=10, color=GREY)
    fig.text(0.52, 0.045, "Importância: tempo de vínculo, tipo de contrato e faixa salarial dominam (~45%).",
             fontsize=10, color=GREY)
    footer(fig, "5"); pages.append(fig)
desempenho()

# ======================= 6. DIVISOR: CATEGORIZAÇÃO =======================
def divisor(titulo, sub, n):
    fig = new_slide(); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.add_patch(Rectangle((0,0),1,1,color=NAVY))
    ax.add_patch(Rectangle((0.06,0.52),0.18,0.01,color="#f4a722"))
    # título adaptativo: encolhe quando longo, p/ não sair da área do slide
    fs = 34 if len(titulo) <= 30 else max(24, int(34 * 30 / len(titulo)))
    ax.text(0.06, 0.60, titulo, color="white", fontsize=fs, weight="bold")
    ax.text(0.06, 0.43, sub, color="#9fc0e8", fontsize=16)
    pages.append(fig)
divisor("Categorização do risco", "Do score contínuo a faixas interpretáveis — maximizando o ganho de informação", "6")

# ======================= 7. MÉTODO DE CATEGORIZAÇÃO =======================
def metodo_cat():
    fig = new_slide(); header(fig, "CATEGORIZAÇÃO · MÉTODO", "Cortes que maximizam o ganho de informação")
    bullet(fig, 0.05, 0.75, [
        (False, "Ideia"),
        (True, "Fatiar a probabilidade prevista em faixas que separem ao máximo"),
        (True, "quem é desligado de quem não é (ganho de informação sobre o alvo)."),
        (False, "Como"),
        (True, "Para cada nº de categorias K, a divisão ótima é achada por"),
        (True, "programação dinâmica (maximiza a informação mútua I(faixa; alvo))."),
        (False, "Critério de parada"),
        (True, "Aumenta-se K enquanto o risco médio das faixas continua"),
        (True, "estritamente crescente. Quando a ordem quebra, para."),
    ], fs=12.8, dy=0.066)
    # painel: IG x K
    ax = fig.add_axes([0.60, 0.16, 0.36, 0.60])
    mono = SWEEP[SWEEP.monotonico]; nmono = SWEEP[~SWEEP.monotonico]
    ax.plot(SWEEP.K_categorias, SWEEP.IG_bits, color="#9aa7b8", zorder=1)
    ax.scatter(mono.K_categorias, mono.IG_bits, color="#1a9850", s=22, zorder=3, label="ordenação mantida")
    ax.scatter(nmono.K_categorias, nmono.IG_bits, color="#d73027", s=22, zorder=3, label="ordenação quebrada")
    ax.axvline(23, ls="--", color="#1a9850")
    ax.text(23, 0.012, " K*=23", color="#1a9850", fontsize=11, weight="bold")
    ax.set_xlabel("nº de categorias (K)"); ax.set_ylabel("ganho de informação (bits)")
    ax.set_title("O ponto ótimo: K = 23", fontsize=12, weight="bold")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3)
    footer(fig, "7"); pages.append(fig)
metodo_cat()

# ======================= 8. TABELA DAS 23 CATEGORIAS =======================
def tabela_cats():
    fig = new_slide(); header(fig, "CATEGORIZAÇÃO · RESULTADO", "As 23 categorias de risco (ponto ótimo)")
    ax = fig.add_axes([0.04, 0.05, 0.50, 0.78]); ax.axis("off")
    d = CATS.copy()
    cell = []
    for _, r in d.iterrows():
        cell.append([int(r.categoria), f"{r.prob_min*100:.1f}–{r.prob_max*100:.1f}%",
                     f"{r.n/1e6:.1f}", f"{r.taxa_y*100:.1f}%", f"{r.prob_media*100:.1f}%", f"{r.lift:.2f}×"])
    col = ["cat", "faixa de prob.", "n (mi)", "taxa real", "prob. méd.", "lift"]
    tbl = ax.table(cellText=cell, colLabels=col, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.2); tbl.scale(1, 1.22)
    for j in range(len(col)):
        c = tbl[0, j]; c.set_facecolor(NAVY); c.set_text_props(color="white", weight="bold")
    for i in range(1, len(cell) + 1):
        catn = cell[i-1][0]
        tbl[i, 0].set_facecolor(gcolor(catn)); tbl[i, 0].set_text_props(color="white", weight="bold")
        if i % 2 == 0:
            for j in range(1, len(col)): tbl[i, j].set_facecolor("#f4f7fb")
    # escada (barras) à direita
    ax2 = fig.add_axes([0.60, 0.13, 0.37, 0.66])
    cols = [gcolor(c) for c in d.categoria]
    ax2.bar(d.categoria, d.taxa_y*100, color=cols)
    ax2.axhline(13.29, ls=":", color=GREY); ax2.text(1, 14.5, "média 13,3%", fontsize=8, color=GREY)
    ax2.set_xlabel("categoria (1 = menor risco · 23 = maior)"); ax2.set_ylabel("taxa real de dispensa (%)")
    ax2.set_title("Risco observado por categoria — sempre crescente", fontsize=11, weight="bold")
    ax2.grid(axis="y", alpha=.3)
    fig.text(0.60, 0.045, "Risco de 0,6% (cat 1) a 66,7% (cat 23) — lift de até 5,0×.",
             fontsize=10, color=GREY)
    footer(fig, "8"); pages.append(fig)
tabela_cats()

# ======================= 9. DIVISOR: PERSONAS =======================
divisor("Personas das categorias", "Quem é cada grupo de risco — segundo as variáveis da RAIS", "9")

# ======================= 10. MÉTODO PERSONAS + GRADIENTE =======================
def metodo_persona():
    fig = new_slide(); header(fig, "PERSONAS · MÉTODO", "Como cada categoria foi perfilada")
    bullet(fig, 0.05, 0.74, [
        (False, "Duas medidas por característica"),
        (True, "Composição interna — do que a categoria é feita (% de cada valor)."),
        (True, "Distintividade (lift) — share na categoria ÷ share na base;"),
        (True, "o que é desproporcionalmente comum ali (piso de 5%)."),
        (False, "Tradução"),
        (True, "Códigos de CBO/CNAE/tipo de vínculo traduzidos pelo dicionário da RAIS."),
        (False, "Leitura"),
        (True, "Os indicadores são lidos ao longo das 23 categorias ordenadas"),
        (True, "por risco — como o perfil muda do menor para o maior risco."),
    ], fs=13.2, dy=0.072)
    footer(fig, "10"); pages.append(fig)
metodo_persona()

def gradiente_persona():
    fig = new_slide(); header(fig, "PERSONAS · GRADIENTE", "O que muda à medida que o risco sobe (categorias 1 → 23)")
    p = PERS.sort_values("categoria"); cats = p.categoria.values; XT = [1, 5, 10, 15, 20, 23]
    def panel(pos, titulo, vals, kind, cor, xlab=False):
        ax = fig.add_axes(pos)
        if kind == "bar":
            ax.bar(cats, vals, color=[gcolor(c) for c in cats], alpha=.92)
        else:
            ax.fill_between(cats, vals, color=cor, alpha=.16)
            ax.plot(cats, vals, color=cor, lw=2.6, marker="o", ms=3.5)
        ax.set_title(titulo, fontsize=12.5, weight="bold", color=INK, pad=5, loc="left")
        ax.set_xlim(0.5, 23.5); ax.set_xticks(XT); ax.tick_params(labelsize=8.5)
        if xlab: ax.set_xlabel("categoria de risco  (1 → 23)", fontsize=9.5)
        ax.grid(axis="y", alpha=.25)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        return ax
    panel([0.075, 0.50, 0.40, 0.255], "Risco — taxa real de dispensa (%)", p["taxa_y"].values, "bar", None)
    panel([0.565, 0.50, 0.40, 0.255], "Setor público (%)  ▼ desaba", p["publico%"].values, "line", "#1a9850")
    panel([0.075, 0.185, 0.40, 0.255], "Tempo de vínculo (anos)  ▼ encurta", p["tempo_anos"].values, "line", "#7a3b9e", True)
    panel([0.565, 0.185, 0.40, 0.255], "Micro/peq. empresa ≤49 func. (%)  ▲ sobe", p["micro_peq%"].values, "line", "#e8743b", True)
    fig.text(0.075, 0.095, "Cada painel mostra UMA variável nas 23 categorias (eixo x). As cores do 1º painel = os 5 grupos de risco.",
             fontsize=10.5, color=GREY)
    fig.text(0.075, 0.06, "Risco ↑  ⇒  setor público some, tempo de casa encurta e empresa diminui: do servidor estável à "
             "construção em micro construtora.", fontsize=10.5, color=GREY)
    footer(fig, "11"); pages.append(fig)
gradiente_persona()

# ======================= 11–15. PERSONAS POR GRUPO =======================
PERSONA_TXT = {
    "Risco Mínimo": ("O servidor público concursado",
        ["Servidores estatutários da administração pública e educação; militares fortemente "
         "sobre-representados (lift 10×).", "Meia-idade (~46 anos), longuíssimo tempo de casa "
         "(~15 anos), nível superior, grandes órgãos.", "Estabilidade legal do serviço público "
         "→ risco quase nulo (0,6–1,4%)."]),
    "Risco Baixo": ("CLT consolidado na indústria e na saúde",
        ["CLT por prazo indeterminado em indústria de alimentos e atividades de saúde; "
         "empresas de porte médio/grande.", "5–6 anos de casa, ensino médio/superior, ~37 anos.",
         "Vínculo estável e maduro fora do setor público (risco 3–6,6%)."]),
    "Risco Médio-Baixo": ("CLT do comércio e dos serviços de apoio",
        ["Comércio varejista/atacadista e serviços terceirizados (limpeza, escritório, apoio "
         "a empresas).", "Empresas pequenas, muitas optantes do Simples; 2–4 anos de casa, "
         "trabalhadores mais jovens.", "Maior rotatividade típica do comércio e serviços "
         "(risco 8–14,5%)."]),
    "Risco Médio": ("Alimentação, varejo e o início da construção",
        ["Bares, restaurantes e varejo; a partir daqui entram a produção industrial e a "
         "construção civil.", "Vínculos curtíssimos (~1,5–2 anos), micro/pequenas empresas, "
         "escolaridade começando a cair.", "Setores de alta rotatividade estrutural "
         "(risco 16–29%)."]),
    "Risco Alto": ("Operário da construção civil em micro construtora",
        ["Construção de edifícios e obras de infraestrutura — fortíssima sobre-representação "
         "(CNAE 41/42, lift até 7,6×).", "Micro/pequenas construtoras, ensino fundamental, "
         "remuneração baixa; majoritariamente CLT 'indeterminado'.", "O modelo capta a "
         "rotatividade ESTRUTURAL do setor: na cat 23, 2 em cada 3 são desligados no ano (66,7%)."]),
}

def grupo_slide(nome, cats, cor, n, pers=None, persona_txt=None, inds_spec=None, base_lbl="da base"):
    if pers is None: pers = PERS
    if persona_txt is None: persona_txt = PERSONA_TXT
    if inds_spec is None:
        inds_spec = [("CLT indet.", "clt_indet%"), ("Estatutário", "estatut%"),
                     ("Setor público", "publico%"), ("Simples", "simples%"),
                     ("Superior", "superior%"), ("Micro/peq.", "micro_peq%")]
    fig = new_slide(); header(fig, "PERSONAS · " + nome.upper(), persona_txt[nome][0], band=cor)
    g = pers[pers.categoria.isin(cats)]
    wn = g["n"].values
    def wavg(c): return float(np.average(g[c], weights=wn))
    taxa = wavg("taxa_y"); ntot = g["n"].sum(); pct = 100 * ntot / pers["n"].sum()
    # faixa textual
    fig.text(0.05, 0.79, f"Categorias {cats[0]}–{cats[-1]}  ·  risco {g.taxa_y.min():.1f}%–{g.taxa_y.max():.1f}%  ·  "
             f"{ntot/1e6:.1f} mi de vínculos ({pct:.0f}% {base_lbl})", fontsize=12.5, color=cor, weight="bold")
    # texto persona
    for i, t in enumerate(persona_txt[nome][1]):
        fig.text(0.05, 0.70 - i*0.066, "▸ " + t, fontsize=12.4, color=INK,
                 wrap=True)
    # indicadores como NOTAS: "{%} {característica} (lift {x}× vs. média geral)"
    wn_all = pers["n"].values
    def wavg_all(c): return float(np.average(pers[c], weights=wn_all))   # base = todas as categorias
    notas = []
    for lab, coln in inds_spec:
        v = wavg(coln); base = wavg_all(coln)
        notas.append((lab, v, (v / base if base > 0 else 0.0)))
    notas.sort(key=lambda t: t[2], reverse=True)                        # mais distintivo primeiro
    ax = fig.add_axes([0.05, 0.05, 0.47, 0.42]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.0, 0.95, "Características do grupo  (% no grupo · lift vs. média geral)",
            fontsize=11, weight="bold", color=INK)
    for i, (lab, v, lift) in enumerate(notas):
        y = 0.80 - i * 0.105                                   # linhas mais próximas
        up = lift >= 1.0
        lc = "#1a9850" if up else "#d73027"                    # verde acima de 1, vermelho abaixo
        arw = "▲" if up else "▼"
        ax.text(0.02, y, f"{v:.0f}%", fontsize=15, weight="bold", color=cor, va="center")
        ax.text(0.18, y, lab, fontsize=12, color=INK, va="center")
        ax.text(0.99, y, f"{arw} lift {lift:.1f}×", fontsize=11.5, weight="bold", color=lc, va="center", ha="right")
    # mini-cards numéricos
    ax2 = fig.add_axes([0.54, 0.10, 0.43, 0.34]); ax2.axis("off"); ax2.set_xlim(0,1); ax2.set_ylim(0,1)
    cards = [("Risco médio do grupo", f"{taxa:.1f}%"), ("Idade média", f"{wavg('idade_media'):.0f} anos"),
             ("Tempo de vínculo", f"{wavg('tempo_anos'):.1f} anos"), ("Ensino fundamental", f"{wavg('ate_fund%'):.0f}%")]
    for i, (k, v) in enumerate(cards):
        x = 0.0 + (i % 2) * 0.52; y = 0.55 - (i // 2) * 0.5
        ax2.add_patch(FancyBboxPatch((x, y), 0.46, 0.40, boxstyle="round,pad=0.015", facecolor=LIGHT, linewidth=0))
        ax2.text(x+0.23, y+0.27, v, ha="center", fontsize=20, weight="bold", color=cor)
        ax2.text(x+0.23, y+0.09, k, ha="center", fontsize=10, color=INK)
    footer(fig, n); pages.append(fig)

for i, (nome, cats, cor) in enumerate(GROUPS):
    grupo_slide(nome, cats, cor, str(12 + i))

# ======================= 16. FECHAMENTO =======================
def fecho():
    fig = new_slide(); ax = fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.add_patch(Rectangle((0,0),1,1,color=NAVY))
    ax.add_patch(Rectangle((0.06,0.70),0.18,0.01,color="#f4a722"))
    ax.text(0.06, 0.78, "Síntese", color="white", fontsize=30, weight="bold")
    pts = ["Ensemble CatBoost cross-temporal prevê dispensa sem justa causa com AUC 0,741 e boa calibração.",
           "A harmonização de códigos entre anos garante a consistência do modelo no teste de 2023.",
           "O score foi fatiado em 23 categorias ótimas (máximo ganho de informação mantendo a ordenação).",
           "As personas vão do servidor público estável (risco ~1%) ao operário da construção em micro",
           "    construtora (risco ~67%) — o modelo capta rotatividade estrutural, não só contrato temporário."]
    for i, t in enumerate(pts):
        ax.text(0.06, 0.58 - i*0.085, ("• " if not t.startswith("    ") else "") + t,
                color="#dbe6f3", fontsize=14)
    ax.text(0.06, 0.08, "⚠ Uso ético/LGPD: análise agregada e descritiva; decisões sobre indivíduos exigem "
            "revisão humana e cuidado com vieses (setor, escolaridade, região).", color="#9fc0e8", fontsize=11)
    pages.append(fig)
fecho()

# ======================= APÊNDICE: consignado privado (sem setor público) =======================
PERS_PRIV = pd.read_csv("outputs/tables/persona_categorias_privado.csv")
PERSONA_TXT_PRIV = {
    "Risco Mínimo": ("Veteranos de banco e de grandes empresas estáveis",
        ["Setor financeiro (bancos — CNAE 64, lift 11×) e serviços essenciais/utilities (saneamento, correio); "
         "grandes empregadores privados.",
         "Longuíssimo tempo de casa (11–14 anos), mais velhos (42–50); muitos em afastamento de longa duração "
         "(estabilidade de fato).",
         "Sem o setor público, este é o PISO de risco do crédito privado (0,07–0,18%)."]),
    "Risco Baixo": ("CLT qualificado e consolidado (financeiro, saúde, indústria)",
        ["CLT por prazo indeterminado em setor financeiro, saúde, indústria de alimentos e educação privada; "
         "5–6 anos de casa, ensino médio/superior.",
         "No início do grupo (cat 3) entram temporários de agência e profissionais qualificados — é aqui que cai "
         "o bancário típico.",
         "Risco de 2–6%, bem abaixo da média."]),
    "Risco Médio-Baixo": ("CLT do comércio e dos serviços de apoio",
        ["Comércio (varejo/atacado) e serviços terceirizados (limpeza, apoio a empresas); empresas pequenas, "
         "muitas no Simples; 2–4 anos de casa.",
         "Trabalhadores mais jovens; rotatividade típica do comércio e dos serviços (8–14%)."]),
    "Risco Médio": ("Alimentação, varejo e início da construção",
        ["Bares/restaurantes e varejo; a partir daqui entram a produção industrial e a construção civil; "
         "vínculos curtos (~1,5–2 anos).",
         "Micro/pequenas empresas; escolaridade começando a cair (risco 16–30%)."]),
    "Risco Alto": ("Operário da construção civil em micro construtora",
        ["Construção de edifícios e obras de infraestrutura (CNAE 41/42, lift 4–6×); micro/pequenas construtoras; "
         "ensino fundamental.",
         "Quase 100% CLT 'indeterminado' — rotatividade ESTRUTURAL do setor: até 67% desligados no ano (cat 23)."]),
}
INDS_PRIV = [("CLT indet.", "clt_indet%"), ("Nível superior", "superior%"), ("Salário > 5 SM", "rem_alta%"),
             ("Optante Simples", "simples%"), ("Micro/peq. empresa", "micro_peq%"), ("Ensino fundamental", "ate_fund%")]

def apx_contexto():
    fig = new_slide(); header(fig, "APÊNDICE · CONTEXTO", "Por que remover o setor público")
    ntot = PERS["n"].sum(); npriv = PERS_PRIV["n"].sum(); rem = ntot - npriv
    bullet(fig, 0.05, 0.74, [
        (False, "Uso pretendido: consignado PRIVADO"),
        (True, "O servidor público tem canal próprio de consignado — não é o público-alvo aqui."),
        (True, f"Removendo o setor público saem {rem/1e6:.1f} mi de vínculos ({100*rem/ntot:.0f}% da base);"),
        (True, f"restam {npriv/1e6:.1f} mi de trabalhadores do setor privado."),
        (False, "O que muda nas personas"),
        (True, "O setor público dominava as categorias de MENOR risco (estabilidade legal)."),
        (True, "Sem ele, o piso de risco passa a ser o trabalhador privado mais estável —"),
        (True, "veteranos de banco/grandes empresas — não mais o servidor concursado."),
        (True, "Risco médio e alto (comércio, serviços, construção) quase não mudam."),
    ], fs=12.6, dy=0.066)
    ax = fig.add_axes([0.66, 0.14, 0.30, 0.58]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02", facecolor=LIGHT, linewidth=0))
    ax.text(0.5, 0.91, "Grupo Risco Mínimo (cat 1–2)", ha="center", fontsize=12, weight="bold", color=NAVY)
    full_min = PERS[PERS.categoria.isin([1, 2])]["n"].sum()
    priv_min = PERS_PRIV[PERS_PRIV.categoria.isin([1, 2])]["n"].sum()
    rows = [("Base completa", f"{full_min/1e6:.1f} mi", "≈87% servidores públicos"),
            ("Só setor privado", f"{priv_min/1e6:.1f} mi", "bancos/utilities · 14 anos de casa")]
    for i, (k, v, s) in enumerate(rows):
        y = 0.62 - i * 0.36
        ax.text(0.07, y, k, fontsize=11.5, weight="bold", color=INK)
        ax.text(0.07, y - 0.10, v, fontsize=19, weight="bold", color=GROUPS[0][2])
        ax.text(0.07, y - 0.18, s, fontsize=9.3, color=GREY)
    footer(fig, "A1"); pages.append(fig)

divisor("Apêndice — Personas para consignado privado",
        "Excluindo o setor público (servidores não são o público-alvo do consignado privado)", "A")
apx_contexto()
for i, (nome, cats, cor) in enumerate(GROUPS):
    grupo_slide(nome, cats, cor, f"A{i + 2}", pers=PERS_PRIV, persona_txt=PERSONA_TXT_PRIV,
                inds_spec=INDS_PRIV, base_lbl="do privado")

# ======================= APÊNDICE — TEMPO ATÉ O DESLIGAMENTO =======================
divisor("Apêndice — Quando ocorre o desligamento",
        "Curvas de sobrevivência e tempo até a dispensa, por categoria de risco", "B")

# B1 — teoria da curva de sobrevivência + gráfico KM
def surv_curva():
    fig = new_slide(); header(fig, "TEMPO ATÉ O DESLIGAMENTO · SOBREVIVÊNCIA",
                              "Curvas de sobrevivência por categoria (Kaplan-Meier)")
    bullet(fig, 0.05, 0.76, [
        (False, "A ideia"),
        (True, "O modelo prevê QUEM/SE é desligado; a sobrevivência mede QUANDO."),
        (True, "S(t) = probabilidade de continuar empregado após t meses."),
        (False, "Dos microdados (RAIS)"),
        (True, "Evento = dispensa s/ justa causa;  tempo = mês do desligamento."),
        (True, "Censura: quem fica ativo (ou sai por outro motivo) não é 'evento'."),
        (False, "Kaplan–Meier"),
        (True, "S(t) = Π (nₘ−dₘ)/nₘ — usa a censura sem viés, mês a mês."),
        (True, "RMST(12) = área sob S(t) = meses esperados de emprego no ano."),
    ], fs=12.6, dy=0.063)
    ax = fig.add_axes([0.50, 0.10, 0.48, 0.70])
    ax.imshow(plt.imread("outputs/figures/sobrevivencia_categorias_2023.png")); ax.axis("off")
    footer(fig, "B1"); pages.append(fig)
surv_curva()

# B2 — teoria da extrapolação Weibull + gráfico extrapolado
def surv_weibull():
    fig = new_slide(); header(fig, "TEMPO ATÉ O DESLIGAMENTO · EXTRAPOLAÇÃO",
                              "Estendendo as curvas além de 12 meses (Weibull)")
    bullet(fig, 0.05, 0.76, [
        (False, "O problema"),
        (True, "12 meses de dado não enxergam além de 12m (a curva ainda está alta)."),
        (False, "Solução: forma paramétrica de Weibull"),
        (True, "S(t) = exp(−(t/λ)ᵖ);  hazard ∝ t^(p−1)  (p>1 sobe, p<1 cai)."),
        (True, "Ajuste por regressão pura: ln(−ln S) = p·ln t + ln α (OLS, 12 pts)."),
        (True, "R² médio ≈ 0,99 — extrapola a curva até 36 meses (tracejado)."),
        (False, "Ressalva"),
        (True, "Ignora a sazonalidade de dezembro; projeção >12m é suposição."),
    ], fs=12.6, dy=0.063)
    ax = fig.add_axes([0.50, 0.10, 0.48, 0.70])
    ax.imshow(plt.imread("outputs/figures/sobrevivencia_weibull_extrap_2023.png")); ax.axis("off")
    footer(fig, "B2"); pages.append(fig)
surv_weibull()

# B3 — gráfico-caixa (Q1/mediana/média/Q3) + tabela por categoria
def surv_estatisticas():
    fig = new_slide(); header(fig, "TEMPO ATÉ O DESLIGAMENTO · ESTATÍSTICAS",
                              "Q1, mediana, média e Q3 por categoria (meses)")
    ax = fig.add_axes([0.025, 0.07, 0.58, 0.74])
    ax.imshow(plt.imread("outputs/figures/estatisticas_tempo_categorias_2023.png")); ax.axis("off")
    # tabela à direita (estatísticas monotonizadas por isotônico)
    STAT = pd.read_csv("outputs/tables/sobrevivencia_weibull_estatisticas_mono_2023.csv").sort_values("categoria")
    ax2 = fig.add_axes([0.625, 0.04, 0.355, 0.78]); ax2.axis("off")
    cell = [[int(r.categoria), f"{r.q1_meses_mono:.0f}", f"{r.mediana_meses_mono:.0f}",
             f"{r.media_meses_mono:.0f}", f"{r.q3_meses_mono:.0f}"] for _, r in STAT.iterrows()]
    col = ["cat", "Q1", "med", "méd", "Q3"]
    tbl = ax2.table(cellText=cell, colLabels=col, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(6.6); tbl.scale(1, 1.04)
    for j in range(len(col)):
        c = tbl[0, j]; c.set_facecolor(NAVY); c.set_text_props(color="white", weight="bold")
    for i in range(1, len(cell) + 1):
        tbl[i, 0].set_facecolor(gcolor(cell[i-1][0])); tbl[i, 0].set_text_props(color="white", weight="bold")
        if i % 2 == 0:
            for j in range(1, len(col)): tbl[i, j].set_facecolor("#f4f7fb")
    fig.text(0.625, 0.022, "Caixa = IQR · linha = mediana · losango = média (isotônico).",
             fontsize=7.5, color=GREY)
    footer(fig, "B3"); pages.append(fig)
surv_estatisticas()

# ======================= salvar =======================
# Se DECK_DUMP_PNG estiver setado, também exporta cada slide como PNG nesse diretório
# (usado por gerar_apresentacao_html.py para montar a versão HTML do deck).
_DUMP = os.environ.get("DECK_DUMP_PNG")
_FMT = os.environ.get("DECK_DUMP_FMT", "png")     # png | svg (usado pelo deck HTML)
if _DUMP:
    os.makedirs(_DUMP, exist_ok=True)
with PdfPages(PDF) as pdf:
    for i, f in enumerate(pages):
        pdf.savefig(f, facecolor="white")
        if _DUMP:
            f.savefig(f"{_DUMP}/slide_{i:02d}.{_FMT}", dpi=100, facecolor="white")
        plt.close(f)
print(f"OK: {PDF} ({len(pages)} slides)")
