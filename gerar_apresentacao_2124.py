"""Apresentação em PDF (16:9) do MODELO NOVO — esteira 2124 (retreino 2021–2024,
interim leak-free, 14 categorias, sobrevivência MOB ref. 2021–2024).

NÃO substitui gerar_apresentacao.py (deck do modelo antigo) — é um deck novo.
Estrutura: (1) desenvolvimento/treino, (2) categorização (critério ano-a-ano),
(3) personas, (A) apêndice consignado privado, (B) sobrevivência, (C) tabelas
de política de consignado. matplotlib PdfPages (sem deps externas).

Saída: outputs/apresentacao_risco_2124.pdf
Suporta DECK_DUMP_PNG/DECK_DUMP_FMT p/ o deck HTML (gerar_apresentacao_html_2124.py).
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
ART = "outputs/runpod_retreino_2124"
MET = pd.read_csv(f"{ART}/metricas_por_ano.csv")
ME = MET[MET.modelo == "ensemble"].set_index("ano")
SWEEP = pd.read_csv("outputs/tables/binning_infogain_sweep_2124.csv")
CATS = pd.read_csv("outputs/tables/binning_infogain_escolhido_2124.csv")
PERS = pd.read_csv("outputs/tables/persona_categorias_2124.csv")
PERS_PRIV = pd.read_csv("outputs/tables/persona_categorias_2124_privado.csv")
STAT = pd.read_csv("outputs/tables/sobrevivencia_weibull_estatisticas_mono_mob_2124.csv")
PRAZO = pd.read_csv("outputs/tables/consignado_prazo_max_2124.csv")
COB = pd.read_csv("outputs/tables/consignado_cobertura_parcelas_2124.csv")
TAXA = pd.read_csv("outputs/tables/consignado_taxa_breakeven_2124.csv")

AUC25 = ME.loc[2025, "AUC"]; KS25 = ME.loc[2025, "KS"]
def pct(v, nd=1):
    """0.776 -> '77,6%' (AUC/KS exibidos em porcentagem, padrão pt-BR)."""
    return f"{v*100:.{nd}f}%".replace(".", ",")

# ---------- estilo ----------
W, H = 13.33, 7.5
NAVY = "#14233f"; BLUE = "#2c5f9e"; STEEL = "#3b7dba"; INK = "#1b2430"; GREY = "#5b6675"; LIGHT = "#eef2f7"
GROUPS = [
    ("Risco Mínimo",      [1],              "#1a9850"),
    ("Risco Baixo",       [2, 3, 4],        "#86cb66"),
    ("Risco Médio-Baixo", [5, 6, 7],        "#f6c544"),
    ("Risco Médio",       [8, 9, 10],       "#fb8d3d"),
    ("Risco Alto",        [11, 12, 13, 14], "#d73027"),
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
    fig.text(0.035, 0.03, "Risco de Desligamento · RAIS 2016–2025 · treino 2021–2024",
             fontsize=8, color=GREY)
    fig.text(0.965, 0.03, f"{n}", fontsize=9, color=GREY, ha="right")

def bullet(fig, x, y, lines, fs=13, dy=0.062, color=INK, gap_color="#f4a722"):
    for i, (b, t) in enumerate(lines):
        yy = y - i * dy
        if b is True:
            fig.text(x, yy + 0.006, "▸", fontsize=fs, color=gap_color, va="center")
            fig.text(x + 0.022, yy, t, fontsize=fs, color=color, va="center")
        elif b is None:                      # continuação do bullet anterior (sem ▸)
            fig.text(x + 0.022, yy, t, fontsize=fs, color=color, va="center")
        else:
            fig.text(x, yy, t, fontsize=fs, color=color, va="center", weight="bold")

PDF = "outputs/apresentacao_risco_2124.pdf"
pages = []
IDX = {}   # índices (0-based) de slides especiais p/ o deck HTML (robusto a reordenação)

# ======================= 1. CAPA =======================
def capa():
    fig = new_slide()
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0, 0), 1, 1, color=NAVY))
    ax.add_patch(Rectangle((0, 0.64), 1, 0.012, color="#f4a722"))
    ax.text(0.06, 0.80, "MODELO DE RISCO DE DESLIGAMENTO", color="#9fc0e8",
            fontsize=15, weight="bold")
    ax.text(0.06, 0.70, "Probabilidade de dispensa sem justa causa", color="white", fontsize=30, weight="bold")
    ax.text(0.06, 0.615, "Ensemble CatBoost treinado em 2021–2024 · "
            "validado em 10 anos de RAIS (2016–2025)", color="#c9d6e8", fontsize=13.5)
    cards = [("AUC (2025, futuro)", pct(AUC25), "out-of-time puro"),
             ("KS (2025)", pct(KS25), "separação forte"),
             ("Categorias", "14", "ordenadas em 10 safras"),
             ("Base", "743 mi", "vínculos 2016–2025")]
    for i, (k, v, s) in enumerate(cards):
        x = 0.06 + i * 0.225
        ax.add_patch(FancyBboxPatch((x, 0.30), 0.20, 0.18, boxstyle="round,pad=0.012",
                                    linewidth=0, facecolor="#1e3357"))
        ax.text(x + 0.10, 0.445, k, color="#9fc0e8", fontsize=11, ha="center", weight="bold")
        ax.text(x + 0.10, 0.385, v, color="white", fontsize=24, ha="center", weight="bold")
        ax.text(x + 0.10, 0.325, s, color="#9fc0e8", fontsize=9.5, ha="center")
    ax.text(0.06, 0.12, "Conteúdo:  1) Desenvolvimento e treino    2) Categorização do risco    "
            "3) Personas    A) Consignado privado    B) Sobrevivência    C) Política de prazos",
            color="#c9d6e8", fontsize=12)
    pages.append(fig)
capa()

# ======================= 2. CONTEXTO =======================
def contexto():
    fig = new_slide(); header(fig, "VISÃO GERAL", "O problema e a abordagem")
    bullet(fig, 0.05, 0.76, [
        (False, "Objetivo"),
        (True, "Estimar a probabilidade de um vínculo formal ser encerrado por"),
        (None, "dispensa sem justa causa nos meses seguintes."),
        (False, "Dados"),
        (True, "RAIS — registro oficial de todos os vínculos formais do país."),
        (True, "10 anos de microdados (2016–2025), 743 milhões de vínculos."),
        (False, "Validação honesta (out-of-time)"),
        (True, "Treina em 2021–2024 e avalia em TODOS os anos 2016–2025 —"),
        (None, "2025 é futuro puro: mede a capacidade real de prever, não de decorar."),
        (True, "Só usa o que se sabe NA ENTRADA do vínculo (desenho prospectivo)."),
    ], fs=12.8, dy=0.066)
    ax = fig.add_axes([0.66, 0.16, 0.30, 0.60]); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.add_patch(FancyBboxPatch((0,0),1,1, boxstyle="round,pad=0.02", facecolor=LIGHT, linewidth=0))
    ax.text(0.5, 0.92, "Em números", ha="center", fontsize=14, weight="bold", color=NAVY)
    rows = [("Vínculos avaliados", "743 mi"), ("Treino (2021–24)", "319,7 mi"),
            ("Features do modelo", "21"), ("AUC em 2025 (futuro)", pct(AUC25)),
            ("KS em 2025", pct(KS25)), ("Categorias de risco", "14")]
    for i, (k, v) in enumerate(rows):
        y = 0.80 - i * 0.118
        ax.text(0.07, y, k, fontsize=11, color=INK, va="center")
        ax.text(0.93, y, v, fontsize=12.5, color=BLUE, va="center", ha="right", weight="bold")
    footer(fig, "2"); pages.append(fig)
contexto()

# ======================= 3. DADOS / ANTI-VAZAMENTO =======================
def preparo():
    fig = new_slide(); header(fig, "DESENVOLVIMENTO · DADOS", "Features: medir apenas o que se sabe na entrada")
    bullet(fig, 0.05, 0.76, [
        (False, "21 features por vínculo (todas da RAIS pública)"),
        (True, "Ocupação (CBO) e setor (CNAE) em níveis hierárquicos; UF; idade;"),
        (None, "tipo de contrato; natureza jurídica/setor; Simples; intermitente;"),
        (None, "escolaridade, porte, faixas de remuneração/horas (ordinais numéricas)."),
        (False, "Desenho prospectivo (anti-vazamento, auditado empiricamente)"),
        (True, "Tempo de vínculo = antiguidade NA ENTRADA da janela de observação."),
        (True, "Afastamento = dias POR MÊS de exposição (taxa, não acumulado)."),
        (True, "Nada do desfecho (mês/motivo do desligamento) entra como feature —"),
        (None, "o desfecho é usado apenas como ALVO do aprendizado."),
    ], fs=12.6, dy=0.063)
    ax = fig.add_axes([0.56, 0.14, 0.42, 0.62]); ax.axis("off")
    cell = [["tempo_vinculo", "antiguidade na entrada"],
            ["dias_afastamento", "taxa por mês de exposição"],
            ["escolaridade/faixas", "numéricas ordinais (1..11 etc.)"],
            ["mês/motivo do desligamento", "APENAS alvo — nunca feature"]]
    col = ["variável", "como entra no modelo"]
    tbl = ax.table(cellText=cell, colLabels=col, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1, 1.9)
    for j in range(len(col)):
        c = tbl[0, j]; c.set_facecolor(NAVY); c.set_text_props(color="white", weight="bold")
    for i in range(1, 5):
        tbl[i, 1].set_text_props(color="#1a9850", weight="bold")
    fig.text(0.56, 0.105, "Tudo que alimenta o modelo é conhecido no início do vínculo — o risco estimado\n"
             "é prospectivo e se sustenta em produção.", fontsize=10, color=GREY)
    footer(fig, "3"); pages.append(fig)
preparo()

# ======================= 4. ENSEMBLE =======================
def ensemble():
    fig = new_slide(); header(fig, "DESENVOLVIMENTO · MODELO", "Ensemble cross-temporal dentro de 2021–2024")
    ax = fig.add_axes([0.04, 0.12, 0.55, 0.70]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    def box(x, y, w, h, txt, fc, tc="white", fs=11, weight="bold"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", facecolor=fc, linewidth=0))
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", color=tc, fontsize=fs, weight=weight)
    box(0.02, 0.74, 0.46, 0.13, "Modelo A", BLUE, fs=13)
    box(0.02, 0.58, 0.46, 0.13, "treina: 2021–2022\nvalida: 2023–2024", "#dce6f2", INK, 10.5, "normal")
    box(0.52, 0.74, 0.46, 0.13, "Modelo B", STEEL, fs=13)
    box(0.52, 0.58, 0.46, 0.13, "treina: 2023–2024\nvalida: 2021–2022", "#dce6f2", INK, 10.5, "normal")
    ax.annotate("", (0.35, 0.40), (0.25, 0.57), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2))
    ax.annotate("", (0.65, 0.40), (0.75, 0.57), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2))
    box(0.27, 0.27, 0.46, 0.13, "Ensemble = média (A + B) / 2", "#1a9850", fs=12.5)
    ax.annotate("", (0.5, 0.14), (0.5, 0.26), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2))
    box(0.27, 0.02, 0.46, 0.11, "prob_desligamento", NAVY, fs=12)
    bullet(fig, 0.62, 0.74, [
        (False, "Por que cross-temporal?"),
        (True, "Cada modelo aprende num par de anos"),
        (None, "e valida no outro; a média reduz variância."),
        (False, "Avaliação em 10 anos"),
        (True, "2016–2020 e 2025: totalmente fora do treino."),
        (True, "2025 = futuro puro (out-of-time)."),
        (False, "Resultado"),
        (True, f"AUC {pct(AUC25)} · KS {pct(KS25)} em 2025"),
        (True, "AUC 76–81% estável nos 10 anos"),
    ], fs=12.2, dy=0.066)
    footer(fig, "4"); pages.append(fig)
ensemble()

# ======================= 5. DESEMPENHO =======================
def desempenho():
    """Tudo desenhado NATIVAMENTE (vetorial — vira SVG puro no deck HTML):
    AUC e KS em gráficos de LINHA separados (AUC y∈[0,5;1]; KS y∈[0;1]) + calibração."""
    fig = new_slide(); header(fig, "DESENVOLVIMENTO · RESULTADO", "Estável em 10 safras, calibrado no futuro")
    e = ME.reset_index().sort_values("ano")
    anos = e["ano"].values
    treino = e["papel"].ne("out_of_sample").values        # anos usados em fit/val (2021–24)

    def linha_metrica(pos, valores, nome, cor, ylo, yhi, xlab=False):
        ax = fig.add_axes(pos)
        # faixa sombreada = anos do treino (fit/val); fora dela = out-of-sample
        a0, a1 = anos[treino].min() - 0.5, anos[treino].max() + 0.5
        ax.axvspan(a0, a1, color="#dce6f2", alpha=0.6, zorder=0)
        ax.plot(anos, valores, color=cor, lw=2.4, marker="o", ms=5, zorder=3)
        for xx, vv in zip(anos, valores):
            ax.annotate(f"{vv*100:.1f}%".replace(".", ","), (xx, vv),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7.6, color=INK)
        ax.set_ylim(ylo, yhi); ax.set_xlim(anos.min() - 0.4, anos.max() + 0.4)
        ax.set_xticks(anos); ax.tick_params(labelsize=8.5)
        if not xlab: ax.set_xticklabels([])
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax.set_title(f"{nome} por ano  (eixo y: {ylo*100:.0f}%–{yhi*100:.0f}%)",
                     fontsize=11.5, weight="bold", loc="left", color=INK)
        ax.grid(axis="y", alpha=.3)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.text(a0 + 0.1, ylo + (yhi - ylo) * 0.05, "treino/val (2021–24)", fontsize=7.5, color="#5b7da8")
        return ax

    linha_metrica([0.055, 0.50, 0.46, 0.295], e["AUC"].values, "AUC", BLUE, 0.5, 1.0)
    linha_metrica([0.055, 0.125, 0.46, 0.295], e["KS"].values, "KS", "#e8742c", 0.0, 1.0, xlab=True)

    # calibração 2025 desenhada nativa (decis previsto × observado)
    cal = pd.read_csv(f"{ART}/calibracao_2025.csv")
    axc = fig.add_axes([0.60, 0.125, 0.36, 0.645])
    lim = max(cal.prevista.max(), cal.observada.max()) * 1.08
    axc.plot([0, lim], [0, lim], ls="--", color="#999", lw=1.2, label="calibração perfeita")
    axc.plot(cal.prevista, cal.observada, marker="o", ms=5, color=BLUE, lw=1.8, label="ensemble 21–24")
    axc.set_xlabel("risco previsto (média do decil)", fontsize=9.5)
    axc.set_ylabel("risco observado (freq. real)", fontsize=9.5)
    axc.set_title("Calibração — out-of-time 2025", fontsize=11.5, weight="bold", loc="left", color=INK)
    axc.tick_params(labelsize=8.5); axc.legend(fontsize=8.5, loc="upper left"); axc.grid(alpha=.3)
    for s in ("top", "right"): axc.spines[s].set_visible(False)

    fig.text(0.055, 0.052, "Fora da faixa sombreada = anos NUNCA vistos no treino — incl. 2025, o futuro puro.",
             fontsize=10, color=GREY)
    fig.text(0.60, 0.052, "Risco previsto ≈ observado em todos os decis de 2025.", fontsize=10, color=GREY)
    footer(fig, "5"); pages.append(fig)
desempenho()

# ======================= 6. DIVISOR =======================
def divisor(titulo, sub, n):
    fig = new_slide(); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.add_patch(Rectangle((0,0),1,1,color=NAVY))
    ax.add_patch(Rectangle((0.06,0.52),0.18,0.01,color="#f4a722"))
    fs = 34 if len(titulo) <= 30 else max(24, int(34 * 30 / len(titulo)))
    ax.text(0.06, 0.60, titulo, color="white", fontsize=fs, weight="bold")
    ax.text(0.06, 0.43, sub, color="#9fc0e8", fontsize=16)
    pages.append(fig)
divisor("Categorização do risco", "Cortes ótimos por ganho de informação — com ordenação validada ano a ano", "6")

# ======================= 7. MÉTODO DE CATEGORIZAÇÃO =======================
def metodo_cat():
    fig = new_slide(); header(fig, "CATEGORIZAÇÃO · MÉTODO", "Ganho de informação + estabilidade entre safras")
    bullet(fig, 0.05, 0.76, [
        (False, "Cortes ótimos"),
        (True, "Para cada K, programação dinâmica acha os cortes de probabilidade"),
        (None, "que maximizam a informação mútua I(faixa; alvo) em 2021–2024."),
        (False, "Critério de K — duplo"),
        (True, "O risco médio das faixas deve crescer estritamente no AGREGADO"),
        (None, "e DENTRO de cada ano (2021, 22, 23 e 24) individualmente."),
        (True, "Com 15+ faixas, a ordenação falha em uma das 4 safras de"),
        (None, "referência (2024) → K*=14, com 99,1% do ganho máximo."),
        (False, "Validação fora da janela"),
        (True, "As 14 faixas ranqueiam o risco corretamente em TODOS os anos"),
        (None, "2016–2025 — inclusive nos 6 anos fora da referência."),
    ], fs=12.6, dy=0.060)
    ax = fig.add_axes([0.60, 0.16, 0.36, 0.60])
    mono = SWEEP[SWEEP.monotonico_por_ano & SWEEP.monotonico_pooled]
    nmono = SWEEP[~(SWEEP.monotonico_por_ano & SWEEP.monotonico_pooled)]
    ax.plot(SWEEP.K_categorias, SWEEP.IG_bits, color="#9aa7b8", zorder=1)
    ax.scatter(mono.K_categorias, mono.IG_bits, color="#1a9850", s=22, zorder=3, label="ordenado em TODO ano")
    ax.scatter(nmono.K_categorias, nmono.IG_bits, color="#d73027", s=22, zorder=3, label="quebra em algum ano")
    ax.axvline(14, ls="--", color="#1a9850")
    ax.text(14.5, SWEEP.IG_bits.min() + 0.004, "K*=14", color="#1a9850", fontsize=11, weight="bold")
    ax.set_xlabel("nº de categorias (K)"); ax.set_ylabel("ganho de informação (bits)")
    ax.set_title("K* = maior K ordenado em todas as safras", fontsize=11.5, weight="bold")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3)
    footer(fig, "7"); pages.append(fig)
metodo_cat()

# ======================= 8. TABELA DAS 14 CATEGORIAS =======================
def tabela_cats():
    fig = new_slide(); header(fig, "CATEGORIZAÇÃO · RESULTADO", "As 14 categorias de risco (ponto ótimo)")
    ax = fig.add_axes([0.03, 0.07, 0.58, 0.76]); ax.axis("off")
    d = CATS.copy()
    has_pm = "prob_media" in d.columns
    cell = []
    for _, r in d.iterrows():
        pm = f"{r.prob_media*100:.1f}%" if has_pm else "—"
        cell.append([int(r.categoria), f"{r.prob_min*100:.1f}–{r.prob_max*100:.1f}%", f"{r.n/1e6:.1f}",
                     f"{r.taxa_y*100:.1f}%", pm, f"{r.lift_vs_global:.2f}×"])
    col = ["cat", "faixa de prob.", "n (mi)", "taxa real", "prob. méd.", "lift"]
    tbl = ax.table(cellText=cell, colLabels=col, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.6); tbl.scale(1, 1.55)
    for j in range(len(col)):
        c = tbl[0, j]; c.set_facecolor(NAVY); c.set_text_props(color="white", weight="bold")
    for i in range(1, len(cell) + 1):
        catn = cell[i-1][0]
        tbl[i, 0].set_facecolor(gcolor(catn)); tbl[i, 0].set_text_props(color="white", weight="bold")
        if i % 2 == 0:
            for j in range(1, len(col)): tbl[i, j].set_facecolor("#f4f7fb")
    ax2 = fig.add_axes([0.66, 0.13, 0.31, 0.64])
    cols = [gcolor(c) for c in d.categoria]
    ax2.bar(d.categoria, d.taxa_y*100, color=cols)
    ax2.axhline(13.78, ls=":", color=GREY); ax2.text(1, 15.2, "média 13,8%", fontsize=8, color=GREY)
    ax2.set_xlabel("categoria (1 = menor risco · 14 = maior)"); ax2.set_ylabel("taxa real de dispensa (%)")
    ax2.set_title("Risco sempre crescente — em\nTODAS as safras 2016–2025", fontsize=10.5, weight="bold")
    ax2.grid(axis="y", alpha=.3)
    fig.text(0.03, 0.045, "Risco real de 0,6% (cat 1) a 71,5% (cat 14) — lift de 0,04× a 5,2×.",
             fontsize=9.5, color=GREY)
    footer(fig, "8"); pages.append(fig)
tabela_cats()
IDX["tabcat"] = len(pages) - 1

# ======================= 9. DIVISOR PERSONAS =======================
divisor("Personas das categorias", "Quem é cada faixa de risco — perfil em 2025", "9")

# ======================= 10. GRADIENTE =======================
def gradiente_persona():
    fig = new_slide(); header(fig, "PERSONAS · GRADIENTE", "O que muda à medida que o risco sobe (categorias 1 → 14)")
    p = PERS.sort_values("categoria"); cats = p.categoria.values; XT = [1, 4, 7, 10, 14]
    def panel(pos, titulo, vals, kind, cor, xlab=False):
        ax = fig.add_axes(pos)
        if kind == "bar":
            ax.bar(cats, vals, color=[gcolor(c) for c in cats], alpha=.92)
        else:
            ax.fill_between(cats, vals, color=cor, alpha=.16)
            ax.plot(cats, vals, color=cor, lw=2.6, marker="o", ms=3.5)
        ax.set_title(titulo, fontsize=12.5, weight="bold", color=INK, pad=5, loc="left")
        ax.set_xlim(0.5, 14.5); ax.set_xticks(XT); ax.tick_params(labelsize=8.5)
        if xlab: ax.set_xlabel("categoria de risco  (1 → 14)", fontsize=9.5)
        ax.grid(axis="y", alpha=.25)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    panel([0.075, 0.50, 0.40, 0.255], "Risco — taxa real de dispensa (%)", p["taxa_y"].values, "bar", None)
    panel([0.565, 0.50, 0.40, 0.255], "Setor público (%)  ▼ desaba", p["publico%"].values, "line", "#1a9850")
    panel([0.075, 0.185, 0.40, 0.255], "Tempo de casa na entrada (anos)  ▼ encurta", p["tempo_anos"].values, "line", "#7a3b9e", True)
    panel([0.565, 0.185, 0.40, 0.255], "Micro/peq. empresa ≤49 func. (%)  ▲ sobe", p["micro_peq%"].values, "line", "#e8743b", True)
    fig.text(0.075, 0.095, "Cada painel mostra UMA variável nas 14 categorias (eixo x). Cores do 1º painel = os 5 grupos de risco.",
             fontsize=10.5, color=GREY)
    fig.text(0.075, 0.06, "Risco ↑  ⇒  setor público some, tempo de casa encurta e a empresa diminui: do servidor estável "
             "à obra em micro construtora.", fontsize=10.5, color=GREY)
    footer(fig, "10"); pages.append(fig)
gradiente_persona()

# ======================= 11–15. PERSONAS POR GRUPO =======================
PERSONA_TXT = {
    "Risco Mínimo": ("O servidor público estável",
        ["Setor público domina (75%): estatutários (71%), militares e profissionais de nível "
         "superior; administração pública (CNAE 84, lift 4,6×).",
         "Quase 13 anos de casa na entrada, 45 anos de idade, 52% com nível superior e 28% "
         "ganhando acima de 5 SM; 85% em grandes organizações.",
         "Estabilidade legal e senioridade → risco quase nulo (0,6% ao ano)."]),
    "Risco Baixo": ("CLT qualificado e consolidado (saúde e serviços profissionais)",
        ["CLT por prazo indeterminado (72–77%) em saúde (CNAE 86), seleção/agenciamento de RH "
         "(78) e educação; profissionais e técnicos (CBO 2/3).",
         "3,4–4,3 anos de casa na entrada, 24–31% com superior, empresas médias e grandes.",
         "Vínculo maduro fora do setor público → risco 2,2–6,2%."]),
    "Risco Médio-Baixo": ("CLT do comércio e dos serviços de apoio",
        ["Comércio varejista (CNAE 47) e serviços de apoio a empresas (82); vendedores e "
         "operadores (CBO 5/7).",
         "1,8–2,6 anos de casa, ~37 anos; Simples 24–29% e micro/pequenas 43–50%.",
         "Rotatividade típica do comércio → risco 8,9–15,1%."]),
    "Risco Médio": ("Alimentação e varejo de alta rotação",
        ["Bares e restaurantes (CNAE 56, lift até 1,9×) e varejo; trabalhadores mais jovens "
         "(~34 anos) e vínculos curtos (1,1–1,5 ano na entrada).",
         "Remuneração baixa cresce (20→33% até 1 SM); Simples 35–42%; micro/pequenas 55–64%.",
         "Setores de giro estrutural → risco 19–28,6%."]),
    "Risco Alto": ("Operário da construção civil em micro construtora",
        ["Construção de edifícios e infraestrutura (CNAE 41/42, lift até 8×); trabalhadores "
         "da produção (CBO 7, até 47%).",
         "Micro/pequenas construtoras (63–86%), escolaridade baixa (até-fundamental 10–15%), "
         "remuneração até 1 SM em 29–49%; ~1–1,7 ano de casa.",
         "Rotatividade ESTRUTURAL do setor: na cat 14, 72% são desligados no ano."]),
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
    _rng = f"{cats[0]}–{cats[-1]}" if len(cats) > 1 else f"{cats[0]}"
    fig.text(0.05, 0.79, f"Categoria{'s' if len(cats)>1 else ''} {_rng}  ·  risco "
             f"{g.taxa_y.min():.1f}%–{g.taxa_y.max():.1f}%  ·  {ntot/1e6:.0f} mi de vínculos "
             f"({pct:.0f}% {base_lbl})", fontsize=12.5, color=cor, weight="bold")
    for i, t in enumerate(persona_txt[nome][1]):
        fig.text(0.05, 0.70 - i*0.066, "▸ " + t, fontsize=12.4, color=INK, wrap=True)
    wn_all = pers["n"].values
    def wavg_all(c): return float(np.average(pers[c], weights=wn_all))
    notas = []
    for lab, coln in inds_spec:
        v = wavg(coln); base = wavg_all(coln)
        notas.append((lab, v, (v / base if base > 0 else 0.0)))
    notas.sort(key=lambda t: t[2], reverse=True)
    ax = fig.add_axes([0.05, 0.05, 0.47, 0.42]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.0, 0.95, "Características do grupo  (% no grupo · lift vs. média geral)",
            fontsize=11, weight="bold", color=INK)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer(); inv = ax.transData.inverted()
    for i, (lab, v, lift) in enumerate(notas):
        y = 0.80 - i * 0.105
        up = lift >= 1.0
        lc = "#1a9850" if up else "#d73027"
        arw = "▲" if up else "▼"
        ax.text(0.02, y, f"{v:.0f}%", fontsize=15, weight="bold", color=cor, va="center")
        tl = ax.text(0.18, y, lab, fontsize=12, color=INK, va="center")
        xend = inv.transform((tl.get_window_extent(renderer=rend).x1, 0))[0]
        ax.text(xend + 0.025, y, f"{arw} lift {lift:.1f}×", fontsize=11.5, weight="bold", color=lc, va="center")
    ax2 = fig.add_axes([0.54, 0.10, 0.43, 0.34]); ax2.axis("off"); ax2.set_xlim(0,1); ax2.set_ylim(0,1)
    cards = [("Risco médio do grupo", f"{taxa:.1f}%"), ("Idade média", f"{wavg('idade_media'):.0f} anos"),
             ("Tempo de casa (entrada)", f"{wavg('tempo_anos'):.1f} anos"), ("Ensino fundamental", f"{wavg('ate_fund%'):.0f}%")]
    for i, (k, v) in enumerate(cards):
        x = 0.0 + (i % 2) * 0.52; y = 0.55 - (i // 2) * 0.5
        ax2.add_patch(FancyBboxPatch((x, y), 0.46, 0.40, boxstyle="round,pad=0.015", facecolor=LIGHT, linewidth=0))
        ax2.text(x+0.23, y+0.27, v, ha="center", fontsize=20, weight="bold", color=cor)
        ax2.text(x+0.23, y+0.09, k, ha="center", fontsize=10, color=INK)
    footer(fig, n); pages.append(fig)

for i, (nome, cats, cor) in enumerate(GROUPS):
    grupo_slide(nome, cats, cor, str(11 + i))

# ======================= 16. FECHO =======================
def fecho():
    fig = new_slide(); ax = fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.add_patch(Rectangle((0,0),1,1,color=NAVY))
    ax.add_patch(Rectangle((0.06,0.70),0.18,0.01,color="#f4a722"))
    ax.text(0.06, 0.78, "Síntese", color="white", fontsize=30, weight="bold")
    pts = [f"Ensemble treinado em 2021–24: AUC {pct(AUC25)} e KS {pct(KS25)} no futuro puro (2025).",
           "Desempenho estável em 10 safras (AUC 76–81%), incluindo a pandemia — generaliza p/ trás e p/ frente.",
           "14 categorias ótimas por ganho de informação, com ordenação validada DENTRO de cada ano (2016–2025).",
           "Personas (2025): do servidor público (%s%%) ao operário da construção em micro construtora (%s%%)."
           % (f"{PERS.taxa_y.min():.1f}".replace(".", ","), f"{PERS.taxa_y.max():.1f}".replace(".", ",")),
           "Sobrevivência MOB (ref. 2021–24) alimenta a política de prazos do consignado (apêndices B e C)."]
    for i, t in enumerate(pts):
        ax.text(0.06, 0.60 - i*0.082, "• " + t, color="#dbe6f3", fontsize=13.2)
    ax.text(0.06, 0.10, "⚠ Uso ético/LGPD: análise agregada e descritiva; decisões sobre indivíduos exigem "
            "revisão humana e cuidado com vieses (setor, escolaridade, região).", color="#9fc0e8", fontsize=11)
    pages.append(fig)
# (chamado no FIM da apresentação — ver abaixo)

# ======================= APÊNDICE A — CONSIGNADO PRIVADO =======================
PERSONA_TXT_PRIV = {
    "Risco Mínimo": ("Veterano do setor financeiro e da saúde",
        ["Sem o setor público, o piso de risco vira o financeiro (CNAE 64, lift 5,9×) e a "
         "saúde (86); profissionais de nível superior (CBO 2, lift 2,3×).",
         "5,8 anos de casa na entrada, 34% com superior, 21% acima de 5 SM; 57% em "
         "empresas grandes.",
         "É o piso do crédito privado: risco 0,6% ao ano."]),
    "Risco Baixo": ("CLT qualificado da saúde e dos serviços profissionais",
        ["CLT indeterminado (82–85%) em saúde (86) e agenciamento/seleção de RH (78); "
         "profissionais e técnicos (CBO 2/3).",
         "3,5–4,1 anos de casa, 20–27% com superior; porte médio/grande.",
         "Risco 2,2–6,1%, bem abaixo da média."]),
    "Risco Médio-Baixo": ("CLT do comércio e dos serviços de apoio",
        ["Comércio (47) e serviços de apoio (82); atacado (46) aparece no início do grupo.",
         "1,9–2,8 anos de casa; Simples 27–32%; micro/pequenas 48–54%.",
         "Rotatividade típica do comércio → risco 8,7–15,0%."]),
    "Risco Médio": ("Alimentação e varejo de alta rotação",
        ["Bares/restaurantes (56) e varejo; jovens (~34 anos), 1,1–1,6 ano de casa.",
         "Remuneração baixa em 19–33%; Simples 38–45%; micro/pequenas 60–67%.",
         "Risco 18,8–28,5%."]),
    "Risco Alto": ("Operário da construção civil em micro construtora",
        ["Construção (41/42/43, lift até 7×); CBO 7 (produção) até 51%.",
         "97–99,7% CLT 'indeterminado' — a saída é por dispensa, não fim de contrato; "
         "micro/pequenas 65–87%; até-fundamental 10–15%.",
         "Rotatividade estrutural: risco 35–71% ao ano."]),
}
INDS_PRIV = [("CLT indet.", "clt_indet%"), ("Nível superior", "superior%"), ("Salário > 5 SM", "rem_alta%"),
             ("Optante Simples", "simples%"), ("Micro/peq. empresa", "micro_peq%"), ("Ensino fundamental", "ate_fund%")]

def apx_contexto():
    fig = new_slide(); header(fig, "APÊNDICE A · CONTEXTO", "Por que remover o setor público")
    ntot = PERS["n"].sum(); npriv = PERS_PRIV["n"].sum(); rem = ntot - npriv
    bullet(fig, 0.05, 0.74, [
        (False, "Uso pretendido: consignado PRIVADO"),
        (True, "O servidor público tem canal próprio de consignado — não é o público-alvo aqui."),
        (True, f"Removendo o setor público saem {rem/1e6:.0f} mi de vínculos ({100*rem/ntot:.0f}% da base);"),
        (None, f"restam {npriv/1e6:.0f} mi de trabalhadores do setor privado."),
        (False, "O que muda nas personas"),
        (True, "O setor público dominava a categoria 1 (75% do grupo de menor risco)."),
        (True, "Sem ele, o piso de risco passa a ser o veterano do financeiro e da"),
        (None, "saúde — não mais o servidor concursado."),
        (True, "Risco médio e alto (comércio, alimentação, construção) quase não mudam."),
    ], fs=12.6, dy=0.066)
    ax = fig.add_axes([0.66, 0.14, 0.30, 0.58]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02", facecolor=LIGHT, linewidth=0))
    ax.text(0.5, 0.91, "Categoria 1 (risco mínimo)", ha="center", fontsize=12, weight="bold", color=NAVY)
    full_min = PERS[PERS.categoria == 1]["n"].sum()
    priv_min = PERS_PRIV[PERS_PRIV.categoria == 1]["n"].sum()
    rows = [("Base completa", f"{full_min/1e6:.0f} mi", "75% setor público · 13 anos de casa"),
            ("Só setor privado", f"{priv_min/1e6:.0f} mi", "financeiro/saúde · 5,8 anos de casa")]
    for i, (k, v, s) in enumerate(rows):
        y = 0.62 - i * 0.36
        ax.text(0.07, y, k, fontsize=11.5, weight="bold", color=INK)
        ax.text(0.07, y - 0.10, v, fontsize=19, weight="bold", color=GROUPS[0][2])
        ax.text(0.07, y - 0.18, s, fontsize=9.3, color=GREY)
    footer(fig, "A1"); pages.append(fig)

divisor("Apêndice A — Personas para consignado privado",
        "Excluindo o setor público (servidores não são o público-alvo do consignado privado)", "A")
apx_contexto()
for i, (nome, cats, cor) in enumerate(GROUPS):
    grupo_slide(nome, cats, cor, f"A{i + 2}", pers=PERS_PRIV, persona_txt=PERSONA_TXT_PRIV,
                inds_spec=INDS_PRIV, base_lbl="do privado")

# ======================= APÊNDICE B — SOBREVIVÊNCIA =======================
divisor("Apêndice B — Quando ocorre o desligamento",
        "Curvas de sobrevivência (relógio MOB) por categoria — referência 2021–2024", "B")

def surv_curva():
    fig = new_slide(); header(fig, "TEMPO ATÉ O DESLIGAMENTO · SOBREVIVÊNCIA",
                              "Curvas de sobrevivência por categoria (Kaplan-Meier, MOB)")
    bullet(fig, 0.05, 0.76, [
        (False, "A ideia"),
        (True, "O modelo prevê QUEM/SE é desligado; a sobrevivência mede QUANDO."),
        (True, "S(t) = prob. de seguir empregado t meses após a ENTRADA (relógio MOB)."),
        (False, "Dos microdados (RAIS 2021–2024 agrupados)"),
        (True, "Evento = dispensa s/ justa causa; censura = ativo ou outra saída."),
        (True, "Pré-existente entra em janeiro; admitido no ano, no mês de admissão."),
        (False, "Kaplan–Meier"),
        (True, "S(t) = Π (nₘ−dₘ)/nₘ — usa a censura sem viés, mês a mês."),
        (True, "4 safras agregadas → sazonalidade de calendário diluída."),
    ], fs=12.4, dy=0.063)
    ax = fig.add_axes([0.50, 0.10, 0.48, 0.70])
    ax.imshow(plt.imread("outputs/figures/sobrevivencia_categorias_mob_2124.png")); ax.axis("off")
    footer(fig, "B1"); pages.append(fig)
surv_curva()
IDX["B1"] = len(pages) - 1

def surv_weibull():
    fig = new_slide(); header(fig, "TEMPO ATÉ O DESLIGAMENTO · EXTRAPOLAÇÃO",
                              "Estendendo as curvas além de 12 meses (Weibull)")
    bullet(fig, 0.05, 0.76, [
        (False, "O problema"),
        (True, "12 meses de dado não enxergam além de 12m (a curva ainda está alta)."),
        (False, "Solução: forma paramétrica de Weibull"),
        (True, "S(t) = exp(−(t/λ)ᵖ);  hazard ∝ t^(p−1)."),
        (True, "Ajuste por regressão pura: ln(−ln S) = p·ln t + ln α (OLS, 12 pts)."),
        (True, "R² médio ≈ 0,994 — extrapola até 36 MOB (tracejado)."),
        (False, "Qualidade do ajuste"),
        (True, "Q1/mediana/média/Q3 decrescem monotonicamente com a categoria"),
        (None, "(0 inversões) — a isotônica de salvaguarda não precisou atuar."),
    ], fs=12.4, dy=0.063)
    ax = fig.add_axes([0.50, 0.10, 0.48, 0.70])
    ax.imshow(plt.imread("outputs/figures/sobrevivencia_weibull_extrap_mob_2124.png")); ax.axis("off")
    footer(fig, "B2"); pages.append(fig)
surv_weibull()
IDX["B2"] = len(pages) - 1

def surv_estatisticas():
    fig = new_slide(); header(fig, "TEMPO ATÉ O DESLIGAMENTO · ESTATÍSTICAS",
                              "Q1, mediana, média e Q3 por categoria (meses MOB)")
    s = STAT.sort_values("categoria")
    ax = fig.add_axes([0.05, 0.12, 0.55, 0.68])
    ks = s.categoria.values
    for i, k in enumerate(ks):
        cor = gcolor(int(k))
        q1, q3 = s.q1_meses_mono.iloc[i], s.q3_meses_mono.iloc[i]
        med, mea = s.mediana_meses_mono.iloc[i], s.media_meses_mono.iloc[i]
        ax.bar(k, q3 - q1, bottom=q1, width=0.64, color=cor, alpha=0.45, edgecolor=cor, lw=1.1)
        ax.hlines(med, k - .32, k + .32, color="#222", lw=2.2)
        ax.plot(k, mea, marker="D", ms=6, color="white", mec="#222", mew=1.3)
    ax.set_yscale("log"); ax.set_xticks(ks)
    ax.set_xlabel("categoria de risco"); ax.set_ylabel("tempo até desligamento (meses, log)")
    ax.set_title("Caixa = IQR (Q1–Q3) · linha = mediana · losango = média", fontsize=11)
    for ym, lb in [(12, "12m"), (36, "36m")]:
        ax.axhline(ym, color="#888", lw=.8, ls=":"); ax.text(0.55, ym, lb, fontsize=8, color="#666", va="bottom")
    ax.grid(axis="y", which="both", alpha=.2)
    ax2 = fig.add_axes([0.64, 0.08, 0.33, 0.74]); ax2.axis("off")
    cell = [[int(r.categoria), f"{r.q1_meses_mono:.0f}", f"{r.mediana_meses_mono:.0f}",
             f"{r.media_meses_mono:.0f}", f"{r.q3_meses_mono:.0f}"] for _, r in s.iterrows()]
    col = ["cat", "Q1", "mediana", "média", "Q3"]
    tbl = ax2.table(cellText=cell, colLabels=col, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.45)
    for j in range(len(col)):
        c = tbl[0, j]; c.set_facecolor(NAVY); c.set_text_props(color="white", weight="bold")
    for i in range(1, len(cell) + 1):
        tbl[i, 0].set_facecolor(gcolor(cell[i-1][0])); tbl[i, 0].set_text_props(color="white", weight="bold")
    fig.text(0.64, 0.045, "Meses até o desligamento (>12m = extrapolação Weibull).", fontsize=9, color=GREY)
    footer(fig, "B3"); pages.append(fig)
surv_estatisticas()
IDX["B3"] = len(pages) - 1

# ======================= APÊNDICE C — POLÍTICA DE CONSIGNADO =======================
divisor("Apêndice C — Política de prazos e taxas do consignado",
        "Prazo máximo, cobertura de parcelas e taxa de juros mínima, por categoria", "C")

def consignado_conceitos():
    fig = new_slide(); header(fig, "APLICAÇÃO · COMO LER AS TABELAS",
                              "Da curva de sobrevivência aos números da política")
    bullet(fig, 0.05, 0.77, [
        (False, "Ponto de partida: S(t), a curva de sobrevivência da categoria"),
        (True, "S(t) = probabilidade de o vínculo seguir ativo t meses após a entrada."),
        (True, "Até 12 meses: medida direto dos dados (Kaplan-Meier); além de 12:"),
        (None, "extrapolada pela curva de Weibull S(t)=exp(−(t/λ)ᵖ) ajustada à categoria."),
        (False, "Prazo máximo por confiança c  (tabela da esquerda)"),
        (True, "É o maior prazo t com confiança c de o tomador seguir empregado:"),
        (None, "S(t) = c  ⇒  t = λ·(−ln c)^(1/p).  Ex.: c=90% → prazo onde S cai a 0,90."),
        (True, "Colunas 95/90/85/80% = quão conservador é o limite (maior c = prazo menor)."),
        (False, "Cobertura esperada de parcelas T  (tabela da direita)"),
        (True, "Fração média das T parcelas que devem ser pagas em folha (com vínculo):"),
        (None, "cobertura(T) = (S(1)+S(2)+…+S(T)) / T.  Ex.: 90% ≈ 9 de cada 10 parcelas."),
    ], fs=12.2, dy=0.0585)
    ax = fig.add_axes([0.63, 0.13, 0.33, 0.60]); ax.set_xlim(0, 36); ax.set_ylim(0, 1.02)
    import numpy as _np
    from math import log as _ln
    p, lam = 1.15, 30.0
    t = _np.linspace(0.2, 36, 200); S = _np.exp(-(t / lam) ** p)
    ax.fill_between(t, S, color=BLUE, alpha=0.10)
    ax.plot(t, S, color=BLUE, lw=2.4)
    ax.axhline(0.90, ls="--", color="#d9822b", lw=1.2)
    t90 = lam * (-_ln(0.90)) ** (1 / p)
    ax.plot([t90, t90], [0, 0.90], ls="--", color="#d9822b", lw=1.2)
    ax.plot(t90, 0.90, "o", color="#d9822b")
    ax.text(0.5, 0.93, "c = 90%", fontsize=9, color="#b9671a")
    ax.annotate("prazo máx.", (t90, 0.02), (t90 + 1.5, 0.20), fontsize=9, color="#b9671a",
                arrowprops=dict(arrowstyle="->", color="#b9671a"))
    ax.text(14, 0.45, "área ÷ T\n= cobertura", fontsize=8.5, color=BLUE, ha="center")
    ax.set_xlabel("meses (MOB)", fontsize=9.5); ax.set_ylabel("S(t)", fontsize=9.5)
    ax.set_title("S(t) define prazo e cobertura", fontsize=10.5, weight="bold")
    ax.grid(alpha=.25)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.text(0.05, 0.06, "λ (escala) e p (forma) vêm do ajuste de Weibull de cada categoria. "
             ">12 meses é projeção; ≤12 é observado.", fontsize=9.5, color=GREY)
    footer(fig, "C1"); pages.append(fig)
consignado_conceitos()

def consignado_tabelas():
    fig = new_slide(); header(fig, "APLICAÇÃO · CRÉDITO CONSIGNADO",
                              "Tabela de referência para a política de concessão")
    ax = fig.add_axes([0.03, 0.07, 0.34, 0.74]); ax.axis("off")
    cell = []
    for _, r in PRAZO.iterrows():
        cell.append([int(r.categoria)] + [("120+" if r[c] > 120 else f"{r[c]:.0f}")
                                          for c in ("conf_95", "conf_90", "conf_85", "conf_80")])
    col = ["cat", "95%", "90%", "85%", "80%"]
    tbl = ax.table(cellText=cell, colLabels=col, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.6); tbl.scale(1, 1.5)
    for j in range(len(col)):
        c = tbl[0, j]; c.set_facecolor(NAVY); c.set_text_props(color="white", weight="bold")
    for i in range(1, len(cell) + 1):
        tbl[i, 0].set_facecolor(gcolor(cell[i-1][0])); tbl[i, 0].set_text_props(color="white", weight="bold")
    fig.text(0.05, 0.83, "Prazo máx. (meses) por confiança\nde seguir empregado", fontsize=11.5,
             weight="bold", color=INK)
    ax2 = fig.add_axes([0.40, 0.07, 0.57, 0.74]); ax2.axis("off")
    TS = [6, 12, 18, 24, 36, 48, 60]
    cell2 = [[int(r.categoria)] + [f"{r[f'T_{t}']:.0f}%" for t in TS] for _, r in COB.iterrows()]
    col2 = ["cat"] + [f"T={t}" for t in TS]
    tb2 = ax2.table(cellText=cell2, colLabels=col2, loc="center", cellLoc="center")
    tb2.auto_set_font_size(False); tb2.set_fontsize(9.2); tb2.scale(1, 1.5)
    for j in range(len(col2)):
        c = tb2[0, j]; c.set_facecolor(NAVY); c.set_text_props(color="white", weight="bold")
    for i in range(1, len(cell2) + 1):
        tb2[i, 0].set_facecolor(gcolor(cell2[i-1][0])); tb2[i, 0].set_text_props(color="white", weight="bold")
    fig.text(0.42, 0.83, "Cobertura esperada de parcelas (% pagas em folha) por prazo T", fontsize=11.5,
             weight="bold", color=INK)
    fig.text(0.03, 0.035, "Prazo: t = λ·(−ln c)^(1/p). Cobertura: Σ S(m)/T, com S = KM (≤12 MOB) + Weibull (>12). "
             "Referência 2021–2024. >12m é projeção.", fontsize=9.5, color=GREY)
    footer(fig, "C2"); pages.append(fig)
consignado_tabelas()
IDX["CTAB"] = len(pages) - 1

def taxa_conceitos():
    fig = new_slide(); header(fig, "APLICAÇÃO · TAXA DE JUROS MÍNIMA",
                              "Da cobertura de parcelas à taxa de equilíbrio")
    bullet(fig, 0.05, 0.77, [
        (False, "A pergunta"),
        (True, "Qual a MENOR taxa que ainda recupera o valor emprestado, dado que parte"),
        (None, "das parcelas deixa de ser paga quando o tomador é desligado?"),
        (False, "O modelo financeiro (Tabela Price)"),
        (True, "Parcela fixa:  A = P · i / (1 − (1+i)^−T)   (P = principal, i = juros, T = prazo)."),
        (True, "Só se paga a parcela do mês m se o vínculo seguir ativo — prob. S(m)."),
        (None, "Recebido esperado = A · Σ S(m) = A · T · cobertura(T)."),
        (False, "Break-even (receber ≥ principal)"),
        (True, "A · T · cobertura(T) ≥ P   ⟺   i · T · c / (1 − (1+i)^−T) ≥ 1."),
        (True, "Resolve-se i (a taxa mínima) numericamente para cada categoria e prazo;"),
        (None, "anual = (1+i)¹² − 1.  É um PISO de quebra-zero — some custo de funding + margem."),
    ], fs=12.1, dy=0.0575)
    # diagrama: recebido esperado vs principal
    ax = fig.add_axes([0.63, 0.13, 0.33, 0.60]); ax.set_xlim(0, 1); ax.set_ylim(0, 1.35)
    ax.axhline(1.0, ls="--", color="#444", lw=1.2); ax.text(0.02, 1.04, "principal (P)", fontsize=9, color="#444")
    import numpy as _np
    cats_ex = [2, 6, 10, 13]; xs = _np.arange(len(cats_ex))
    # recebido com taxa baixa (1%/mês, T=24) -> mostra quem recupera e quem não
    i_demo = 0.01; T = 24
    A = i_demo / (1 - (1 + i_demo) ** (-T))
    vals = [A * T * (float(COB.loc[COB.categoria == c, "T_24"].iloc[0]) / 100.0) for c in cats_ex]
    cores = [gcolor(c) for c in cats_ex]
    ax.bar(xs, vals, color=cores, width=0.6)
    for x, v, c in zip(xs, vals, cats_ex):
        ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels([f"cat {c}" for c in cats_ex], fontsize=8.5)
    ax.set_ylabel("recebido / principal", fontsize=9)
    ax.set_title("Mesma taxa (1%/mês, T=24): só\nbaixo risco recupera P", fontsize=9.5, weight="bold")
    ax.set_yticks([0, 0.5, 1.0]); ax.tick_params(labelsize=8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.text(0.05, 0.055, "Hipótese conservadora: ZERO recuperação do saldo após o desligamento "
             "(FGTS/rescisão/portabilidade reduziriam o piso). Valores nominais.", fontsize=9.5, color=GREY)
    footer(fig, "C3"); pages.append(fig)
taxa_conceitos()

def consignado_taxas():
    fig = new_slide(); header(fig, "APLICAÇÃO · TAXA DE JUROS MÍNIMA",
                              "Taxa de equilíbrio (% ao mês) por categoria e prazo")
    ax = fig.add_axes([0.05, 0.10, 0.66, 0.70]); ax.axis("off")
    TS = [6, 12, 18, 24, 36, 48, 60]
    cell = [[int(r.categoria)] + [f"{r[f'm_T{t}']:.2f}" for t in TS] for _, r in TAXA.iterrows()]
    col = ["cat"] + [f"T={t}" for t in TS]
    tbl = ax.table(cellText=cell, colLabels=col, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.6); tbl.scale(1, 1.55)
    for j in range(len(col)):
        c = tbl[0, j]; c.set_facecolor(NAVY); c.set_text_props(color="white", weight="bold")
    for i in range(1, len(cell) + 1):
        tbl[i, 0].set_facecolor(gcolor(cell[i-1][0])); tbl[i, 0].set_text_props(color="white", weight="bold")
        if i % 2 == 0:
            for j in range(1, len(col)): tbl[i, j].set_facecolor("#f4f7fb")
    fig.text(0.05, 0.84, "Taxa MÍNIMA para recuperar o principal — % ao mês", fontsize=12, weight="bold", color=INK)
    # destaque executivo
    ax2 = fig.add_axes([0.74, 0.12, 0.22, 0.66]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02", facecolor=LIGHT, linewidth=0))
    ax2.text(0.5, 0.93, "Leitura (T=24)", ha="center", fontsize=11.5, weight="bold", color=NAVY)
    destaques = [(1, "0,06%/mês"), (4, "0,82%/mês"), (8, "2,68%/mês"), (11, "5,18%/mês"), (14, "18,2%/mês")]
    for k, (c, txt) in enumerate(destaques):
        y = 0.78 - k * 0.155
        ax2.add_patch(Rectangle((0.06, y - 0.02), 0.06, 0.10, color=gcolor(c)))
        ax2.text(0.18, y + 0.03, f"cat {c}", fontsize=10.5, color=INK, va="center", weight="bold")
        ax2.text(0.97, y + 0.03, txt, fontsize=10.5, color=BLUE, va="center", ha="right", weight="bold")
    fig.text(0.74, 0.075, "No HTML: botão alterna\nentre % ao mês e % ao ano.", fontsize=9, color=GREY)
    fig.text(0.05, 0.045, "Piso de quebra-zero (recebimento nominal ≥ principal). Taxa praticada = piso + "
             "funding + custo operacional + margem. Categorias altas só fecham em prazos curtos.",
             fontsize=9.5, color=GREY)
    footer(fig, "C4"); pages.append(fig)
consignado_taxas()
IDX["TAXTAB"] = len(pages) - 1

# ======================= SÍNTESE (último slide) =======================
fecho()
IDX["FECHO"] = len(pages) - 1

# ======================= salvar =======================
_DUMP = os.environ.get("DECK_DUMP_PNG")
_FMT = os.environ.get("DECK_DUMP_FMT", "png")
if _DUMP:
    os.makedirs(_DUMP, exist_ok=True)
import shutil as _sh
with PdfPages("/tmp/apresentacao_risco_2124.pdf") as pdf:
    for i, f in enumerate(pages):
        pdf.savefig(f, facecolor="white")
        if _DUMP:
            f.savefig(f"{_DUMP}/slide_{i:02d}.{_FMT}", dpi=100, facecolor="white")
        plt.close(f)
_sh.copy("/tmp/apresentacao_risco_2124.pdf", PDF)
print(f"OK: {PDF} ({len(pages)} slides)")
