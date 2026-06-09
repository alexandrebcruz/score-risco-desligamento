"""Gera um PDF (A4) explicando passo a passo o cálculo da cobertura esperada de parcelas
para a categoria 9, prazo T=60 (visão MOB), com as fórmulas renderizadas em LaTeX
(mathtext do matplotlib — sem dependência de LaTeX instalado).

Saída: outputs/explicacao_cobertura_cat9_mob.pdf
"""
import os, shutil
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

NAVY = "#14233f"; GREY = "#5b6675"; INK = "#1b2430"
fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

L = 0.08          # margem esquerda
y = 0.955         # cursor vertical
def line(txt, dy=0.034, fs=11.5, color=INK, x=L, weight="normal"):
    global y
    ax.text(x, y, txt, fontsize=fs, color=color, weight=weight, va="top", ha="left")
    y -= dy
def head(txt):
    global y
    y -= 0.012
    line(txt, dy=0.040, fs=13, color=NAVY, weight="bold")
def math(txt, dy=0.052, fs=14):
    line(txt, dy=dy, fs=fs, x=0.12)

# título
line("Cobertura esperada de parcelas — passo a passo", dy=0.040, fs=17, color=NAVY, weight="bold")
line("Categoria 9  ·  prazo T = 60 meses  ·  visão MOB (months on book)", dy=0.050, fs=11, color=GREY)

head("A ideia")
line("A parcela do mês m é paga (desconto em folha) apenas se a pessoa seguir empregada em m;")
line("essa probabilidade é S(m). Logo, o nº esperado de parcelas pagas e a cobertura são:")
math(r"$\mathrm{parcelas\ esperadas}=\sum_{m=1}^{T} S(m)"
     r"\qquad \mathrm{cobertura}(T)=\frac{1}{T}\sum_{m=1}^{T} S(m)$", dy=0.060, fs=15)

head("De onde vem S(m)")
line(r"$\bullet$  m = 1..12:  observado (Kaplan-Meier, relógio MOB).")
line(r"$\bullet$  m = 13..60:  extrapolado pela Weibull ajustada —")
math(r"$S(m)=\mathrm{exp}\left[-\left(\frac{m}{\lambda}\right)^{p}\right],"
     r"\quad p=1{,}2354,\ \ \lambda=47{,}85\ \mathrm{meses}$", dy=0.060, fs=15)

head("Passo a passo (cat 9, T = 60)")
line("1) Parte observada (m = 1..12), S vai de 0,993 (mês 1) a 0,847 (mês 12):")
math(r"$\sum_{m=1}^{12} S(m) = 11{,}011$")
line("2) Parte extrapolada (m = 13..60), ex.: S(24)=0,653; S(36)=0,495; S(60)=0,266:")
math(r"$\sum_{m=13}^{60} S(m) = 24{,}391$")
line("3) Total de parcelas esperadas pagas:")
math(r"$\sum_{m=1}^{60} S(m) = 11{,}011 + 24{,}391 = 35{,}401$")
line("4) Cobertura:")
math(r"$\mathrm{cobertura}(60)=\frac{35{,}401}{60}=0{,}590=59{,}0\%$", dy=0.058, fs=15)

head("Interpretação")
line("Num consignado de 60 parcelas para a cat 9, esperam-se ~35 das 60 pagas em folha (~59%);")
line("as ~41% restantes seriam interrompidas por desligamento ao longo do prazo — a cobrir por")
line("mitigantes (rescisão/FGTS, portabilidade, conversão para crédito pessoal).")
y -= 0.006
line("Atenção: como T = 60 > 12, a maior parte da soma (24,4 de 35,4) vem da cauda EXTRAPOLADA",
     dy=0.030, fs=10.5, color=GREY)
line("pela Weibull — é projeção, sujeita às ressalvas (sazonalidade de dezembro, frailty;",
     dy=0.030, fs=10.5, color=GREY)
line("o ideal é calibrar a cauda com a RAIS 2024/2025).", dy=0.030, fs=10.5, color=GREY)

OUT = "outputs/explicacao_cobertura_cat9_mob.pdf"
TMP = "/tmp/explicacao_cobertura_cat9_mob.pdf"
with PdfPages(TMP) as pdf:
    pdf.savefig(fig, facecolor="white")
plt.close(fig)
os.makedirs("outputs", exist_ok=True)
shutil.copy(TMP, OUT)
print(f"FIM -> {OUT}")
