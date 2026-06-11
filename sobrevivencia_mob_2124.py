"""Sobrevivência MOB das 14 categorias NOVAS — referência 2021–2024 agrupados.

Mesma metodologia da §6-B (CLAUDE.md), em 3 passos:
  1) KM observado (1..12 MOB): evento = dispensa s/ justa causa; censura = ativo
     em 31/12 ou saída por outro motivo; entrada = mes_admissao (pré-existente: jan).
     Fonte: data/processed/predicoes_2124/ano={2021..2024} (categoria + desfecho
     no MESMO parquet — sem alinhamento posicional).
  2) Extrapolação Weibull por REGRESSÃO PURA: cloglog ln(−ln S)=p·ln t+ln α (OLS,
     12 pontos) -> S(t)=exp(−α·t^p) até 36 MOB; Q1/mediana/média/Q3 em forma fechada.
  3) Monotonização isotônica (PAVA ponderado por n) das estatísticas.

Saídas (sufixo _mob_2124): tables/sobrevivencia_{km,resumo,weibull_params,
weibull_extrap,weibull_estatisticas,weibull_estatisticas_mono}_mob_2124.csv
+ figures/sobrevivencia_{categorias,weibull_extrap}_mob_2124.png

Uso: MPLCONFIGDIR=/tmp/mpl /tmp/consig_venv/bin/python sobrevivencia_mob_2124.py
"""
import os, glob, time, math, shutil
import numpy as np, pandas as pd
import pyarrow.parquet as pq

PRED = "data/processed/predicoes_2124"
ANOS_REF = [2021, 2022, 2023, 2024]
TARGET = "involuntario_sjc"
H = 12; H_EXT = 36
TAB = "outputs/tables"; FIG = "outputs/figures"
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
COUNTS = f"{TAB}/_surv_counts_mob_2124.csv"

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# ---------------------------------------------------------------------------
# Passo 1A — estatística suficiente (categoria, mob) -> eventos/censuras (cache)
# ---------------------------------------------------------------------------
if not os.path.exists(COUNTS):
    log("Passo 1A: agregando eventos/censuras por (categoria, MOB), 2021–2024 ...")
    acc = {}
    for a in ANOS_REF:
        for fp in sorted(glob.glob(f"{PRED}/ano={a}/*.parquet")):
            d = pq.ParquetFile(fp).read(
                columns=["categoria_risco", "mes_admissao", "mes_deslig", "motivo_unificado"]
            ).to_pandas()
            mes_a = d["mes_admissao"].to_numpy("int16")
            mes_d = d["mes_deslig"].to_numpy("int16")
            evento = (d["motivo_unificado"].to_numpy() == TARGET).astype("int8")
            cat = d["categoria_risco"].to_numpy("int16")
            e = np.where((mes_a >= 1) & (mes_a <= 12), mes_a, 1).astype("int16")
            m_cal = np.where((mes_d >= 1) & (mes_d <= 12), mes_d, 12).astype("int16")
            mob = np.clip(m_cal - e + 1, 1, H).astype("int16")
            g = pd.DataFrame({"categoria": cat, "mob": mob, "evento": evento}) \
                  .groupby(["categoria", "mob", "evento"]).size()
            for key, n in g.items():
                acc[key] = acc.get(key, 0) + int(n)
            log(f"  {a}/{os.path.basename(fp)} ({len(d):,})")
            del d
    rows = [{"categoria": k[0], "mob": k[1], "evento": k[2], "n": v} for k, v in acc.items()]
    g = (pd.DataFrame(rows).pivot_table(index=["categoria", "mob"], columns="evento",
                                        values="n", fill_value=0).reset_index())
    g = g.rename(columns={0: "censuras", 1: "eventos"})
    for c in ("censuras", "eventos"):
        if c not in g.columns: g[c] = 0
    g[["categoria", "mob", "eventos", "censuras"]].sort_values(["categoria", "mob"]) \
        .to_csv(COUNTS, index=False)
    log(f"Passo 1A ok -> {COUNTS}")
else:
    log(f"Passo 1A: cache {COUNTS}")
counts = pd.read_csv(COUNTS)

# ---------------------------------------------------------------------------
# Passo 1B — KM por categoria (relógio MOB) + Greenwood + RMST(12)
# ---------------------------------------------------------------------------
log("Passo 1B: Kaplan-Meier ...")
km_rows, resumo_rows = [], []
for k in sorted(counts["categoria"].unique()):
    sub = counts[counts["categoria"] == k].set_index("mob").reindex(range(1, H + 1), fill_value=0)
    d = sub["eventos"].to_numpy(float); c = sub["censuras"].to_numpy(float)
    N = int((d + c).sum()); n = N; S = 1.0; var_acc = 0.0
    km_rows.append(dict(categoria=int(k), mob=0, n_risco=N, eventos=0, censuras=0,
                        S=1.0, S_lo=1.0, S_hi=1.0))
    rmst = 0.0; mediana = None; surv = {0: 1.0}
    for m in range(1, H + 1):
        n_risco = n; dm, cm = d[m - 1], c[m - 1]
        if n_risco > 0:
            S *= (n_risco - dm) / n_risco
            if n_risco - dm > 0:
                var_acc += dm / (n_risco * (n_risco - dm))
        se = S * np.sqrt(var_acc)
        km_rows.append(dict(categoria=int(k), mob=m, n_risco=int(n_risco), eventos=int(dm),
                            censuras=int(cm), S=S,
                            S_lo=max(0.0, S - 1.96 * se), S_hi=min(1.0, S + 1.96 * se)))
        rmst += surv[m - 1]; surv[m] = S
        if mediana is None and S <= 0.5: mediana = m
        n = n_risco - dm - cm
    resumo_rows.append(dict(categoria=int(k), n=N, eventos_12m=int(d.sum()),
                            S12=surv[H], risco_deslig_12m_KM=1 - surv[H],
                            RMST12_meses=rmst, mediana_mob=mediana))
km = pd.DataFrame(km_rows); km.to_csv(f"{TAB}/sobrevivencia_km_mob_2124.csv", index=False)
res = pd.DataFrame(resumo_rows); res.to_csv(f"{TAB}/sobrevivencia_resumo_mob_2124.csv", index=False)
log("KM salvo")

# ---------------------------------------------------------------------------
# Passo 2 — Weibull por regressão cloglog (12 pontos) + estatísticas fechadas
# ---------------------------------------------------------------------------
log("Passo 2: Weibull (cloglog OLS) ...")
param_rows, curve_rows = [], []
for k in sorted(km["categoria"].unique()):
    g = km[km.categoria == k].sort_values("mob")
    Sobs = {int(m): float(s) for m, s in zip(g.mob, g.S)}
    fit = g[(g.mob >= 1) & (g.mob <= H) & (g.S > 0) & (g.S < 1)]
    x = np.log(fit.mob.values.astype(float)); yv = np.log(-np.log(fit.S.values))
    p, ln_a = np.polyfit(x, yv, 1)
    alpha = float(np.exp(ln_a)); p = float(p)
    lam = alpha ** (-1.0 / p)
    r2 = float(np.corrcoef(x, yv)[0, 1] ** 2)
    S_w = lambda tt: np.exp(-alpha * np.asarray(tt, float) ** p)
    tq = lambda q: float(lam * (-math.log(1.0 - q)) ** (1.0 / p))
    media = float(lam * math.gamma(1.0 + 1.0 / p))
    param_rows.append(dict(categoria=int(k), shape_p=round(p, 4),
                           escala_lambda_meses=round(lam, 2), alpha=round(alpha, 6),
                           R2_ajuste=round(r2, 4), S12_obs=round(Sobs[H], 4),
                           S12_fit=round(float(S_w(12)), 4), S24_weib=round(float(S_w(24)), 4),
                           S36_weib=round(float(S_w(36)), 4),
                           media_meses=round(media, 1), q1_meses=round(tq(.25), 1),
                           mediana_meses=round(tq(.50), 1), q3_meses=round(tq(.75), 1),
                           IQR_meses=round(tq(.75) - tq(.25), 1)))
    for m in range(0, H_EXT + 1):
        curve_rows.append(dict(categoria=int(k), mes=m,
                               S_obs=(round(Sobs[m], 5) if m in Sobs else np.nan),
                               S_weibull=round(float(S_w(m)) if m >= 1 else 1.0, 5)))
params = pd.DataFrame(param_rows)
params.to_csv(f"{TAB}/sobrevivencia_weibull_params_mob_2124.csv", index=False)
pd.DataFrame(curve_rows).to_csv(f"{TAB}/sobrevivencia_weibull_extrap_mob_2124.csv", index=False)
params[["categoria", "shape_p", "escala_lambda_meses", "media_meses", "q1_meses",
        "mediana_meses", "q3_meses", "IQR_meses"]] \
    .to_csv(f"{TAB}/sobrevivencia_weibull_estatisticas_mob_2124.csv", index=False)
log(f"Weibull salvo | R² médio = {params.R2_ajuste.mean():.4f}")

# ---------------------------------------------------------------------------
# Passo 3 — monotonização isotônica (PAVA ponderado por n)
# ---------------------------------------------------------------------------
log("Passo 3: isotônica ...")
s = params.sort_values("categoria").reset_index(drop=True)
w = s["categoria"].map(res.set_index("categoria")["n"]).to_numpy(float)

def isotonic_decreasing(y, w):
    y = np.asarray(y, float); w = np.asarray(w, float)
    neg = -y
    val, cnt, wsum = [], [], []
    for yi, wi in zip(neg, w):
        val.append(yi); cnt.append(1); wsum.append(wi)
        while len(val) > 1 and val[-2] > val[-1]:
            nw = wsum[-2] + wsum[-1]
            nv = (val[-2] * wsum[-2] + val[-1] * wsum[-1]) / nw
            nc = cnt[-2] + cnt[-1]
            val[-2:] = [nv]; wsum[-2:] = [nw]; cnt[-2:] = [nc]
    out = []
    for v, c in zip(val, cnt):
        out += [v] * c
    return -np.asarray(out)

for c in ["q1_meses", "mediana_meses", "media_meses", "q3_meses"]:
    s[c + "_mono"] = np.round(isotonic_decreasing(s[c].to_numpy(), w), 1)
    v = s[c + "_mono"].to_numpy()
    ok = all(v[i + 1] <= v[i] + 1e-9 for i in range(len(v) - 1))
    log(f"  {c}_mono: {'OK' if ok else 'VIOLA'}")
s[["categoria", "shape_p", "escala_lambda_meses",
   "q1_meses", "q1_meses_mono", "mediana_meses", "mediana_meses_mono",
   "media_meses", "media_meses_mono", "q3_meses", "q3_meses_mono"]] \
    .to_csv(f"{TAB}/sobrevivencia_weibull_estatisticas_mono_mob_2124.csv", index=False)

# ---------------------------------------------------------------------------
# Figuras: KM (bolinhas+linha) e extrapolação Weibull
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.ticker import PercentFormatter
ks = sorted(km["categoria"].unique())
cmap = matplotlib.colormaps["RdYlGn_r"]; norm = mcolors.Normalize(vmin=min(ks), vmax=max(ks))
cor = {k: cmap(norm(k)) for k in ks}

fig, ax = plt.subplots(figsize=(11, 6.5))
for k in ks:
    g = km[km.categoria == k].sort_values("mob")
    ax.plot(g.mob, g.S, marker="o", ms=3.5, lw=1.6, color=cor[k], label=f"{k}")
ax.set_xlabel("MOB — meses desde a entrada"); ax.set_ylabel("S(t) = P(seguir empregado)")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
ax.set_title("Sobrevivência por categoria (KM, relógio MOB) — referência 2021–2024, 14 categorias")
ax.grid(alpha=.3); ax.legend(title="cat", ncol=2, fontsize=8)
fig.tight_layout(); fig.savefig("/tmp/skm2124.png", dpi=130)
shutil.copy("/tmp/skm2124.png", f"{FIG}/sobrevivencia_categorias_mob_2124.png"); plt.close(fig)

cur = pd.read_csv(f"{TAB}/sobrevivencia_weibull_extrap_mob_2124.csv")
fig, ax = plt.subplots(figsize=(11, 6.5))
for k in ks:
    g = cur[cur.categoria == k].sort_values("mes")
    obs = g[g.mes <= H]
    ax.plot(obs.mes, obs.S_obs, marker="o", ms=3, lw=1.4, color=cor[k])
    ext = g[g.mes >= H]
    ax.plot(ext.mes, ext.S_weibull, ls="--", lw=1.2, color=cor[k])
ax.axvline(H, color="#888", ls=":", lw=1)
ax.text(H + .3, .02, "observado ◄ | ► extrapolado (Weibull)", fontsize=9, color="#666")
ax.set_xlabel("MOB (meses)"); ax.set_ylabel("S(t)")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
ax.set_title("Extrapolação Weibull até 36 MOB — regressão cloglog nos 12 pontos observados")
ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("/tmp/swb2124.png", dpi=130)
shutil.copy("/tmp/swb2124.png", f"{FIG}/sobrevivencia_weibull_extrap_mob_2124.png"); plt.close(fig)

log("figuras salvas")
print("FIM_SURV_2124", flush=True)
