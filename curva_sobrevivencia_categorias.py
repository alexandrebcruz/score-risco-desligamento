"""Curva de sobrevivência (Kaplan-Meier) do emprego por categoria de risco (1..23),
restrita à janela OBSERVÁVEL de 12 meses — SEM extrapolação.

Conceito (ver CLAUDE.md / discussão de "tempo até desligamento"):
- evento  = dispensa sem justa causa  (motivo_unificado == "involuntario_sjc").
- tempo   = mês do desligamento (mes_deslig, 1..12).
- censura = ativo em dezembro (mes_deslig==0 -> censurado em 12) OU saída por OUTRO
            motivo no mês m (censurado em m: deixou de estar em risco sem o evento).
Estima S(t) = P(continuar empregado após t meses) por categoria, com IC95% (Greenwood),
e resume cada categoria por mediana de sobrevivência e RMST(12) = meses esperados de
emprego dentro do ano (= área sob S(t)); "meses perdidos" = 12 - RMST.

Alinhamento: o parquet categorizado preserva a MESMA ordem de linha do interim 2023
(merge em ordem de partição/lote). Recompomos `mes_deslig`/`motivo` do interim na mesma
ordem e VALIDAMOS que o evento recalculado == coluna `y` salva (garante o alinhamento).

Uso:  MPLCONFIGDIR=/tmp/mpl /tmp/consig_venv/bin/python curva_sobrevivencia_categorias.py
Saídas:
  outputs/tables/sobrevivencia_km_2023.csv       (longo: categoria x mes -> n_risco,S,IC)
  outputs/tables/sobrevivencia_resumo_2023.csv   (1 linha por categoria: RMST, mediana...)
  outputs/figures/sobrevivencia_categorias_2023.png
"""
import os, glob, time, shutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

INTERIM = "data/interim/rais/ano=2023"
CATPARQ = "outputs/predicoes_2023_ensemble_base_categorizado.parquet"
TARGET = "involuntario_sjc"
H = 12  # horizonte observável (meses)

TAB = "outputs/tables"; FIG = "outputs/figures"
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
COUNTS = f"{TAB}/_surv_counts_2023.csv"   # estatística suficiente p/ KM (cache resumível)

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# ---------------------------------------------------------------------------
# Passo A — montar a estatística suficiente: por (categoria, mes) -> eventos, censuras.
# (cacheado; é só ~23*12 linhas, então o KM recomputa em instantes)
# ---------------------------------------------------------------------------
if not os.path.exists(COUNTS):
    log("Passo A: lendo evento/tempo do interim e categoria do parquet categorizado ...")

    # (1) interim, na ordem de iteração (== ordem do categorizado): evento + mes_deslig
    ev_parts, mes_parts = [], []
    for fp in sorted(glob.glob(f"{INTERIM}/*.parquet")):
        pf = pq.ParquetFile(fp)
        for rb in pf.iter_batches(batch_size=4_000_000,
                                  columns=["motivo_unificado", "mes_deslig"]):
            mot = rb.column("motivo_unificado").to_numpy(zero_copy_only=False)
            ev_parts.append((mot == TARGET).astype("int8"))
            mes_parts.append(rb.column("mes_deslig").to_numpy(zero_copy_only=False).astype("int16"))
        log(f"  interim lido: {os.path.basename(fp)}")
    evento = np.concatenate(ev_parts); mes = np.concatenate(mes_parts)
    del ev_parts, mes_parts

    # (2) categorizado, mesma ordem: categoria_risco + y (p/ validar alinhamento)
    cat_parts, y_parts = [], []
    for rb in pq.ParquetFile(CATPARQ).iter_batches(batch_size=4_000_000,
                                                   columns=["categoria_risco", "y"]):
        cat_parts.append(rb.column("categoria_risco").to_numpy(zero_copy_only=False).astype("int16"))
        y_parts.append(rb.column("y").to_numpy(zero_copy_only=False).astype("int8"))
    categoria = np.concatenate(cat_parts); y = np.concatenate(y_parts)
    del cat_parts, y_parts

    # (3) VALIDAÇÃO de alinhamento: evento recomputado tem de bater com y salvo
    assert len(evento) == len(categoria) == len(y), "tamanhos divergentes — ordem quebrada"
    mism = int((evento != y).sum())
    if mism:
        raise SystemExit(f"ALINHAMENTO FALHOU: {mism:,} linhas com evento!=y. Abortando.")
    log(f"alinhamento OK: {len(y):,} linhas, evento==y em 100%")

    # (4) tempo até evento/censura (clip à janela observável de 12 meses):
    #     ativo (mes==0) -> censurado em 12; saída no mês m (1..12) -> tempo m.
    #     Mesma fórmula serve a evento e censura; o que difere é o flag `evento`.
    tempo = np.where((mes >= 1) & (mes <= H), mes, H).astype("int16")
    anom = int(((evento == 1) & ~((mes >= 1) & (mes <= H))).sum())  # evento sem mês válido
    if anom:
        log(f"AVISO: {anom:,} eventos sem mês válido (1..12) -> alocados em t=12")

    # (5) agrega: (categoria, tempo, evento) -> contagem  [estatística suficiente]
    df = pd.DataFrame({"categoria": categoria, "mes": tempo, "evento": evento})
    g = (df.groupby(["categoria", "mes", "evento"]).size()
           .unstack("evento", fill_value=0).reset_index())
    g = g.rename(columns={0: "censuras", 1: "eventos"})
    for c in ("censuras", "eventos"):
        if c not in g.columns: g[c] = 0
    g = g[["categoria", "mes", "eventos", "censuras"]].sort_values(["categoria", "mes"])
    g.to_csv(COUNTS, index=False)
    log(f"Passo A ok -> {COUNTS} ({len(g)} linhas)")
else:
    log(f"Passo A: usando cache {COUNTS}")

counts = pd.read_csv(COUNTS)

# ---------------------------------------------------------------------------
# Passo B — Kaplan-Meier por categoria (a partir das contagens) + IC Greenwood + RMST
# ---------------------------------------------------------------------------
log("Passo B: estimando KM por categoria ...")
km_rows, resumo_rows = [], []
for k in sorted(counts["categoria"].unique()):
    sub = counts[counts["categoria"] == k].set_index("mes").reindex(range(1, H + 1),
                                                                     fill_value=0)
    d = sub["eventos"].to_numpy(dtype=float)      # eventos no mês m
    c = sub["censuras"].to_numpy(dtype=float)     # censuras no mês m
    N = int((d + c).sum())                        # total na categoria
    n = N                                         # em risco no início do mês 1
    S = 1.0; var_acc = 0.0
    # m=0: S=1 (âncora p/ RMST e plot)
    km_rows.append(dict(categoria=int(k), mes=0, n_risco=N, eventos=0, censuras=0,
                        S=1.0, S_lo=1.0, S_hi=1.0))
    rmst = 0.0; mediana = None
    surv_por_mes = {0: 1.0}
    for m in range(1, H + 1):
        n_risco = n
        dm, cm = d[m - 1], c[m - 1]
        if n_risco > 0:
            S *= (n_risco - dm) / n_risco
            if n_risco - dm > 0:
                var_acc += dm / (n_risco * (n_risco - dm))   # termo de Greenwood
        se = S * np.sqrt(var_acc)
        S_lo = max(0.0, S - 1.96 * se); S_hi = min(1.0, S + 1.96 * se)
        km_rows.append(dict(categoria=int(k), mes=m, n_risco=int(n_risco),
                            eventos=int(dm), censuras=int(cm),
                            S=S, S_lo=S_lo, S_hi=S_hi))
        rmst += surv_por_mes[m - 1]          # área: faixa [m-1,m) vale S(m-1) (step pós)
        surv_por_mes[m] = S
        if mediana is None and S <= 0.5:
            mediana = m
        n = n_risco - dm - cm                # em risco no próximo mês
    resumo_rows.append(dict(
        categoria=int(k), n=N, eventos_total=int(d.sum()),
        S12=round(S, 5),
        risco_deslig_12m_KM=round(1 - S, 5),     # incidência acumulada KM em 12 meses
        mediana_meses=(mediana if mediana is not None else np.nan),  # NaN => > 12 (indef.)
        RMST12_meses=round(rmst, 4),
        meses_perdidos=round(H - rmst, 4),
    ))

km = pd.DataFrame(km_rows)
resumo = pd.DataFrame(resumo_rows).sort_values("categoria")
km.to_csv(f"{TAB}/sobrevivencia_km_2023.csv", index=False)
resumo.to_csv(f"{TAB}/sobrevivencia_resumo_2023.csv", index=False)
log(f"tabelas salvas -> {TAB}/sobrevivencia_km_2023.csv, sobrevivencia_resumo_2023.csv")

# ---------------------------------------------------------------------------
# Figura — 23 curvas S(t), gradiente de cor por risco; mediana marcada quando existir
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ks = sorted(km["categoria"].unique()); K = len(ks)
cmap = plt.get_cmap("RdYlGn_r"); norm = Normalize(vmin=min(ks), vmax=max(ks))
fig, ax = plt.subplots(figsize=(11, 6.2))
for k in ks:
    s = km[km["categoria"] == k].sort_values("mes")
    ax.plot(s["mes"], s["S"], color=cmap(norm(k)), lw=1.4, alpha=0.9,
            marker="o", markersize=3.2, markeredgecolor="white", markeredgewidth=0.4)
ax.set_xlim(0, H); ax.set_xticks(range(0, H + 1))
ax.set_ylim(top=1.001)
ax.set_xlabel("meses desde jan/2023"); ax.set_ylabel("S(t) = P(continuar empregado)")
ax.set_title("Sobrevivência do emprego por categoria de risco — holdout 2023 (janela observável de 12 meses)")
ax.grid(alpha=0.25)
sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=ax); cb.set_label("categoria de risco (1=mínimo ... %d=alto)" % max(ks))
fig.tight_layout()
TMPPNG = "/tmp/sobrevivencia_categorias_2023.png"
fig.savefig(TMPPNG, dpi=130); shutil.copy(TMPPNG, f"{FIG}/sobrevivencia_categorias_2023.png")
log(f"figura salva -> {FIG}/sobrevivencia_categorias_2023.png")

# ---------------------------------------------------------------------------
# Resumo no console
# ---------------------------------------------------------------------------
pd.set_option("display.width", 140, "display.max_rows", 50)
print("\n=== RESUMO POR CATEGORIA (2023, janela 12m) ===")
print(resumo.to_string(index=False))
ndef = int(resumo["mediana_meses"].notna().sum())
log(f"FIM. Mediana DEFINIDA (<=12m) em {ndef}/{K} categorias; "
    f"nas demais a mediana é > 12 meses (precisaria de follow-up plurianual).")
