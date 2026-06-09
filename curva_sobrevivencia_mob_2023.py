"""Curva de sobrevivência por MOB (months on book) — variante APARTADA para comparar
com a abordagem atual (relógio = mês do calendário).

Diferença de relógio:
- abordagem ATUAL (sobrevivencia_km_2023.csv): t = mês do calendário; TODOS são tratados
  como em risco desde janeiro.
- abordagem MOB (este script): t = meses desde a ENTRADA na janela de observação.
  * vínculo já vigente no início (mes_admissao==0): entra em janeiro -> MOB idêntico ao atual.
  * vínculo criado no ano (mes_admissao=a, 1..12): entra no mês a -> MOB = mês_cal − a + 1.
  Corrige a exposição (não conta os meses em que o vínculo nem existia). KM padrão de
  censura à direita (todos entram em MOB 0; entrantes tardios têm follow-up mais curto).

NÃO sobrescreve nada da abordagem atual — salva tudo com sufixo _mob + arquivos de comparação.

Uso: MPLCONFIGDIR=/tmp/mpl /tmp/consig_venv/bin/python curva_sobrevivencia_mob_2023.py
Saídas:
  outputs/tables/sobrevivencia_km_mob_2023.csv          (longo: categoria x MOB -> n,S,IC)
  outputs/tables/sobrevivencia_resumo_mob_2023.csv      (1 linha/categoria: RMST, mediana, %novo)
  outputs/tables/sobrevivencia_comparacao_atual_vs_mob_2023.csv  (atual vs MOB por categoria)
  outputs/figures/sobrevivencia_mob_vs_atual_2023.png
"""
import os, glob, time, shutil
import numpy as np, pandas as pd
import pyarrow.parquet as pq

INTERIM = "data/interim/rais/ano=2023"
CATPARQ = "outputs/predicoes_2023_ensemble_base_categorizado.parquet"
TARGET = "involuntario_sjc"
H = 12
TAB = "outputs/tables"; FIG = "outputs/figures"
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
COUNTS = f"{TAB}/_surv_counts_mob_2023.csv"
MIX = f"{TAB}/_surv_mob_mix_2023.csv"

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# ---------------------------------------------------------------------------
# Passo A — estatística suficiente por (categoria, MOB) + mix pré-existente/novo
# ---------------------------------------------------------------------------
if not (os.path.exists(COUNTS) and os.path.exists(MIX)):
    log("Passo A: lendo evento/mes_deslig/mes_admissao do interim + categoria do categorizado ...")
    ev_parts, md_parts, ma_parts = [], [], []
    for fp in sorted(glob.glob(f"{INTERIM}/*.parquet")):
        pf = pq.ParquetFile(fp)
        for rb in pf.iter_batches(batch_size=4_000_000,
                                  columns=["motivo_unificado", "mes_deslig", "mes_admissao"]):
            mot = rb.column("motivo_unificado").to_numpy(zero_copy_only=False)
            ev_parts.append((mot == TARGET).astype("int8"))
            md_parts.append(rb.column("mes_deslig").to_numpy(zero_copy_only=False).astype("int16"))
            ma_parts.append(rb.column("mes_admissao").to_numpy(zero_copy_only=False).astype("int16"))
        log(f"  interim lido: {os.path.basename(fp)}")
    evento = np.concatenate(ev_parts); mes_d = np.concatenate(md_parts); mes_a = np.concatenate(ma_parts)
    del ev_parts, md_parts, ma_parts

    cat_parts, y_parts = [], []
    for rb in pq.ParquetFile(CATPARQ).iter_batches(batch_size=4_000_000, columns=["categoria_risco", "y"]):
        cat_parts.append(rb.column("categoria_risco").to_numpy(zero_copy_only=False).astype("int16"))
        y_parts.append(rb.column("y").to_numpy(zero_copy_only=False).astype("int8"))
    categoria = np.concatenate(cat_parts); y = np.concatenate(y_parts)
    del cat_parts, y_parts

    assert len(evento) == len(categoria) == len(y), "tamanhos divergentes — ordem quebrada"
    mism = int((evento != y).sum())
    if mism:
        raise SystemExit(f"ALINHAMENTO FALHOU: {mism:,} linhas com evento!=y. Abortando.")
    log(f"alinhamento OK: {len(y):,} linhas, evento==y em 100%")

    # mês de entrada na janela: pré-existente (0) -> janeiro(1); novo(a) -> a
    e = np.where((mes_a >= 1) & (mes_a <= 12), mes_a, 1).astype("int16")
    # mês-calendário de saída/censura: ativo (mes_d==0) -> dezembro(12); senão mes_d
    m_cal = np.where((mes_d >= 1) & (mes_d <= 12), mes_d, 12).astype("int16")
    mob = (m_cal - e + 1).astype("int32")
    anom = int((mob < 1).sum())                       # desligamento antes da admissão (inconsistência)
    if anom:
        log(f"AVISO: {anom:,} linhas com MOB<1 (mes_deslig<mes_admissao) -> clip em 1")
    mob = np.clip(mob, 1, H).astype("int16")

    df = pd.DataFrame({"categoria": categoria, "mob": mob, "evento": evento})
    g = (df.groupby(["categoria", "mob", "evento"]).size().unstack("evento", fill_value=0).reset_index())
    g = g.rename(columns={0: "censuras", 1: "eventos"})
    for c in ("censuras", "eventos"):
        if c not in g.columns: g[c] = 0
    g = g[["categoria", "mob", "eventos", "censuras"]].sort_values(["categoria", "mob"])
    g.to_csv(COUNTS, index=False)

    novo = ((mes_a >= 1) & (mes_a <= 12)).astype("int8")
    mix = pd.DataFrame({"categoria": categoria, "novo": novo}).groupby("categoria")["novo"].agg(
        n="count", n_novo="sum").reset_index()
    mix["n_preexist"] = mix["n"] - mix["n_novo"]
    mix["pct_novo"] = (100 * mix["n_novo"] / mix["n"]).round(2)
    mix.to_csv(MIX, index=False)
    log(f"Passo A ok -> {COUNTS} ({len(g)} linhas) + {MIX}")
else:
    log(f"Passo A: usando cache {COUNTS} + {MIX}")

counts = pd.read_csv(COUNTS); mix = pd.read_csv(MIX).set_index("categoria")

# ---------------------------------------------------------------------------
# Passo B — KM por categoria no relógio MOB (idêntico à maquinaria atual)
# ---------------------------------------------------------------------------
log("Passo B: KM por categoria (relógio MOB) ...")
km_rows, resumo_rows = [], []
for k in sorted(counts["categoria"].unique()):
    sub = counts[counts["categoria"] == k].set_index("mob").reindex(range(1, H + 1), fill_value=0)
    d = sub["eventos"].to_numpy(float); c = sub["censuras"].to_numpy(float)
    N = int((d + c).sum()); n = N; S = 1.0; var_acc = 0.0
    km_rows.append(dict(categoria=int(k), mob=0, n_risco=N, eventos=0, censuras=0, S=1.0, S_lo=1.0, S_hi=1.0))
    rmst = 0.0; mediana = None; surv = {0: 1.0}
    for m in range(1, H + 1):
        n_risco = n; dm, cm = d[m - 1], c[m - 1]
        if n_risco > 0:
            S *= (n_risco - dm) / n_risco
            if n_risco - dm > 0: var_acc += dm / (n_risco * (n_risco - dm))
        se = S * np.sqrt(var_acc)
        km_rows.append(dict(categoria=int(k), mob=m, n_risco=int(n_risco), eventos=int(dm),
                            censuras=int(cm), S=S, S_lo=max(0.0, S - 1.96 * se), S_hi=min(1.0, S + 1.96 * se)))
        rmst += surv[m - 1]; surv[m] = S
        if mediana is None and S <= 0.5: mediana = m
        n = n_risco - dm - cm
    resumo_rows.append(dict(categoria=int(k), n=N, eventos_total=int(d.sum()), S12=round(S, 5),
                            risco_deslig_12m_KM=round(1 - S, 5),
                            mediana_meses=(mediana if mediana is not None else np.nan),
                            RMST12_meses=round(rmst, 4), meses_perdidos=round(H - rmst, 4),
                            pct_novo=float(mix.loc[k, "pct_novo"])))

km = pd.DataFrame(km_rows)
resumo = pd.DataFrame(resumo_rows).sort_values("categoria")
km.to_csv(f"{TAB}/sobrevivencia_km_mob_2023.csv", index=False)
resumo.to_csv(f"{TAB}/sobrevivencia_resumo_mob_2023.csv", index=False)
log("tabelas MOB salvas (sobrevivencia_km_mob_2023.csv, sobrevivencia_resumo_mob_2023.csv)")

# ---------------------------------------------------------------------------
# Comparação com a abordagem atual
# ---------------------------------------------------------------------------
km_at = pd.read_csv(f"{TAB}/sobrevivencia_km_2023.csv").rename(columns={"mes": "t", "S": "S_atual"})
km_mob = km.rename(columns={"mob": "t", "S": "S_mob"})
comp = (km_at[["categoria", "t", "S_atual"]].merge(km_mob[["categoria", "t", "S_mob"]], on=["categoria", "t"]))
comp["delta"] = (comp["S_mob"] - comp["S_atual"]).round(5)
comp.to_csv(f"{TAB}/sobrevivencia_comparacao_atual_vs_mob_2023.csv", index=False)

res_at = pd.read_csv(f"{TAB}/sobrevivencia_resumo_2023.csv").set_index("categoria")
cmp = resumo.set_index("categoria").join(res_at[["S12", "RMST12_meses", "mediana_meses"]],
                                         rsuffix="_atual")
cmp = cmp.rename(columns={"S12": "S12_mob", "RMST12_meses": "RMST_mob", "mediana_meses": "mediana_mob"})
cmp["dS12"] = (cmp["S12_mob"] - cmp["S12_atual"]).round(4)
cmp["dRMST"] = (cmp["RMST_mob"] - cmp["RMST12_meses_atual"]).round(3)

# ---------------------------------------------------------------------------
# Figura comparativa: (A) curvas atual(sólido) vs MOB(tracejado); (B) %novo vs ΔS(12)
# ---------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
ks = sorted(km["categoria"].unique())
cmap = plt.get_cmap("RdYlGn_r"); norm = Normalize(vmin=min(ks), vmax=max(ks))
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6))
for k in ks:
    col = cmap(norm(k))
    a = km_at[km_at.categoria == k].sort_values("t")
    m = km_mob[km_mob.categoria == k].sort_values("t")
    axA.plot(a["t"], a["S_atual"], color=col, lw=1.2, alpha=0.8)
    axA.plot(m["t"], m["S_mob"], color=col, lw=1.2, ls="--", alpha=0.85)
axA.set_xlim(0, H); axA.set_xticks(range(0, H + 1)); axA.set_ylim(top=1.001)
axA.set_xlabel("t (meses)"); axA.set_ylabel("S(t)")
axA.set_title("Sólido = atual (calendário) · tracejado = MOB (desde a entrada)")
axA.grid(alpha=0.25)
sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
fig.colorbar(sm, ax=axA, label="categoria de risco")
axB.scatter(cmp["pct_novo"], cmp["dS12"] * 100, c=[cmap(norm(k)) for k in cmp.index], s=60,
            edgecolors="#333", linewidths=0.5)
for k in cmp.index:
    axB.annotate(str(k), (cmp.loc[k, "pct_novo"], cmp.loc[k, "dS12"] * 100), fontsize=7,
                 ha="center", va="center")
axB.axhline(0, color="#888", lw=0.8)
axB.set_xlabel("% de vínculos novos no ano (admitidos 1-12)")
axB.set_ylabel("ΔS(12) = S_MOB − S_atual  (pontos %)")
axB.set_title("Quanto mais vínculos novos, maior o deslocamento da curva")
axB.grid(alpha=0.25)
fig.tight_layout()
TMP = "/tmp/sobrevivencia_mob_vs_atual_2023.png"
fig.savefig(TMP, dpi=130); shutil.copy(TMP, f"{FIG}/sobrevivencia_mob_vs_atual_2023.png")
log(f"figura comparativa salva -> {FIG}/sobrevivencia_mob_vs_atual_2023.png")

# ---------------------------------------------------------------------------
# Resumo comparativo no console
# ---------------------------------------------------------------------------
pd.set_option("display.width", 200, "display.max_rows", 30)
show = cmp.reset_index()[["categoria", "pct_novo", "S12_atual", "S12_mob", "dS12",
                          "RMST12_meses_atual", "RMST_mob", "dRMST", "mediana_meses_atual", "mediana_mob"]]
show = show.rename(columns={"RMST12_meses_atual": "RMST_atual", "mediana_meses_atual": "mediana_atual"})
print("\n=== COMPARAÇÃO ATUAL vs MOB (por categoria) ===")
print(show.round(4).to_string(index=False))
log(f"FIM. ΔS(12) médio = {cmp['dS12'].mean():+.4f} | ΔRMST médio = {cmp['dRMST'].mean():+.3f} meses")
