"""KM MOB 0–12 da CATEGORIA 1 (modelo 2124) — público todo vs. SEM setor público.

Mesma metodologia do sobrevivencia_mob_2124.py (ref. 2021–2024 agregados):
  evento = motivo_unificado=="involuntario_sjc"; censura = ativo em 31/12 ou
  saída por outro motivo; relógio MOB = mes_deslig − mes_admissao + 1 (vigente: jan).
Filtro: categoria_risco == 1. Duas populações:
  - "geral":   todo mundo da categoria 1;
  - "privado": natureza_setor != '1' (exclui servidores/empregados do setor público).

Resumível por (ano, arquivo) — cache em outputs/tables/_surv_counts_cat1_pub_priv_2124.csv.

Saídas: outputs/tables/sobrevivencia_km_cat1_pub_priv_2124.csv
        outputs/figures/sobrevivencia_cat1_pub_priv_2124.png

Uso: MPLCONFIGDIR=/tmp/mpl /tmp/consig_venv/bin/python surv_cat1_publico_vs_privado_2124.py
"""
import os, glob, time
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc

BASE = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(BASE, "data/processed/predicoes_2124")
ANOS_REF = [2021, 2022, 2023, 2024]
TARGET = "involuntario_sjc"
H = 12
TAB = os.path.join(BASE, "outputs/tables"); FIG = os.path.join(BASE, "outputs/figures")
CACHE = os.path.join(TAB, "_surv_counts_cat1_pub_priv_2124.csv")
COLS = ["categoria_risco", "mes_admissao", "mes_deslig", "motivo_unificado", "natureza_setor"]

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# ---------------------------------------------------------------------------
# Passo 1 — contagens (pop, mob, evento) por arquivo, resumível
# ---------------------------------------------------------------------------
done = set()
if os.path.exists(CACHE):
    cache = pd.read_csv(CACHE)
    done = set(cache["arquivo"].unique())
    log(f"cache: {len(done)} arquivos já processados")

for a in ANOS_REF:
    for fp in sorted(glob.glob(f"{PRED}/ano={a}/*.parquet")):
        tag = f"{a}/{os.path.basename(fp)}"
        if tag in done:
            continue
        pf = pq.ParquetFile(fp)
        acc = {}
        n_tot = n_cat1 = 0
        for batch in pf.iter_batches(batch_size=3_000_000, columns=COLS):
            tb = batch
            n_tot += tb.num_rows
            # filtro categoria 1
            mask = pc.equal(tb.column("categoria_risco"), 1)
            d = tb.filter(mask).to_pandas()
            if len(d) == 0:
                continue
            n_cat1 += len(d)
            mes_a = d["mes_admissao"].to_numpy("int16")
            mes_d = d["mes_deslig"].to_numpy("int16")
            evento = (d["motivo_unificado"].to_numpy() == TARGET).astype("int8")
            e = np.where((mes_a >= 1) & (mes_a <= 12), mes_a, 1).astype("int16")
            m_cal = np.where((mes_d >= 1) & (mes_d <= 12), mes_d, 12).astype("int16")
            mob = np.clip(m_cal - e + 1, 1, H).astype("int16")
            # setor: normaliza p/ string sem padding ('1' = público)
            ns = d["natureza_setor"].astype(str).str.strip().str.lstrip("0")
            priv = (ns != "1").to_numpy()
            for pop, sel in (("geral", np.ones(len(d), bool)), ("privado", priv)):
                g = pd.DataFrame({"mob": mob[sel], "evento": evento[sel]}) \
                      .groupby(["mob", "evento"]).size()
                for key, n in g.items():
                    k = (pop,) + key
                    acc[k] = acc.get(k, 0) + int(n)
        rows = [{"arquivo": tag, "pop": k[0], "mob": k[1], "evento": k[2], "n": v}
                for k, v in acc.items()]
        pd.DataFrame(rows).to_csv(CACHE, mode="a", header=not os.path.exists(CACHE), index=False)
        log(f"  {tag}: {n_tot:,} linhas, {n_cat1:,} na cat 1")

counts = (pd.read_csv(CACHE).groupby(["pop", "mob", "evento"])["n"].sum()
            .unstack("evento", fill_value=0).reset_index()
            .rename(columns={0: "censuras", 1: "eventos"}))
for c in ("censuras", "eventos"):
    if c not in counts.columns: counts[c] = 0

# ---------------------------------------------------------------------------
# Passo 2 — KM (Greenwood) por população
# ---------------------------------------------------------------------------
log("Kaplan-Meier ...")
km_rows = []
for pop in ("geral", "privado"):
    sub = counts[counts["pop"] == pop].set_index("mob").reindex(range(1, H + 1), fill_value=0)
    d = sub["eventos"].to_numpy(float); c = sub["censuras"].to_numpy(float)
    N = int((d + c).sum()); n = N; S = 1.0; var_acc = 0.0
    km_rows.append(dict(pop=pop, mob=0, n_risco=N, eventos=0, censuras=0, S=1.0, S_lo=1.0, S_hi=1.0))
    for m in range(1, H + 1):
        n_risco = n; dm, cm = d[m - 1], c[m - 1]
        if n_risco > 0:
            S *= (n_risco - dm) / n_risco
            if n_risco - dm > 0:
                var_acc += dm / (n_risco * (n_risco - dm))
        se = S * np.sqrt(var_acc)
        km_rows.append(dict(pop=pop, mob=m, n_risco=int(n_risco), eventos=int(dm),
                            censuras=int(cm), S=S,
                            S_lo=max(0.0, S - 1.96 * se), S_hi=min(1.0, S + 1.96 * se)))
        n = n_risco - dm - cm
km = pd.DataFrame(km_rows)
km.to_csv(f"{TAB}/sobrevivencia_km_cat1_pub_priv_2124.csv", index=False)
log(f"KM salvo -> {TAB}/sobrevivencia_km_cat1_pub_priv_2124.csv")

# resumo no console
for pop in ("geral", "privado"):
    g = km[km["pop"] == pop]
    s12 = g[g.mob == H].S.iloc[0]; n0 = g[g.mob == 0].n_risco.iloc[0]
    log(f"  cat1/{pop}: N={n0:,}  S(12)={s12:.5f}  risco 12m={1-s12:.5%}")

# ---------------------------------------------------------------------------
# Passo 3 — figura comparativa
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import shutil

fig, ax = plt.subplots(figsize=(10, 6))
estilo = {"geral":   dict(color="#1f77b4", label="Geral (todo o público)"),
          "privado": dict(color="#d62728", label="Sem setor público (natureza_setor ≠ 1)")}
for pop in ("geral", "privado"):
    g = km[km["pop"] == pop].sort_values("mob")
    ax.plot(g.mob, g.S, marker="o", ms=4, lw=1.8, **estilo[pop])
    ax.fill_between(g.mob, g.S_lo, g.S_hi, color=estilo[pop]["color"], alpha=.15)
ax.set_xlabel("MOB — meses desde a entrada"); ax.set_ylabel("S(t) = P(seguir empregado)")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
ax.set_xticks(range(0, H + 1))
ax.set_title("Categoria 1 — KM MOB 0–12 (ref. 2021–2024): geral vs. sem setor público")
ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig("/tmp/skm_cat1.png", dpi=130)
shutil.copy("/tmp/skm_cat1.png", f"{FIG}/sobrevivencia_cat1_pub_priv_2124.png"); plt.close(fig)
log("figura salva")
print("FIM_SURV_CAT1", flush=True)
