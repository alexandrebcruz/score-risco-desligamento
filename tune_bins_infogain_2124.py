"""Recategorização do risco para o ENSEMBLE NOVO (retreino 2021–2024).

Como o tune_bins_infogain.py (PD maximizando I(bin;y) sobre micro-bins por quantil),
com referência nos predicts de 2021–2024 agrupados, MAS com critério de K mais
exigente: além do y médio por categoria ser estritamente crescente no AGREGADO,
ele deve ser estritamente crescente DENTRO DE CADA ANO individualmente (estabilidade
da ordenação entre safras). K* = maior K que satisfaz ambos antes da 1ª quebra.

Uso: /tmp/consig_venv/bin/python tune_bins_infogain_2124.py [N_MICRO] [K_MAX]
Saídas: outputs/tables/binning_infogain_sweep_2124.csv
        outputs/tables/binning_infogain_escolhido_2124.csv
"""
import sys, os, glob
import numpy as np
import pandas as pd

PRED = "data/processed/predicoes_2124"
ANOS_REF = [2021, 2022, 2023, 2024]
N_MICRO = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
K_MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 100
os.makedirs("outputs/tables", exist_ok=True)


def binary_entropy_bits(p):
    p = np.asarray(p, dtype="float64")
    out = np.zeros_like(p)
    m = (p > 0) & (p < 1)
    pp = p[m]
    out[m] = -(pp * np.log2(pp) + (1 - pp) * np.log2(1 - pp))
    return out


def main():
    # ---- carrega prob/y POR ANO (p/ o critério ano-a-ano) ----
    probs, ys = {}, {}
    for a in ANOS_REF:
        ps, yy = [], []
        for f in sorted(glob.glob(f"{PRED}/ano={a}/*.parquet")):
            d = pd.read_parquet(f, columns=["prob_desligamento", "y"])
            ps.append(d["prob_desligamento"].to_numpy("float32"))
            yy.append(d["y"].to_numpy("int8"))
        probs[a] = np.concatenate(ps); ys[a] = np.concatenate(yy)
        print(f"  {a}: {len(probs[a]):,} (taxa {ys[a].mean():.4f})", flush=True)
    N = sum(len(v) for v in probs.values())
    k_tot = sum(int(v.sum()) for v in ys.values())
    p_global = k_tot / N
    H_y = float(binary_entropy_bits(np.array([p_global]))[0])
    print(f"POOLED {ANOS_REF}: N={N:,} | taxa={p_global:.4f} | H(y)={H_y:.5f} bits", flush=True)

    # ---- micro-bins por quantil do POOLED ----
    allp = np.concatenate([probs[a] for a in ANOS_REF]).astype("float64")
    qs = np.linspace(0, 1, N_MICRO + 1)
    edges = np.unique(np.quantile(allp, qs))
    edges[0] = min(edges[0], allp.min()); edges[-1] = max(edges[-1], allp.max()) + 1e-12
    del allp
    M0 = len(edges) - 1

    # agrega n/k por micro-bin POR ANO (e pooled = soma)
    nY, kY = {}, {}
    for a in ANOS_REF:
        idx = np.clip(np.digitize(probs[a].astype("float64"), edges[1:-1], right=False), 0, M0 - 1)
        nY[a] = np.bincount(idx, minlength=M0).astype("float64")
        kY[a] = np.bincount(idx, weights=ys[a].astype("float64"), minlength=M0)
        del idx
    del probs, ys
    n_i = sum(nY.values()); k_i = sum(kY.values())
    keep = n_i > 0
    n_i, k_i = n_i[keep], k_i[keep]
    for a in ANOS_REF:
        nY[a], kY[a] = nY[a][keep], kY[a][keep]
    lo_edges = edges[:-1][keep]; hi_edges = edges[1:][keep]
    M = len(n_i)
    print(f"{M} micro-bins efetivos", flush=True)

    # prefixos pooled (DP) e por ano (checagem)
    Cn = np.concatenate([[0.0], np.cumsum(n_i)])
    Ck = np.concatenate([[0.0], np.cumsum(k_i)])
    CnY = {a: np.concatenate([[0.0], np.cumsum(nY[a])]) for a in ANOS_REF}
    CkY = {a: np.concatenate([[0.0], np.cumsum(kY[a])]) for a in ANOS_REF}

    nn = Cn[None, :] - Cn[:, None]
    kk = Ck[None, :] - Ck[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(nn > 0, kk / nn, 0.0)
    COST = nn * binary_entropy_bits(p)
    COST[np.tril_indices(M + 1)] = np.inf
    COST[~np.isfinite(COST)] = np.inf

    DP = np.full((K_MAX + 1, M + 1), np.inf)
    PAR = np.full((K_MAX + 1, M + 1), -1, dtype=int)
    DP[0, 0] = 0.0
    for k in range(1, K_MAX + 1):
        cand = DP[k - 1][:, None] + COST
        PAR[k] = np.argmin(cand, axis=0)
        DP[k] = cand[PAR[k], np.arange(M + 1)]

    def boundaries(K):
        b = [M]; j = M
        for k in range(K, 0, -1):
            a = PAR[k, j]; b.append(a); j = a
        return list(reversed(b))

    def seg_rates(cuts, Cn_, Ck_):
        K = len(cuts) - 1
        sn = np.array([Cn_[cuts[i + 1]] - Cn_[cuts[i]] for i in range(K)])
        sk = np.array([Ck_[cuts[i + 1]] - Ck_[cuts[i]] for i in range(K)])
        with np.errstate(divide="ignore", invalid="ignore"):
            return sn, np.where(sn > 0, sk / sn, np.nan)

    rows = []; detalhe_por_K = {}
    for K in range(1, K_MAX + 1):
        if not np.isfinite(DP[K, M]):
            continue
        cuts = boundaries(K)
        seg_n, seg_taxa = seg_rates(cuts, Cn, Ck)
        IG = H_y - DP[K, M] / N
        mono_pool = bool(np.all(np.diff(seg_taxa) > 0))
        anos_quebra = []
        for a in ANOS_REF:
            sn_a, tx_a = seg_rates(cuts, CnY[a], CkY[a])
            # categoria vazia num ano OU ordem não-estritamente-crescente -> quebra
            if np.any(sn_a <= 0) or not np.all(np.diff(tx_a) > 0):
                anos_quebra.append(a)
        mono_anos = not anos_quebra
        rows.append({"K_categorias": K, "n_cortes": K - 1, "IG_bits": IG,
                     "IG_frac_%": 100 * IG / H_y,
                     "monotonico_pooled": mono_pool,
                     "monotonico_por_ano": mono_anos,
                     "anos_quebra": ";".join(map(str, anos_quebra)),
                     "min_n_categoria": int(seg_n.min())})
        detalhe_por_K[K] = cuts

    sweep = pd.DataFrame(rows)
    sweep.to_csv("outputs/tables/binning_infogain_sweep_2124.csv", index=False)

    # ---- K*: maior K com pooled E por-ano monotônicos ANTES da 1ª quebra ----
    primeiro_quebra = None
    for r in rows:
        if not (r["monotonico_pooled"] and r["monotonico_por_ano"]):
            primeiro_quebra = r["K_categorias"]; break
    K_star = (primeiro_quebra - 1) if primeiro_quebra else max(r["K_categorias"] for r in rows)

    print("\n=== Varredura ===", flush=True)
    print(sweep.to_string(index=False,
          formatters={"IG_bits": "{:.5f}".format, "IG_frac_%": "{:.2f}".format}), flush=True)
    if primeiro_quebra:
        rq = [r for r in rows if r["K_categorias"] == primeiro_quebra][0]
        print(f"\n1ª QUEBRA em K={primeiro_quebra} (anos: {rq['anos_quebra'] or 'pooled'}).", flush=True)
    print(f">>> K*={K_star} (critério: ordenação estrita no pooled E em CADA ano 2021–2024) <<<", flush=True)

    # ---- detalhe do K escolhido (pooled + taxa por ano p/ auditoria) ----
    cuts = detalhe_por_K[K_star]
    seg_n, seg_taxa = seg_rates(cuts, Cn, Ck)
    det = pd.DataFrame({
        "categoria": range(1, K_star + 1),
        "prob_min": [lo_edges[cuts[i]] for i in range(K_star)],
        "prob_max": [hi_edges[cuts[i + 1] - 1] for i in range(K_star)],
        "n": seg_n.astype("int64"),
        "taxa_y": seg_taxa,
        "lift_vs_global": seg_taxa / p_global,
    })
    for a in ANOS_REF:
        _, tx_a = seg_rates(cuts, CnY[a], CkY[a])
        det[f"taxa_{a}"] = tx_a
    det.to_csv("outputs/tables/binning_infogain_escolhido_2124.csv", index=False)
    print(f"\n=== Categorias (K={K_star}) — taxa pooled e por ano ===", flush=True)
    fmt = {c: "{:.4f}".format for c in det.columns if c.startswith(("prob", "taxa"))}
    fmt["lift_vs_global"] = "{:.2f}".format
    print(det.to_string(index=False, formatters=fmt), flush=True)
    print("FIM_TUNE_2124", flush=True)


if __name__ == "__main__":
    main()
