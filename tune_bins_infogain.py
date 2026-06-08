"""Discretização ÓTIMA de prob_desligamento maximizando o ganho de informação de y.

Usa apenas as colunas `prob_desligamento` e `y` do parquet de predições 2023.

Para cada número de categorias K, encontra os pontos de corte que MAXIMIZAM a
informação mútua I(bin; y) (= ganho de informação, em bits) via PROGRAMAÇÃO DINÂMICA
sobre micro-bins por quantil. Depois varre K crescente e detecta o ponto em que o
`y` médio por categoria deixa de ser estritamente crescente (ordenação quebra).

Objetivo: o MAIOR K que maximiza o ganho de informação MANTENDO a ordenação do
y médio por categoria.

Uso: /tmp/consig_venv/bin/python tune_bins_infogain.py [N_MICRO] [K_MAX]
Saídas: outputs/tables/binning_infogain_sweep.csv  (varredura por K)
        outputs/tables/binning_infogain_escolhido.csv (detalhe do K escolhido)
"""
import sys, os
import numpy as np
import pandas as pd

PARQUET = "outputs/predicoes_2023_ensemble_base.parquet"
N_MICRO = int(sys.argv[1]) if len(sys.argv) > 1 else 1000   # micro-bins por quantil
K_MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 40        # nº máx de categorias a testar
os.makedirs("outputs/tables", exist_ok=True)


def binary_entropy_bits(p):
    """H(p) em bits, com H(0)=H(1)=0."""
    p = np.asarray(p, dtype="float64")
    out = np.zeros_like(p)
    m = (p > 0) & (p < 1)
    pp = p[m]
    out[m] = -(pp * np.log2(pp) + (1 - pp) * np.log2(1 - pp))
    return out


def main():
    print(f"lendo {PARQUET} (apenas prob_desligamento e y) ...", flush=True)
    df = pd.read_parquet(PARQUET, columns=["prob_desligamento", "y"])
    prob = df["prob_desligamento"].to_numpy("float64")
    y = df["y"].to_numpy("float64")
    N = len(y)
    p_global = y.mean()
    H_y = float(binary_entropy_bits(np.array([p_global]))[0])
    print(f"N={N:,} | taxa global y={p_global:.4f} | H(y)={H_y:.5f} bits", flush=True)

    # --- micro-bins por quantil (cada um ~N/N_MICRO linhas), agregando n e positivos ---
    qs = np.linspace(0, 1, N_MICRO + 1)
    edges = np.unique(np.quantile(prob, qs))          # bordas dedup
    # garante que cobre tudo
    edges[0] = min(edges[0], prob.min()); edges[-1] = max(edges[-1], prob.max()) + 1e-12
    idx = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, len(edges) - 2)
    M = len(edges) - 1
    n_i = np.bincount(idx, minlength=M).astype("float64")
    k_i = np.bincount(idx, weights=y, minlength=M).astype("float64")
    # remove micro-bins vazios (mantendo ordem)
    keep = n_i > 0
    n_i, k_i = n_i[keep], k_i[keep]
    lo_edges = edges[:-1][keep]          # prob mínima de cada micro-bin
    hi_edges = edges[1:][keep]           # prob máxima de cada micro-bin
    M = len(n_i)
    print(f"{M} micro-bins efetivos (de {N_MICRO} pedidos)", flush=True)

    # prefixos
    Cn = np.concatenate([[0.0], np.cumsum(n_i)])      # len M+1
    Ck = np.concatenate([[0.0], np.cumsum(k_i)])

    # --- matriz de custo COST[a,j] = entropia ponderada do segmento de micro-bins [a, j) ---
    # custo = n_seg * H(p_seg) ; minimizar soma = maximizar informação mútua
    nn = Cn[None, :] - Cn[:, None]                    # (M+1, M+1)
    kk = Ck[None, :] - Ck[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(nn > 0, kk / nn, 0.0)
    COST = nn * binary_entropy_bits(p)
    COST[np.tril_indices(M + 1)] = np.inf             # exige j > a (segmento não-vazio)
    COST[~np.isfinite(COST)] = np.inf

    # --- DP: DP[k][j] = menor custo p/ cobrir [0,j) com k segmentos ---
    INF = np.inf
    DP = np.full((K_MAX + 1, M + 1), INF)
    PAR = np.full((K_MAX + 1, M + 1), -1, dtype=int)
    DP[0, 0] = 0.0
    for k in range(1, K_MAX + 1):
        cand = DP[k - 1][:, None] + COST              # (M+1, M+1): a -> j
        PAR[k] = np.argmin(cand, axis=0)
        DP[k] = cand[PAR[k], np.arange(M + 1)]

    def boundaries(K):
        """recupera as bordas (índices de micro-bin) da melhor partição em K segmentos."""
        b = [M]; j = M
        for k in range(K, 0, -1):
            a = PAR[k, j]; b.append(a); j = a
        return list(reversed(b))                      # [0, c1, ..., M]

    # --- varredura por K ---
    rows = []
    detalhe_por_K = {}
    for K in range(1, K_MAX + 1):
        if not np.isfinite(DP[K, M]):
            continue
        cuts = boundaries(K)
        seg_n = np.array([Cn[cuts[i + 1]] - Cn[cuts[i]] for i in range(K)])
        seg_k = np.array([Ck[cuts[i + 1]] - Ck[cuts[i]] for i in range(K)])
        seg_taxa = seg_k / seg_n
        cond_H = DP[K, M] / N                          # H(y|bins) em bits
        IG = H_y - cond_H                              # ganho de informação (bits)
        # monotonicidade estritamente crescente do y médio por categoria
        mono = bool(np.all(np.diff(seg_taxa) > 0))
        rows.append({"K_categorias": K, "n_cortes": K - 1, "IG_bits": IG,
                     "IG_frac_%": 100 * IG / H_y, "monotonico": mono,
                     "min_n_categoria": int(seg_n.min())})
        detalhe_por_K[K] = (cuts, seg_n, seg_k, seg_taxa)

    sweep = pd.DataFrame(rows)
    sweep.to_csv("outputs/tables/binning_infogain_sweep.csv", index=False)

    # --- escolhe o MAIOR K monotônico ANTES da 1ª quebra de ordenação ---
    primeiro_quebra = None
    for r in rows:
        if not r["monotonico"]:
            primeiro_quebra = r["K_categorias"]; break
    if primeiro_quebra is None:
        K_star = max(r["K_categorias"] for r in rows)
    else:
        K_star = primeiro_quebra - 1

    print("\n=== Varredura (K categorias) ===", flush=True)
    print(sweep.to_string(index=False,
          formatters={"IG_bits": "{:.5f}".format, "IG_frac_%": "{:.2f}".format}), flush=True)
    if primeiro_quebra:
        print(f"\nOrdenação do y médio QUEBRA em K={primeiro_quebra} categorias.", flush=True)
    print(f">>> MAIOR K que maximiza o ganho de informação MANTENDO a ordenação: "
          f"K={K_star} categorias ({K_star-1} cortes) <<<", flush=True)

    # --- detalhe do K escolhido ---
    cuts, seg_n, seg_k, seg_taxa = detalhe_por_K[K_star]
    det = pd.DataFrame({
        "categoria": range(1, K_star + 1),
        "prob_min": [lo_edges[cuts[i]] for i in range(K_star)],
        "prob_max": [hi_edges[cuts[i + 1] - 1] for i in range(K_star)],
        "n": seg_n.astype("int64"),
        "positivos": seg_k.astype("int64"),
        "taxa_y": seg_taxa,
        "lift_vs_global": seg_taxa / p_global,
    })
    det.to_csv("outputs/tables/binning_infogain_escolhido.csv", index=False)
    print(f"\n=== Categorias do K escolhido (K={K_star}) ===", flush=True)
    print(det.to_string(index=False,
          formatters={"prob_min": "{:.4f}".format, "prob_max": "{:.4f}".format,
                      "taxa_y": "{:.4f}".format, "lift_vs_global": "{:.2f}".format}), flush=True)
    print("\nsalvos: outputs/tables/binning_infogain_sweep.csv e binning_infogain_escolhido.csv", flush=True)


if __name__ == "__main__":
    main()
