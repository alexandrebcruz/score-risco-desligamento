"""Impõe monotonicidade (decrescente com a categoria de risco 1->23) às estatísticas
do tempo-até-desligamento (q1, mediana, média, q3) via ISOTONIC REGRESSION (PAVA).

As categorias estão ordenadas por risco crescente, então tempo deve ser NÃO-CRESCENTE.
Q1 e mediana já saem monotônicas; média e q3 têm inversões nas faixas com shape p<1
(cauda pesada do frailty). O PAVA "achata" só os trechos que violam a ordem, pela média
dos blocos (ponderada por n da categoria = nº de vínculos).

Entrada: outputs/tables/sobrevivencia_weibull_estatisticas_2023.csv  (+ params p/ pegar n)
Saída:   outputs/tables/sobrevivencia_weibull_estatisticas_mono_2023.csv
"""
import pandas as pd, numpy as np

S = "outputs/tables/sobrevivencia_weibull_estatisticas_2023.csv"
PARAMS = "outputs/tables/sobrevivencia_weibull_params_2023.csv"
RES = "outputs/tables/sobrevivencia_resumo_2023.csv"
OUT = "outputs/tables/sobrevivencia_weibull_estatisticas_mono_2023.csv"

s = pd.read_csv(S).sort_values("categoria").reset_index(drop=True)
n = pd.read_csv(RES).set_index("categoria")["n"]               # peso = nº de vínculos
w = s["categoria"].map(n).to_numpy(dtype=float)

def isotonic_decreasing(y, w):
    """Ajuste isotônico NÃO-CRESCENTE: nega, roda não-decrescente, nega de volta."""
    y = np.asarray(y, float); w = np.asarray(w, float)
    neg = -y
    # PAVA não-decrescente guardando o nº de itens por bloco
    val, cnt, wsum = [], [], []
    for yi, wi in zip(neg, w):
        val.append(yi); cnt.append(1); wsum.append(wi)
        while len(val) > 1 and val[-2] > val[-1]:
            nw = wsum[-2] + wsum[-1]
            nv = (val[-2] * wsum[-2] + val[-1] * wsum[-1]) / nw
            nc = cnt[-2] + cnt[-1]
            val[-2:] = [nv]; wsum[-2:] = [nw]; cnt[-2:] = [nc]
    fitted = []
    for v, c in zip(val, cnt):
        fitted += [v] * c
    return -np.asarray(fitted)

cols = ["q1_meses", "mediana_meses", "media_meses", "q3_meses"]
for c in cols:
    s[c + "_mono"] = np.round(isotonic_decreasing(s[c].to_numpy(), w), 1)

# verificação
print("=== checagem de monotonicidade (deve decrescer com a categoria) ===")
for c in cols:
    v = s[c + "_mono"].to_numpy()
    bad = [(int(s.categoria[i]), int(s.categoria[i+1])) for i in range(len(v)-1) if v[i+1] > v[i] + 1e-9]
    print(f"  {c+'_mono'}: {'OK' if not bad else 'VIOLA '+str(bad)}")

# coerência interna Q1<=mediana<=Q3 após monotonizar
viol = s[(s.q1_meses_mono > s.mediana_meses_mono + 1e-9) |
         (s.mediana_meses_mono > s.q3_meses_mono + 1e-9)]
print(f"  coerência Q1<=mediana<=Q3: {'OK' if viol.empty else 'VIOLA em cats '+str(viol.categoria.tolist())}")

out = s[["categoria", "shape_p", "escala_lambda_meses",
         "q1_meses", "q1_meses_mono", "mediana_meses", "mediana_meses_mono",
         "media_meses", "media_meses_mono", "q3_meses", "q3_meses_mono"]]
out.to_csv(OUT, index=False)

pd.set_option("display.width", 220, "display.max_rows", 30)
print("\n=== ESTATÍSTICAS (bruto -> monotonizado por isotonic regression, meses) ===")
show = s[["categoria", "q1_meses", "q1_meses_mono", "mediana_meses", "mediana_meses_mono",
          "media_meses", "media_meses_mono", "q3_meses", "q3_meses_mono"]]
print(show.to_string(index=False))
print(f"\nsalvo em {OUT}")
