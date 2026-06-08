# Relatório — Modelagem do risco de desligamento (holdout 2023)

Consolida o bug de harmonização de códigos descoberto, as correções aplicadas, e a
comparação de **todos os experimentos** avaliados no holdout out-of-time de **2023**
(82.964.122 vínculos da RAIS nacional).

---

## 1. O bug: códigos com formato inconsistente entre anos

Os microdados da RAIS mudaram o **formato de alguns códigos** entre 2022 e 2023.
Como o modelo treina em ≤2022 e é avaliado em 2023, isso quebrava silenciosamente
features importantes no holdout:

| feature | 2019–2022 | 2023 | efeito em 2023 |
|---|---|---|---|
| `faixa_remuneracao` | `'02','03',…` (zero‑padded) | `'2','3',…` | categoria do treino **inexistente** no holdout → feature inútil |
| `faixa_horas` | `'02','03',…` | `'2','3',…` | idem |
| `causa_afastamento` | `'99'` (≈83% — "sem afastamento") | `'999'` (≈84%) | **categoria majoritária** vira desconhecida |
| `cbo` / `cnae` | `'010105'` (zfill) | `'10105'` | quebra apenas códigos iniciados em 0 (grupo militar) |

`tipo_vinculo` foi verificado e **não precisou de ajuste** (códigos naturais de 2 dígitos,
estáveis; a única novidade em 2023 é a classe `'999'`, com 0,002% — irrelevante).

### Correção (`src/cleaning.normalize_short_codes`)
- **remap de conteúdo**: `causa_afastamento` `'999'→'99'`;
- **strip de zero‑padding**: `faixa_remuneracao`, `faixa_horas`, `causa_afastamento`;
- **zfill consistente** de `cbo`(6)/`cnae`(7) antes de derivar os níveis hierárquicos.

Aplicada em todos os pontos de leitura: `cleaning.clean_rais_real`, `lags.build_lag_aggs`,
`cells.add_cell_keys`/`person_keys` (taxas + scoring), e nos scripts de treino do pod.

---

## 2. Métricas no holdout 2023 — todos os experimentos

Ver `outputs/tables/metricas_2023_consolidado.csv`.

### 🔴 Antes da correção (bug de formato ativo)
| experimento | AUC | Brier | LogLoss |
|---|---|---|---|
| Células c/ 2023 *(VAZADO, referência)* | 0,738 | 0,104 | 0,346 |
| CatBoost full‑data (4 anos) | 0,708 | 0,107 | 0,358 |
| CatBoost amostra | 0,701 | 0,108 | 0,361 |
| Score de células (treino‑only, honesto) | 0,685 | 0,111 | 0,372 |
| Ensemble base — Modelo A | 0,675 | 0,111 | 0,368 |
| Ensemble base (A+B) | 0,642 | 0,123 | 0,403 |
| Ensemble +lags (A+B) | 0,654 | — | — |
| Ensemble base — Modelo B *(valida no passado)* | 0,611 | 0,169 | 0,546 |

### 🟢 Depois da correção
| experimento | AUC | Brier | LogLoss |
|---|---|---|---|
| Ensemble base — Modelo B | 0,7436 | 0,105 | 0,3523 |
| **★ Ensemble base (A+B) — MELHOR** | **0,7407** | **0,104** | **0,3477** |
| Ensemble base — Modelo A | 0,7256 | 0,107 | 0,3545 |
| Ensemble +lags — Modelo B | 0,7124 | 0,107 | 0,3566 |
| Ensemble +lags (A+B) | 0,7155 | 0,106 | 0,3549 |
| Ensemble +lags — Modelo A | 0,7095 | 0,106 | 0,3566 |

> Setup do ensemble: **Modelo A** treina em 2019‑20 (val 2021‑22), **Modelo B** em
> 2021‑22 (val 2019‑20); ensemble = média das probabilidades. Métrica de treino e de
> early stopping = **Logloss**. Holdout = 2023 (nunca visto no fit/early stopping).

### 🔵 Variante +lags com early stopping ~1000 iter (árvores rasas)

Experimento dedicado a investigar se o early stopping precoce (iter 70/95) da variante
+lags era a causa do baixo desempenho. Objetivo: forçar o early stopping dos dois modelos
do ensemble para **≥ 1000 iterações**, ajustando `depth` e `learning_rate` **dentro das
faixas recomendadas do CatBoost** (`lr` ∈ [0,01; 0,3]; `depth` ∈ [4; 10]).

Achado da calibração (medido no Modelo A, `lr=0,01`): **a profundidade quase não move o
best_iter — o lever é o `learning_rate`**. Varrendo a profundidade: depth 6 → ~423 iter,
depth 4 → 444, depth 2 → 525, **depth 1 → 803**. Só com `depth=1` (stumps) e
`lr=0,008` (= `0,01·803/1000`, levemente abaixo do piso heurístico 0,01, mas valor que o
próprio CatBoost auto‑seleciona em datasets grandes) o best_iter passou de 1000.

| experimento | AUC | Brier | LogLoss | best_iter |
|---|---|---|---|---|
| Ensemble +lags — Modelo B (depth=1) | 0,7010 | 0,108 | 0,3610 | 4700 |
| Ensemble +lags (A+B) **depth=1** | 0,7033 | 0,108 | 0,3612 | — |
| Ensemble +lags — Modelo A (depth=1) | 0,6971 | 0,108 | 0,3646 | 1071 |

> Ambos os modelos com **`depth=1`, `lr=0,008` (mesmo lr)** e best_iter ≥ 1000 (A=1071,
> B=4700 — o val de B, anos 2019‑20, melhora por muito mais rounds). **Resultado: pior que
> a variante +lags original (0,7033 < 0,7155) e bem abaixo do base (0,741).** Árvores
> `depth=1` são learners fracos demais: muitas iterações não recuperam a capacidade de
> capturar interações. O early stopping precoce **não era a causa** do baixo desempenho —
> forçar ~1000 iter via árvores rasas apenas troca capacidade por número de iterações.
> Artefatos: `outputs/runpod_ensemble_lags_depth1/`.

---

## 3. Conclusões

1. **A correção de normalização foi o maior ganho do projeto**: o ensemble base saltou
   de **0,642 → 0,741 de AUC (+0,099)** sem alterar nenhum modelo — só harmonizando os
   códigos entre anos. (`outputs/figures/impacto_correcao_auc.png`)
2. O **ensemble base corrigido (0,741)** supera inclusive o antigo "vazado" (0,738),
   que era artificialmente inflado por usar 2023 nas próprias taxas.
3. **Os lags pioram** no regime corrigido (0,716 < 0,741). As colunas de lag (`n`/`k_sjc`
   por categoria dos anos anteriores) agem como *target‑encoding*: o modelo decora a
   validação e o early stopping dispara cedíssimo (iter **70/95** vs **825/2516** do base)
   → underfit do sinal generalizável. A crença anterior de que "lags ajudavam"
   (0,642→0,654) era um artefato do regime bugado.
4. **Dados recentes generalizam melhor**: Modelo B (fit 2021‑22) > Modelo A (fit 2019‑20)
   nas duas variantes.
5. **Forçar ~1000 iter não salva os lags**: a variante +lags com `depth=1`/`lr=0,008`
   (early stopping em 1071/4700) ficou em **0,7033 — pior que os lags originais (0,7155)**.
   O early stopping precoce era *sintoma* (capacidade alta + overfit), não a causa; rasear
   as árvores para alongar o treino só troca capacidade por iterações. Confirma a conclusão
   3: os lags não ajudam no regime corrigido.

**Melhor modelo: Ensemble base sem lags** — `outputs/runpod_ensemble_base/`.

---

## 4. Feature importance do melhor modelo (ensemble base)

`outputs/figures/importancia_ensemble_base.png`

| # | feature | imp | # | feature | imp |
|---|---|---|---|---|---|
| 1 | tempo_vinculo_meses | 17,9% | 7 | qtd_dias_afastamento | 5,3% |
| 2 | tipo_vinculo | 16,9% | 8 | natureza_setor | 3,8% |
| 3 | faixa_remuneracao | 10,3% | 9 | cnae2 | 3,5% |
| 4 | tamanho_estab | 7,5% | 10 | cbo1 | 2,5% |
| 5 | natureza_juridica | 6,5% | … | (cbo*/cnae*/idade) | ~2% |
| 6 | uf | 5,8% | 22 | intermitente | 0,2% |

O risco é dominado por **tempo de vínculo + tipo de vínculo + faixa de remuneração (≈45%)**:
vínculos curtos, contratos temporários e faixas salariais baixas concentram a dispensa.
A calibração no holdout é boa (`outputs/figures/calibracao_ensemble_base.png`):
risco previsto ≈ risco observado em todos os decis.

---

## 5. Notas de infraestrutura
- Agregações de lag (2016‑2023) e treinos rodados em **RunPod** via HubService.
- Agg build: pod **CPU 96 núcleos**; ensemble base: **A100 80GB**; ensemble +lags: **B200 180GB**
  (necessária — o fit usa **173 GB de VRAM**; a H200 141GB daria OOM).
- O treino com lags exigiu **construção dos pools em processos isolados** (quantizados em
  disco) para caber no limite de **188 GB de RAM** do container.
- Todas as pods foram destruídas após uso.
