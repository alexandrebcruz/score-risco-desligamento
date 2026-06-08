# Score de Risco de Desligamento (RAIS + Novo CAGED)

Estima o **risco de uma pessoa perder o emprego nos próximos meses** a partir de dados
públicos brasileiros (**RAIS** + **Novo CAGED**), de forma **agregada** — sem base
analítica própria e sem pareamento de painel individual.

O projeto tem **duas linhas que se complementam**:

1. **Score agregado por célula de perfil** (a proposta original): taxas históricas de
   desligamento por `CBO × CNAE × tempo de vínculo × tamanho × UF × idade × escolaridade`,
   suavizadas (Empirical Bayes) com **backoff hierárquico**. O "modelo" é uma tabela de
   taxas consultável — transparente e defensável (LGPD). Módulo `src/scoring.py`.
2. **Benchmark supervisionado (CatBoost)**: ensemble cross-temporal usado para medir o
   teto de performance e validar features.

> 🏆 **Melhor modelo hoje: ensemble CatBoost base — AUC 0,741 / LogLoss 0,3477** no
> holdout out-of-time de **2023** (82,96 milhões de vínculos). Chegou lá após
> **descobrir e corrigir um bug de harmonização de códigos entre anos** que sozinho
> valeu **+0,099 de AUC**. Detalhes em [`outputs/RELATORIO_modelo_2023.md`](outputs/RELATORIO_modelo_2023.md).

Alvo padrão: `motivo_unificado == "involuntario_sjc"` (dispensa sem justa causa).

---

## Índice
1. [Dados e escala](#1-dados-e-escala)
2. [Pipeline (notebooks)](#2-pipeline-notebooks)
3. [Abordagem agregada (células, taxas, scoring, CAGED)](#3-abordagem-agregada)
4. [Benchmark supervisionado: CatBoost → ensemble → lags](#4-benchmark-supervisionado)
5. [A correção de normalização (o maior ganho do projeto)](#5-a-correção-de-normalização)
6. [PGFN — histórico de dívidas das empresas](#6-pgfn--histórico-de-dívidas)
7. [Infraestrutura RunPod](#7-infraestrutura-runpod)
8. [Estrutura do repositório](#8-estrutura-do-repositório)
9. [Como rodar](#9-como-rodar)
10. [Limitações, LGPD e ética](#10-limitações-lgpd-e-ética)

---

## 1. Dados e escala

| Fonte | O que é | Cobertura executada |
|---|---|---|
| **RAIS Vínculos** (PDET/MTE) | base estrutural anual | **Nacional 2016–2023** — 90 partições, ~4,3 GB limpos, **563M vínculos** |
| **Novo CAGED** | fluxo conjuntural recente | 2024 (ajuste por célula) |
| **PGFN — Dívida Ativa** | dívidas previdenciárias/FGTS | trimestral 2020→atual |

- RAIS 2016/2017 vêm por **UF**; 2018+ por **região** (6 regiões). O leitor detecta
  separador (`,` recente / `;` antigo), normaliza decimal e resolve nomes de coluna por
  **tokens**, tolerando as diferenças de layout entre anos (`src/cleaning.py`).
- Limpos e particionados em `data/interim/rais/ano=YYYY/<regiao>.parquet`.
- **Agregação incremental (map-reduce)** (`src/rates.Accumulator`): soma contagens por
  célula lote a lote → cabe centenas de milhões de vínculos em RAM limitada.

> Os microdados e tabelas pesadas **não são versionados** (ver `.gitignore`). O pipeline
> os baixa/gera. Use `config.yaml` (`rais.regioes`, `anos`, `caged`, `ufs_subset`) para
> mudar o recorte; há também `synthetic_mode: true` para rodar offline com dados de mesmo schema.

---

## 2. Pipeline (notebooks)

Os notebooks orquestram a lógica de `src/` (código real e comentado, em PT-BR):

| nb | etapa |
|---|---|
| `00_setup_e_dicionarios` | config + de-para de motivos/escolaridade |
| `01_ingestao_download` | baixa os `.7z` da RAIS por região (ou amostra sintética) |
| `02_limpeza_padronizacao` | harmoniza para o schema canônico → `interim` particionado |
| `03_celulas_e_taxas` | binning, células e taxas por nível de backoff (incremental) |
| `04_eda_taxas` | distribuição, marginais, perfis de maior/menor risco, cobertura |
| `05_funcao_scoring` | `score_pessoa(...)` / `score_lote(...)` |
| `06_validacao_sanidade` | estabilidade temporal, calibração, monotonicidade, sensibilidade |
| `07_ajuste_caged` | fator conjuntural por célula L e `ajuste_conjuntural=True` |
| `08_catboost` | benchmark supervisionado vs score agregado (mesmo holdout) |
| `08_pgfn_lista_empresas` | lista de empresas com dívida previdenciária/FGTS |

---

## 3. Abordagem agregada

A linha original do projeto: **não há previsão individual supervisionada** — calcula-se a
taxa de desligamento de cada célula de perfil e usa-se essa taxa como score.

- **Exposição (denominador)** = nº de vínculos observados na célula no ano.
- **Taxa anual (hazard)** por motivo = desligamentos do motivo ÷ exposição.
- **Suavização**: Empirical Bayes aninhado (Beta-Binomial) da taxa global refinando nível
  a nível: `p̂ = (k + m·p_prior) / (n + m)`.
- **Backoff hierárquico** (`src/cells.py::BACKOFF_LEVELS`): do mais geral ao mais
  específico; o score recua até onde há suporte estatístico, registrando `nivel_usado` e
  `exposicao`. Permite **desligar dimensões sensíveis** (idade/escolaridade) operando em
  níveis que não as incluem.
- **Horizonte**: do hazard anual ao risco em H meses (`1-(1-h)^H`), para 3/6/12 meses.
- **Validação temporal** (treino 2020–2022 → holdout 2023): **corr ≈ 0,52, MAE ≈ 0,063**
  em 119 mil células; calibração agregada bem alinhada nos decis centrais.

### Ajuste conjuntural com o Novo CAGED (`src/caged.py`, nb07)
Calcula um **fator por célula L = (CBO 2díg × CNAE 2díg × UF)**:
`fator = hazard_recente_CAGED / hazard_estrutural_RAIS` (suavizado). No scoring,
`risco_ajustado = risco_estrutural × fator` via `score_pessoa(..., ajuste_conjuntural=True)`.
Atualiza o **nível** do risco para o momento recente sem perder a granularidade da RAIS.

```python
from src.scoring import score_pessoa
score_pessoa(cbo="252105", cnae="6422100", uf="SP", idade=38,
             escolaridade="superior", tempo_vinculo_meses=48,
             tamanho_empresa=400, ajuste_conjuntural=True)
# -> risco por horizonte/motivo + nivel_usado, exposicao, intervalo de confiança
```

---

## 4. Benchmark supervisionado

Modelo CatBoost para medir o teto de performance, sempre **out-of-time**: treina em
≤2022 e testa em **2023** (nunca visto no fit/early-stopping). Setup do ensemble:
**Modelo A** treina 2019-20 (val 2021-22), **Modelo B** treina 2021-22 (val 2019-20),
ensemble = média das probabilidades; perda e early-stopping em **Logloss**.

### Resultados consolidados (holdout 2023)
Tabela completa: [`outputs/tables/metricas_2023_consolidado.csv`](outputs/tables/metricas_2023_consolidado.csv).

| modelo | AUC | LogLoss | obs |
|---|---|---|---|
| **★ Ensemble base (sem lags)** | **0,741** | **0,3477** | **melhor** |
| Ensemble com lags (136 feat) | 0,7155 | 0,3549 | lags pioram (overfit) |
| CatBoost full-data *(pré-fix)* | 0,708 | 0,358 | sem as 8 features novas |
| Score de células (treino-only) | 0,686 | 0,372 | abordagem agregada |
| ⚠️ Células c/ 2023 *(vazado)* | 0,738 | 0,346 | usa 2023 nas taxas — não honesto |

**Leituras:**
- O **ensemble base (0,741)** é o melhor — supera até o antigo "vazado" (0,738).
- **Lags pioram** depois da correção: as features de lag (`n`/`k_sjc` por categoria dos
  anos anteriores) agem como *target-encoding*, overfittam e disparam early-stop cedíssimo
  (iter 70/95 vs 825/2516 do base). A crença anterior de que "lags ajudavam" era um
  artefato do regime bugado.
- **Dados recentes generalizam melhor**: Modelo B (fit 2021-22) > Modelo A (fit 2019-20).

### Importância (ensemble base)
`tempo_vinculo` 17,9% · `tipo_vinculo` 16,9% · `faixa_remuneracao` 10,3% · `tamanho_estab`
7,5% · `natureza_juridica` 6,5% · `uf` 5,8% · `qtd_dias_afastamento` 5,3% …
(figuras em `outputs/figures/`, artefatos em `outputs/runpod_ensemble_base/`).

> **Produção ≠ avaliação.** Para pontuar uma pessoa hoje, o **score de células** usa
> **todos os anos** (`data/processed/rates/`). O recorte treino-only existe só para a
> avaliação honesta. O score de células segue preferível quando o que importa é
> **transparência e defensabilidade (LGPD)**; o CatBoost mostra o teto de performance.

---

## 5. A correção de normalização

O **maior ganho do projeto** não foi escolha de modelo — foi corrigir um bug de dados.
Os microdados da RAIS mudaram o **formato de alguns códigos entre 2022 e 2023**. Como o
modelo treina em ≤2022 e é testado em 2023, isso quebrava silenciosamente features
importantes **só no holdout**:

| feature | 2019–2022 | 2023 | efeito |
|---|---|---|---|
| `faixa_remuneracao`, `faixa_horas` | `'02','03'…` (zero-padded) | `'2','3'…` | categoria do treino some no holdout |
| `causa_afastamento` | `'99'` (≈83%) | `'999'` (≈84%) | categoria **majoritária** vira desconhecida |
| `cbo` / `cnae` | `'010105'` | `'10105'` | quebra códigos iniciados em 0 (grupo militar) |

Corrigido em **`src/cleaning.normalize_short_codes`** (remap `999→99`, strip de
zero-padding, `zfill` consistente de cbo/cnae antes dos níveis hierárquicos) e aplicado em
**todo ponto de leitura** (cleaning, lags, células/scoring, scripts do pod). `tipo_vinculo`
foi verificado e **não** precisou de ajuste.

**Impacto: ensemble base 0,642 → 0,741 (+0,099 de AUC)** sem mexer em nenhum modelo.
(Esse bug foi prenunciado por uma investigação sobre se os códigos `01/02/03` mantinham o
mesmo significado ao longo dos anos.)

---

## 6. PGFN — histórico de dívidas

Módulo `src/pgfn.py` + nb `08_pgfn_lista_empresas`: baixa os **Dados Abertos da Dívida
Ativa da União** (trimestral, 2020→atual), filtra **dívidas previdenciárias e de FGTS** e
extrai a **lista de empresas (CNPJ raiz) que já constaram como devedoras** no período —
um possível sinal complementar de fragilidade do empregador. Validação manual incluída
(ex.: conferência de grandes nomes na base). Saída em `outputs/tables/` (o CSV completo de
devedores é pesado e fica fora do versionamento).

---

## 7. Infraestrutura RunPod

Treinos pesados e recálculo de agregações rodam em **pods GPU/CPU sob demanda**, criadas
via **HubService** (API `http://hub:8788`) e acessadas por SSH (`runpod/remote.py`).
Guia completo em [`runpod/README.md`](runpod/README.md) e [`CLAUDE.md`](CLAUDE.md).

- **`runpod/sync_interim.py`** — upload robusto e retomável (idempotente + atômico,
  reconecta sozinho). Use **um comando só** e aguarde concluir.
- **`runpod/train_model.py`** — ensemble base (`A | B | eval`), 1 fase por processo
  (robusto). Cabe em **A100 80GB**.
- **`runpod/train_model_lags.py`** — ensemble com lags; constrói os pools em processos
  isolados e **quantizados em disco** (`quantized://`) para caber em 188 GB de RAM.
  O fit usa ~**173 GB de VRAM** → exige **B200 180GB** (H200 141GB dá OOM).
- **`runpod/build_aggs_pod.py`** / **`finish_aggs.py`** — recálculo paralelo das
  agregações de lag (pod CPU, multiprocessing).

Lições aprendidas (memória, transferência, custo) estão documentadas para não repetir.
**As pods são sempre destruídas após uso.**

---

## 8. Estrutura do repositório

```
.
├── config.yaml              # anos, paths, binning, suavização, params do ml
├── src/                     # lógica reutilizável e testável
│   ├── cleaning.py          #   leitura/harmonização RAIS + normalize_short_codes ⭐
│   ├── binning.py cells.py rates.py   # binning, células, taxas (EB + backoff)
│   ├── scoring.py           #   score_pessoa / score_lote (+ ajuste CAGED)
│   ├── caged.py             #   fator conjuntural
│   ├── ml.py lags.py        #   benchmark CatBoost e features de lag
│   └── pgfn.py              #   dívida ativa (PGFN)
├── notebooks/00..08         # pipeline ponta a ponta
├── runpod/                  # execução no RunPod (ver runpod/README.md)
├── outputs/
│   ├── RELATORIO_modelo_2023.md     # bug, correções e resultados ⭐
│   ├── figures/ tables/             # figuras e tabelas (pequenas versionadas)
│   └── runpod_ensemble_base/ ...    # métricas/importância/calibração dos modelos
├── tests/                   # pytest de sanidade do scoring
├── CLAUDE.md                # guia rápido de retomada + como rodar no RunPod ⭐
└── README.md
```

---

## 9. Como rodar

```bash
pip install -r requirements.txt        # + py7zr p/ extrair os .7z da RAIS

# Modo REAL (default em config.yaml: synthetic_mode: false)
jupyter lab                            # execute notebooks 00 -> 08 em ordem

# Modo SINTÉTICO (offline, sem downloads): synthetic_mode: true em config.yaml
# Testes de sanidade (amostra sintética interna; independem do modo)
pytest -q
```

Treinos pesados (ensemble/CatBoost full-data): ver seção [7](#7-infraestrutura-runpod) e
[`CLAUDE.md`](CLAUDE.md). Ambiente local: o sandbox mata processos longos (~10 min) e o
venv é volátil — para treinos use RunPod.

---

## 10. Limitações, LGPD e ética

**Limitações da abordagem agregada**
- **Falácia ecológica / viés de composição:** a taxa da célula ≠ risco do indivíduo;
  características não observadas (desempenho, contrato, saúde da empresa) não são controladas.
- **Score determinístico:** perfis idênticos recebem o mesmo score — pode reforçar
  estereótipos (idade, escolaridade, região).
- **Estrutura ≠ conjuntura:** taxas históricas não antecipam choques; 2020–2021 (pandemia)
  distorcem (`excluir_anos_atipicos` permite removê-los).
- **Cobertura:** RAIS/CAGED cobrem o **emprego formal**; informal fica de fora.
- **Sem ground truth individual:** a validação é **agregada** (estabilidade, calibração).

**LGPD e ética**
- Dados de treino são públicos e usados de forma **agregada** (sem dado pessoal de terceiros).
- No **uso** do score, os atributos da pessoa consultada **são dados pessoais**: exigem base
  legal, finalidade, transparência e segurança.
- **Não usar isoladamente** para decisões que afetem direitos (crédito, emprego) sem
  **revisão humana** e **direito a explicação** (LGPD art. 20) — recomenda-se uso como
  **sinal exploratório**.
- **Não discriminação:** idade e escolaridade são sensíveis; considere desligá-las nos
  contextos de decisão (o backoff permite) e auditar disparidades por grupo.

---

### Documentos relacionados
- [`CLAUDE.md`](CLAUDE.md) — guia de retomada e instruções de RunPod.
- [`runpod/README.md`](runpod/README.md) — transferência, memória e scripts do pod.
- [`outputs/RELATORIO_modelo_2023.md`](outputs/RELATORIO_modelo_2023.md) — bug, correções e métricas detalhadas.
