# Score Agregado de Risco de Desligamento (RAIS + Novo CAGED)

Estima o **risco de uma pessoa perder o emprego nos próximos meses** usando uma
**abordagem agregada** sobre dados públicos brasileiros (RAIS + Novo CAGED),
**sem base analítica própria e sem pareamento de painel individual**.

A ideia central: calcular **taxas históricas de desligamento por célula de perfil**
(CBO × CNAE × tempo de vínculo × tamanho da empresa × UF × idade × escolaridade) e
usar a taxa da célula como **score de risco** para qualquer pessoa daquela célula.
Não há modelo supervisionado individual — o "modelo" é uma tabela de taxas
suavizada (Empirical Bayes) com **backoff hierárquico** para células pouco populosas.

## Escopo (decisões)

| Dimensão | Decisão |
|---|---|
| Horizontes | 3, 6 e 12 meses (simultâneos) |
| Período | **Nacional 2020–2023** (4 anos) executado (`anos` em config) |
| CBO/CNAE | Hierárquico/adaptativo (backoff de 4→2 dígitos) |
| Dimensões da célula | CBO, CNAE, tempo de vínculo, tamanho da empresa, UF, idade, escolaridade |
| Tipo de desligamento | Calculado **separadamente por motivo** (default do score: dispensa s/ justa causa) |
| Fonte | RAIS (estrutural) **+ Novo CAGED** (ajuste conjuntural recente) — ambos usados |
| Geografia | UF. **Cobertura nacional executada** (6 regiões, 27 UFs) |
| Entrega | Módulo `src/scoring.py` (com ajuste conjuntural opcional) + notebooks de demonstração |

### Execução com dados reais (já realizada) — escala NACIONAL

Pipeline rodado de ponta a ponta com microdados **reais** do PDET/MTE,
**6 regiões × 2020–2023 = 297,89 milhões de vínculos** (RAIS) + **Novo CAGED 2024**.
Inclui 2020 (ano da pandemia) — `excluir_anos_atipicos: true` em `config.yaml`
remove 2020–2021 das taxas estruturais se quiser o cenário "normal".

**RAIS (estrutural):**
- Download por região (`RAIS_VINC_PUB_<REGIAO>.7z`) — o leitor detecta separador
  (`,` recente / `;` antigo), normaliza decimal e resolve nomes de coluna por tokens,
  tolerando as diferenças de layout entre anos (`src/cleaning.iter_rais_clean_chunks`).
- **Agregação incremental (map-reduce)** — `src/rates.Accumulator`: processa lote a
  lote e soma contagens por célula, com memória baixa (couberam 232M vínculos em 10 GB
  de RAM). Os 2 níveis de backoff mais granulares são excluídos na escala nacional
  (gerariam dezenas de milhões de células 1-2 vínculos sem valor após o shrinkage).
- **Validação temporal** (treino 2020–2022 → holdout 2023): **corr ≈ 0,520, MAE ≈ 0,063**
  em 119 mil células; calibração agregada bem alinhada nos decis centrais, com
  regressão à média nos extremos.

**Novo CAGED (conjuntural, nb07):**
- Layout decodificado (UTF-8, `;`, motivo via `tipomovimentação`) em `src/caged.py`.
- Calcula um **fator de ajuste por célula L = (CBO 2díg × CNAE 2díg × UF)**:
  `fator = hazard_recente_CAGED / hazard_estrutural_RAIS` (suavizado). No scoring,
  `risco_ajustado = risco_estrutural × fator` via `score_pessoa(..., ajuste_conjuntural=True)`.
- O denominador do hazard recente usa o **estoque anual médio** da RAIS (exposição
  acumulada ÷ nº de anos), para o fluxo CAGED anualizado e o estoque ficarem na
  mesma base temporal. Fator mediano ≈ **1,0** (CAGED 2024 em linha com a estrutura
  2021–2023); ~35% das células L sinalizam piora recente (fator > 1).
- Ex.: célula em SP, risco 12m 0,191 → **0,196** com ajuste (fator 1,03).

**Para mudar o recorte:** edite `rais.regioes`, `anos` e `caged` em `config.yaml` e
reexecute. O volume é o único limite (SP ≈ 1 GB compactado/ano).

## Estrutura

```
.
├── config.yaml          # anos, paths, thresholds, faixas de binning, shrinkage
├── data/{raw,interim,processed,dicts}/
├── notebooks/00..08     # pipeline (ingestão -> validação -> CAGED -> benchmark ML)
├── src/                 # lógica reutilizável e testável
├── outputs/{figures,tables}/
└── tests/test_scoring.py
```

## Como rodar

```bash
pip install -r requirements.txt   # inclui py7zr p/ extrair os .7z da RAIS

# Modo REAL (default atual em config.yaml: synthetic_mode: false)
# Baixa a RAIS do FTP do PDET/MTE (regiões/anos em config) e roda o pipeline.
jupyter lab   # execute notebooks 00 -> 06 em ordem

# Modo SINTÉTICO (sem downloads): defina synthetic_mode: true em config.yaml.
# Gera dados de mesmo schema p/ rodar 00->06 ponta a ponta offline.

# Testes de sanidade (usam amostra sintética interna; independem do modo)
pytest -q
```

A cobertura nacional (6 regiões, 2021–2023) está pré-configurada. Ajuste
`rais.regioes`, `anos`, `caged` e `ufs_subset` em `config.yaml` para mudar o escopo.

## Pipeline (notebooks)

0. **Setup e dicionários** — config + de-para de motivos/escolaridade.
1. **Ingestão/download** — baixa os `.7z` da RAIS por região (ou amostra sintética).
2. **Limpeza/padronização** — extrai sob demanda, harmoniza para schema canônico e
   grava `interim` particionado por ano (apaga o extraído para poupar disco).
3. **Células e taxas** — binning, células e taxas por nível de backoff via
   **agregação incremental** (escala nacional sem estourar memória).
4. **EDA** — distribuição, marginais, perfis de maior risco, cobertura.
5. **Scoring** — `score_pessoa(...)` / `score_lote(...)`.
6. **Validação** — estabilidade temporal, calibração agregada, monotonicidade, sensibilidade.
7. **Ajuste conjuntural (CAGED)** — fator recente por célula L e `score_pessoa(..., ajuste_conjuntural=True)`.
8. **Benchmark CatBoost** — modelo supervisionado vs score agregado, no mesmo holdout temporal.

## Benchmark: CatBoost vs score de células (comparação honesta)

Avaliação **out-of-time**: todos os modelos treinam só em **2020–2022** e são testados em **2023**.

| Modelo (holdout 2023) | AUC | Brier |
|---|---|---|
| **CatBoost full-data** (298M, GPU, cardinalidade completa) | **0,708** | 0,107 |
| CatBoost (amostra ~4M) | 0,701 | 0,108 |
| Score de células treino-only | 0,686 | 0,110 |
| ⚠️ Células *com 2023 embutido (vazado)* | 0,738 | 0,104 |

**Conclusão:** na comparação justa, o **CatBoost supera o score de células** (0,708 vs 0,686)
— o modelo supervisionado generaliza melhor para o ano seguinte. Treinar em todos os 298M
vínculos (vs amostra) só melhorou marginalmente (0,701 → 0,708). O 0,738 das células era
**vazamento**: a taxa da célula incluía os desligamentos do próprio ano de teste.

O CatBoost full-data foi treinado em **GPU (RunPod A40)** — o pré-processamento de CTR das
categóricas em 136M linhas é inviável em CPU, mas leva minutos em GPU. Artefatos do modelo
em `outputs/runpod_catboost/` (`catboost_full.cbm`, `metrics.json`, `importancia.csv`,
`vocab.json`) — reaplicáveis. Top features: tempo de vínculo (28,6%), CNAE divisão (12,7%),
tamanho da empresa (10,9%), UF (8,9%).

> **Produção ≠ avaliação.** Para **pontuar uma pessoa hoje**, o score de células usa
> **todos os anos** (`data/processed/rates/`) — correto, aproveita toda a informação. O
> recorte **treino-only** (`data/processed/rates_treino/`) existe **apenas** para esta
> avaliação preditiva honesta. O score de células segue preferível quando o que importa é
> **transparência total, defensabilidade (LGPD)** e operar sem treinar um modelo individual.

## Metodologia em resumo

- **Exposição (denominador)** = nº de vínculos observados na célula no ano.
- **Taxa anual (hazard)** por motivo = desligamentos do motivo / exposição.
- **Suavização**: Empirical Bayes aninhado (Beta-Binomial) começando na taxa global
  e refinando nível a nível: `p̂ = (k + m·p_prior) / (n + m)`.
- **Backoff** (`src/cells.py::BACKOFF_LEVELS`): do mais geral ao mais específico; o
  score recua até onde há suporte estatístico, registrando `nivel_usado` e `exposicao`.
- **Horizonte**: do hazard anual ao risco em H meses assumindo hazard mensal
  aproximadamente constante (`1-(1-h)^H`).
- **Escala (map-reduce)**: as taxas são somas por célula → processadas
  incrementalmente lote a lote (`src/rates.Accumulator`), o que permite cobertura
  nacional × múltiplos anos com RAM limitada.
- **Ajuste conjuntural (CAGED)**: fator recente por célula L = (CBO2 × CNAE2 × UF)
  multiplicando o hazard estrutural — atualiza o nível geral do risco para o
  momento mais recente sem perder a granularidade fina da RAIS (`src/caged.py`).

---

## Limitações da abordagem agregada

- **Falácia ecológica / viés de composição:** a taxa da célula ≠ risco do indivíduo.
  Características não observadas (desempenho, contrato específico, saúde da empresa)
  não são controladas.
- **Score determinístico:** perfis idênticos recebem o mesmo score. Pode **reforçar
  estereótipos** ligados a idade, escolaridade e região.
- **Estrutura ≠ conjuntura:** taxas históricas não antecipam choques futuros; os anos
  de pandemia (2020–2021) distorcem as taxas (`excluir_anos_atipicos` permite removê-los).
- **Cobertura:** RAIS/CAGED cobrem o **emprego formal**; o setor informal fica de fora.
- **Defasagem:** RAIS é anual e com atraso de divulgação; o componente CAGED mitiga,
  mas não elimina, a defasagem.
- **Sem ground truth individual:** a validação é **agregada** (estabilidade, calibração
  por célula), não mede acurácia preditiva individual.

## LGPD e ética

- **Dados de treino** são públicos e usados de forma **agregada** (taxas por célula);
  não há dado pessoal de terceiros armazenado.
- **No uso do score**, os atributos da pessoa consultada **são dados pessoais** (LGPD):
  exigem base legal, finalidade definida, transparência e segurança.
- **Decisões automatizadas:** o score **não deve ser usado isoladamente** para decisões
  que afetem direitos (crédito, emprego) sem **revisão humana** e **direito a explicação**
  (LGPD art. 20). Recomenda-se uso como **sinal exploratório**.
- **Não discriminação:** idade e escolaridade são dimensões sensíveis a discriminação.
  Em contextos de decisão, considere **desligar essas dimensões** (o backoff já permite
  operar em níveis que não as incluem) e auditar disparidades por grupo.
