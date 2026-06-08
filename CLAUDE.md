# CLAUDE.md — Guia do projeto (Score de Risco de Desligamento)

Guia para retomar o trabalho rapidamente. Leia também:
- `outputs/RELATORIO_modelo_2023.md` — bug, correções e resultados.
- `runpod/README.md` — execução no RunPod e lições de transferência/memória.

---

## 1. O que é o projeto

Estima o **risco de uma pessoa perder o emprego nos próximos meses** usando dados
públicos brasileiros (**RAIS** + Novo CAGED), de forma **agregada** (sem base
analítica própria, sem pareamento de painel individual). Duas linhas coexistem:

1. **Score agregado por célula** (a proposta original): taxas históricas de
   desligamento por perfil (CBO × CNAE × tempo × tamanho × UF × idade/escolaridade),
   com suavização Empirical Bayes + backoff hierárquico. Módulo `src/scoring.py`
   (`score_pessoa(...)`), tabelas em `data/processed/rates`.
2. **Benchmark supervisionado (CatBoost)**: ensemble cross-temporal usado para medir
   o teto de performance e validar features. **É o melhor modelo hoje.**

Alvo padrão: `motivo_unificado == "involuntario_sjc"` (dispensa sem justa causa).
Holdout out-of-time: **2023** (nunca usado em treino/early-stopping).

---

## 2. Estrutura

```
src/            # lógica testável (config, cleaning, binning, cells, rates, scoring, ml, lags)
notebooks/      # 00..08 (orquestram src/), gerados por _build_notebooks.py
runpod/         # execução no RunPod via HubService (ver runpod/README.md)
data/raw|interim|processed   # microdados -> limpos -> células/taxas/aggs
outputs/figures|tables       # figuras, tabelas, RELATORIO_modelo_2023.md
config.yaml     # anos, caminhos, binning, suavização, params do ml
```

Dados-chave já materializados:
- `data/interim/rais/ano=YYYY/<regiao>.parquet` — RAIS limpa nacional **2016–2023**
  (90 partições, ~4,3 GB, 563M vínculos). 2016/2017 vêm por UF; 2018+ por região.
- `data/processed/lags/agg_<feature>.parquet` — 19 agregações (valor, ano, n, k_sjc)
  para as features de lag, **2016–2023, já com a normalização corrigida**.

---

## 3. A correção MAIS importante (não reintroduzir!)

Os microdados mudaram o **formato de códigos entre 2022 e 2023**, quebrando features
no holdout. Corrigido em **`src/cleaning.normalize_short_codes`** e aplicado em todo
ponto de leitura (cleaning, `lags.build_lag_aggs`, `cells.add_cell_keys`/`person_keys`,
e nos scripts do pod):
- `causa_afastamento`: remap `'999' -> '99'` (default "sem afastamento", ~84% em 2023);
- strip de zero-padding em `faixa_remuneracao`, `faixa_horas`, `causa_afastamento`
  (`'02' -> '2'`);
- `zfill` consistente de `cbo`(6)/`cnae`(7) **antes** de derivar níveis hierárquicos.

Impacto: **+0,099 de AUC** no ensemble base (0,642 → 0,741) só com isso.
`tipo_vinculo` foi verificado e NÃO precisa de ajuste.

---

## 4. Resultados (holdout 2023) — resumo

Tabela completa: `outputs/tables/metricas_2023_consolidado.csv`.

| modelo | AUC | LogLoss |
|---|---|---|
| **★ Ensemble base (sem lags) — MELHOR** | **0,741** | **0,3477** |
| Ensemble com lags | 0,7155 | 0,3549 |
| (histórico, pré-fix) full-data 0,708 / cells treino-only 0,685 | | |

- **Lags PIORAM** com o bug corrigido (overfit; early-stop em iter 70/95). Não usar.
- Top features: tempo_vínculo (17,9%), tipo_vínculo (16,9%), faixa_remuneração (10,3%).
- Artefatos: `outputs/runpod_ensemble_base/` (melhor) e `outputs/runpod_ensemble_lags/`.

---

## 5. RunPod — como rodar (passo a passo)

Infra: pods GPU/CPU criadas via **HubService** (API em `http://hub:8788`), acesso por
SSH (paramiko) com `runpod/remote.py` lendo `runpod/pod.json`.

### Segredos (NUNCA imprimir)
- Token Bearer: `HUB_API_KEYS` em `/mnt/d/Projetos/HubService/.env`.
  Carregue em variável: `export HUB_TOKEN="$(grep -E '^HUB_API_KEYS=' /mnt/d/Projetos/HubService/.env | head -1 | cut -d= -f2- | tr -d '"' | cut -d, -f1)"`
- Chave SSH privada: `/mnt/d/Projetos/HubService/secrets/runpod_ssh_key` (ed25519).
  O Hub injeta a pubkey correspondente em toda pod automaticamente.

### Ciclo padrão
1. **Criar pod** (`POST /compute/pods`): `template_id` + `gpu_type_id` (NVIDIA) OU
   `cpu_flavor_id` (CPU). Templates: `se7e4g7jo4` (python:3.11 GPU, precisa
   `pip install pandas pyarrow catboost scikit-learn`), `u25nrr587h` (scipy-notebook
   CPU, já tem pandas). GPU ids: `"NVIDIA A100 80GB PCIe"`, `"NVIDIA H200"`, `"NVIDIA B200"`.
2. **Esperar SSH**: pollar `GET /compute/pods/{id}` até `connection.tcp_ports['22']`
   trazer `public_ip`+`public_port`. Gravar em `runpod/pod.json`
   (`{"pod_id","host","port","cost_per_hr","gpu","cmd_timeout":14400}`).
3. **Enviar dados** com `python runpod/sync_interim.py <local> <remoto> [anos...]`
   — UM comando, NÃO disparar outros até concluir (senão o sandbox mata o background).
4. **Rodar** via `python runpod/remote.py exec "cd /workspace && nohup python -u ... > x.log 2>&1 &"`.
5. **Baixar** artefatos: `python runpod/remote.py getdir /workspace/artifacts <local>`.
6. **SEMPRE destruir a pod**: `DELETE /compute/pods/{id}` (custo $0,44–5,89/h; confirmar
   `GET /compute/pods` → `count:0`).

### Treinar o ENSEMBLE BASE (melhor, A100 80GB)
Precisa só do interim 2019–2023 em `/workspace/data/rais` + `train_model.py`.
```bash
# no pod (cada fase = processo isolado p/ robustez):
python -u train_model.py A      # fit 2019-20, val 2021-22 -> catboost_A.cbm
python -u train_model.py B      # fit 2021-22, val 2019-20 -> catboost_B.cbm
python -u train_model.py eval   # holdout 2023 -> metrics/importancia/calibracao
```

### Treinar o ENSEMBLE COM LAGS (B200 180GB obrigatória)
Precisa do interim 2019–2023 + `data/processed/lags/agg_*.parquet` em `/workspace/lags`.
```bash
python -u prep_lags.py                 # enriquece interim com 114 colunas de lag
for M in A B; do
  python -u train_model_lags.py poolfit $M   # processos isolados: monta+quantiza+salva pool
  python -u train_model_lags.py poolval $M   # (quantizado em disco p/ caber em 188GB RAM)
  python -u train_model_lags.py fit $M       # carrega pools quantizados e treina
done
python -u train_model_lags.py eval
```

### Recalcular os aggs de lag (pod CPU 96 núcleos)
```bash
python -u build_aggs_pod.py    # paralelo; se travar nas partições grandes:
python -u finish_aggs.py       # finaliza sequencialmente + merge
```

---

## 6. Pipeline pós-modelo: predict → categorias → personas

Fluxo aplicado ao melhor modelo (ensemble base) sobre 2023. **Etapas 2–4 são
agnósticas ao modelo** — só dependem de um parquet de predições com `prob_desligamento`
+ `y` (+ features p/ personas). Para repetir com OUTRO modelo, refaça só a etapa 1.

Roda LOCALMENTE (CPU). O sandbox mata processos longos (~10 min) → tudo em LOTES e
resumível; o pyarrow falha ao criar arquivo direto no mount `/mnt/d` → escrever em
`/tmp` e copiar. Precisa de `catboost`+`scikit-learn` no venv (`pip install`).

1. **Predict** (`predict_ensemble_base_2023.py`) — model-específico.
   Lê o interim 2023, aplica o MESMO pré-processamento do treino
   (`cleaning.normalize_short_codes` + zfill/níveis), pontua em lotes de 3M linhas
   (resumível) → `outputs/predicoes_2023_ensemble_base.parquet` (22 features + `y` +
   `prob_A`/`prob_B`/`prob_desligamento`; AUC/LogLoss reproduzem o eval).
   Outro modelo: trocar os `.cbm`/caminhos e GARANTIR pré-processamento == treino dele.
2. **Categorias por ganho de informação** (`tune_bins_infogain.py [N_MICRO=1000] [K_MAX=40]`)
   — agnóstico. Para cada K, acha por PROGRAMAÇÃO DINÂMICA os cortes de `prob_desligamento`
   que MAXIMIZAM a informação mútua I(bin;y) (sobre ~1000 micro-bins por quantil). Varre K
   e devolve o MAIOR K que maximiza o IG MANTENDO `y` médio estritamente crescente
   (quebra em K+1) → **K\*=23** (IG 0,0663 bits = 11,7% de H(y)). Saídas:
   `outputs/tables/binning_infogain_{sweep,escolhido}.csv` + figura.
3. **Materializar categoria** (`add_categoria_risco_2023.py`) — agnóstico. Streama o
   parquet, atribui `categoria_risco` 1..K via `searchsorted` nas bordas do
   `binning_infogain_escolhido.csv` → `..._categorizado.parquet` (todas as colunas + a nova).
4. **Personas** (`persona_categorias.py`) — agnóstico (precisa das features no parquet).
   Por categoria: composição interna (buckets) + distintividade via LIFT (share na
   categoria / share global, piso 5%) de CBO/CNAE/UF + médias numéricas, via
   `pyarrow.group_by`, traduzido pelo dicionário RAIS. Saídas:
   `outputs/tables/persona_categorias.csv` + `outputs/PERSONAS.md`. **Modo
   `python persona_categorias.py privado`** filtra o setor público (`natureza_setor!='1'`)
   → `persona_categorias_privado.csv` (público-alvo do **consignado privado**). No deck,
   isso vira os slides de **Apêndice** (o piso de risco deixa de ser o servidor público
   e passa a ser o veterano de banco/grande empresa privada).
5. **Apresentação PDF** (`gerar_apresentacao.py`) — deck 16:9 via `matplotlib.PdfPages`
   (sem deps externas), 3 partes: (a) treino do ensemble base, (b) categorização (curva
   IG×K + tabela das 23), (c) personas (4 small-multiples do gradiente + 1 slide por
   grupo de risco Mínimo→Alto). Lê metrics/importância/calibração + tabelas
   `binning_infogain_*`/`persona_categorias.csv` + figuras de `outputs/figures/`. Saída:
   `outputs/apresentacao_risco_desligamento.pdf`. **Inspeção:** sem poppler, dumpar páginas
   via `runpy.run_path(...)["pages"][i].savefig(png)` e ler o PNG. Outro modelo: refaça 1–4
   e **REESCREVA os textos das personas** (dict `PERSONA_TXT`) e rótulos (são específicos
   dos achados; números/gráficos se atualizam das tabelas). `MPLCONFIGDIR=/tmp/mpl`.

---

## 7. Restrições de memória (aprendizados)
- **RAM do container ~188 GB**: construir 2 Pools (132M+148M × 136 feat) estoura →
  use processos isolados + pools quantizados em disco (`quantized://`).
- **VRAM do fit com 136 features ≈ 173 GB** → exige **B200 180GB** (H200 141GB = OOM).
- Ensemble base (22 feat) cabe em **A100 80GB** (~77 GB de VRAM).
- CatBoost: `task_type=GPU`, `loss/eval=Logloss`, `max_ctr_complexity=1`, `max_bin=128`,
  `boosting_type=Plain`, early stopping. CTR de categóricas de alta cardinalidade é o
  gargalo de pré-processamento (CPU, lento).

---

## 8. Ambiente local
- venv volátil (perde em reinício): recriar em `/tmp/consig_venv`
  `python3 -m venv --copies --without-pip /tmp/consig_venv` + bootstrap get-pip +
  `pip install pandas pyarrow numpy pyyaml` (e `catboost scikit-learn matplotlib` se precisar).
- Não usar `/mnt/d` para venv com symlinks (falha). Comandos longos podem ser mortos
  pelo sandbox (~10 min) → preferir RunPod para treinos pesados.

---

## 9. Convenções
- Código real e comentado em PT-BR; `src/` concentra lógica, notebooks orquestram.
- Confirmar com o usuário antes de subir pod paga. Usuário é consciente de custo.
- Commits só quando pedido; mensagens terminam com a linha de co-autoria padrão.
