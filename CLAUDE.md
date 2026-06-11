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
   o teto de performance e validar features. **O modelo VIGENTE é a esteira 2124
   (ver §1-B)**; o anterior (treino 2019–22, holdout 2023) está preservado como legado.

Alvo padrão: `motivo_unificado == "involuntario_sjc"` (dispensa sem justa causa).
(Legado v1: holdout 2023. Esteira 2124: avaliação em TODOS os anos 2016–2025;
2025 = out-of-time puro.)

---

## 1-B. ESTEIRA ATUAL — "2124" (modelo v2, sufixo `_2124` em scripts/saídas)

Cadeia completa, na ordem (cada etapa lê a saída da anterior):

1. **Interim leak-free 2016–2025** (`rebuild_interim.py` → `data/interim/rais/ano=YYYY/`,
   106 partições, 743.038.238 vínculos; backup do schema antigo em `rais_old_schema_bak`).
   Schema (26 cols): `id_linha` (chave única `ano_regiao_nºlinha`), códigos CRUS
   (escolaridade 1..11, faixas int64 com 99=ignorado), cbo zfill(6)/cnae zfill(7),
   `tempo_vinculo_meses` = antiguidade NA ENTRADA (leak-free), `qtd_dias_afastamento`
   = dias POR MÊS observado (leak-free), `mes_admissao` (0=vigente; 1-12), desfecho ao
   final (`vinculo_ativo`, `mes_deslig`, `motivo_desligamento`, `motivo_unificado`).
   SEM coluna `separado` (usar `vinculo_ativo==0`).
2. **Treino** (`runpod/train_model_2124.py`, pod H200 ~$4,4/h): A fit 21-22/val 23-24;
   B fit 23-24/val 21-22; ensemble=média. 21 features (14 cat + 7 num; ordinais
   escolaridade/tamanho/faixas como NUMÉRICAS com 99→-1; SEM causa_afastamento).
   Eval em 2016–2025 com AUC/KS/LogLoss/Brier → `outputs/runpod_retreino_2124/`
   (`metricas_por_ano.csv`, calibracao_YYYY.csv, importancia, .cbm).
   **Resultado: AUC 0,776 / KS 0,403 em 2025 (futuro puro); AUC 0,76–0,81 nos 10 anos.**
3. **Predict todos os anos** (`predict_ensemble_2124_todos_anos.py`, local ~21min) →
   `data/processed/predicoes_2124/ano=YYYY/` (id_linha + features + desfecho + y +
   prob_A/B/prob_desligamento + categoria_risco).
4. **Categorias** (`tune_bins_infogain_2124.py`, ref. 2021–24): PD maximiza I(bin;y) com
   critério DUPLO — y médio estritamente crescente no pooled E em CADA ano 21–24 →
   **K\*=14** (quebra em K=15/ano 2024; 99,1% do IG). Ordenação validada em TODOS os
   anos 2016–2025 (`resumo_categoria_ano_2124.py`). Materialização in-place:
   `add_categoria_risco_2124.py`. Bordas: `outputs/tables/binning_infogain_escolhido_2124.csv`.
5. **Personas** (`persona_categorias_2124.py`; referência default = **2025**, via
   `ANOS_REF` na argv — o docstring antigo dizia 21–24, mas o perfil descrito nos decks
   é o de 2025; 1 passada p/ geral+privado) → `persona_categorias_2124{,_privado}.csv`.
   Grupos de risco (decks): Mínimo[1], Baixo[2-4], Médio-Baixo[5-7], Médio[8-10],
   Alto[11-14]. **Se mudar K, reescrever.**
6. **Sobrevivência MOB** (`sobrevivencia_mob_2124.py`, ref. 21–24 agregados): KM ≤12 MOB
   + Weibull cloglog >12 (R²=0,994) + isotônica (0 inversões — frailty quase sumiu) →
   `sobrevivencia_*_mob_2124.csv` + figuras.
7. **Política de consignado** (`tabelas_consignado_2124.py`): prazo máx. por confiança
   95/90/85/80 + cobertura de parcelas T=6..60 → `consignado_*_2124.csv`.
8. **Decks** (não substituem os antigos): `gerar_apresentacao_2124.py` (PDF 34 slides),
   `gerar_apresentacao_html_2124.py` (HTML autossuficiente/offline c/ B1/B2/B3
   interativos + tabela C1 + navegação por swipe e aria),
   `gerar_apresentacao_diretoria_2124.py` (executivo 6 slides). Figuras de apoio:
   `fig_modelo_2124.py`. Convenções dos decks 2124 (manter):
   - números dos bullets de persona são COMPUTADOS dos CSVs de personas
     (`_PIDX`/`_rngc`/`_liftmax` em gerar_apresentacao_2124.py) — só a narrativa
     qualitativa é manual; nada de número hardcoded que descole dos dados;
   - chips de característica com lift ≈ 1 (±0,15) viram "≈ média" cinza, sem seta ▲/▼;
   - prazos máximos do consignado: truncados p/ baixo (floor; "<1" quando <1 mês) —
     nunca round, que superestimaria o prazo seguro;
   - referências temporais: slide 8 e tabelas C = pooled 2021–24; personas = perfil
     2025 (sempre rotular qual é qual).

Aprendizados novos (não reintroduzir):
- `tempo_vinculo_meses` da RAIS é medido no FIM do vínculo → vazava o "quando" do alvo
  (corrigido p/ entrada); `qtd_dias_afastamento` truncado por exposição (corr +0,082 com
  mes_deslig → taxa/mês = +0,018); `causa_afastamento` removida das features (mesmo leak).
- Predições salvam ordinais como float32 (`9.0`) → consumidores devem normalizar
  (`_s()` em persona_categorias_2124.py).
- Aggs de lag antigos são INCOMPATÍVEIS com o interim novo (escolaridade era "superior",
  agora código) — reconstruir build_aggs antes de qualquer prep_lags.

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
- `data/interim/rais/ano=YYYY/<regiao>.parquet` — RAIS limpa nacional **2016–2025**
  (106 partições, ~11 GB, 743M vínculos, schema leak-free 26 cols — ver §1-B).
  2016/2017 vêm por UF; 2018+ por região (incl. NI 2022+). Backup do schema antigo
  (2016–2023, 24 cols) em `data/interim/rais_old_schema_bak`.
- `data/processed/predicoes_2124/ano=YYYY/` — predições categorizadas do modelo v2
  (todos os anos, 15 GB).
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

## 4. Resultados (holdout 2023) — resumo [LEGADO v1; vigente = §1-B]

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
   (sem deps externas), 4 partes: (a) treino do ensemble base, (b) categorização (curva
   IG×K + tabela das 23), (c) personas (4 small-multiples do gradiente + 1 slide por
   grupo de risco Mínimo→Alto), (d) **apêndice consignado privado** (contexto + 5 personas
   recomputadas SEM o setor público, de `persona_categorias_privado.csv`). Lê metrics/
   importância/calibração + tabelas `binning_infogain_*`/`persona_categorias*.csv` +
   figuras de `outputs/figures/`. Saída: `outputs/apresentacao_risco_desligamento.pdf`.
   **Inspeção:** sem poppler, dumpar páginas via `runpy.run_path(...)["pages"][i].savefig(png)`
   e ler o PNG. Outro modelo: refaça 1–4 (incl. `persona_categorias.py privado`) e
   **REESCREVA os textos das personas** — dicts `PERSONA_TXT` (geral) e `PERSONA_TXT_PRIV`
   (apêndice) — e rótulos (são específicos dos achados; números/gráficos se atualizam das
   tabelas). `grupo_slide` é parametrizado (`pers`/`persona_txt`/`inds_spec`). `MPLCONFIGDIR=/tmp/mpl`.

---

## 6-B. Sobrevivência e tempo até desligamento (extensão da §6)

Estima **quando** (não só "se") ocorre o desligamento, por categoria de risco, via análise
de sobrevivência (Kaplan-Meier) + extrapolação Weibull. Roda **LOCALMENTE** (venv
`/tmp/consig_venv`; precisa pandas/pyarrow/numpy/matplotlib; sempre `MPLCONFIGDIR=/tmp/mpl`).
**Agnóstico ao modelo**: depende só do parquet categorizado (`categoria_risco`,
`prob_desligamento`,`y`) + interim RAIS (`mes_deslig`,`motivo_unificado`). **Para OUTRO modelo:
refaça a §6 (predict→categoria) e reaponte as constantes SRC/CATPARQ dos 5 scripts.**

Rodar nesta ordem (cada script lê a saída do anterior):

1. **`curva_sobrevivencia_categorias.py`** — Kaplan-Meier por categoria.
   - `evento` = `motivo_unificado=="involuntario_sjc"`; `tempo` = `mes_deslig` (1..12);
     censura = ativo (`mes_deslig==0`)→t=12; saída por OUTRO motivo no mês m→censura em m.
   - **Alinhamento sem ID de pessoa**: lê o interim 2023 na MESMA ordem do predict
     (`sorted(glob)`, `iter_batches`) e VALIDA `evento==y` (do parquet categorizado) em 100%
     das linhas → garante a junção por posição (82.964.122 linhas). Cache `_surv_counts_2023.csv`.
   - KM: `S(t)=Π (n−d)/n`; IC Greenwood; `RMST(12)=Σ_{m=0}^{11} S(m)` (= meses esperados de
     emprego no ano); mediana só definida se S cruza 0,5 (só cats de risco alto, ~21–23).
   - Saídas: `outputs/tables/sobrevivencia_km_2023.csv` (longo cat×mês), `..._resumo_2023.csv`,
     `outputs/figures/sobrevivencia_categorias_2023.png`.
2. **`extrap_weibull_categorias.py`** — Weibull por REGRESSÃO PURA (sem ancoragem).
   - Ajuste: linearização cloglog `ln(−ln S)=p·ln t+ln α` por OLS nos **12 pontos** → shape p,
     α, escala λ=α^(−1/p). Curva: `S(t)=exp(−α·t^p)` (não passa forçada por ponto nenhum;
     extrapola até 36m). R² médio ≈ **0,992**.
   - Estatísticas do tempo T (forma fechada): **média = λ·Γ(1+1/p)**; **quantil q:
     t_q = λ·(−ln(1−q))^(1/p)** → Q1(0,25), mediana(0,5), Q3(0,75).
   - Saídas: `..._weibull_params_2023.csv`, `..._weibull_extrap_2023.csv` (cat×mês 0..36),
     `..._weibull_estatisticas_2023.csv`, `outputs/figures/sobrevivencia_weibull_extrap_2023.png`.
3. **`monotoniza_estatisticas.py`** — impõe monotonicidade (decrescente com a categoria) via
   **isotonic regression (PAVA ponderado por n)** em Q1/mediana/média/Q3. Q1 e mediana já saem
   monotônicos; **média e Q3 têm inversões nas cats ~13–20 (frailty, shape p<1) → viram platô**.
   Saída: `..._weibull_estatisticas_mono_2023.csv` (brutas + `_mono`). **Versão OFICIAL = isotônica.**
   (Regressão log-linear/power-law foi testada como alternativa suave e DESCARTADA: distorce demais.)
4. **`gerar_html_sobrevivencia.py`** — HTML interativo autossuficiente (offline, sem CDN):
   seleção por categoria + 5 botões de persona/grupo de risco, escala-Y dinâmica, toggle
   "Extrapolação Weibull (até 36m)" (tracejado), tooltip com S(t)+mediana+IQR, lista com
   med/IQR/RMST. Lê km + resumo + weibull_extrap/params + estatisticas_mono.
   Saída: `outputs/sobrevivencia_interativa.html`.
5. **`grafico_estatisticas_categorias.py`** — gráfico-caixa por categoria (Y log): caixa=IQR(Q1–Q3),
   linha=mediana, losango=média, **5 personas marcadas por cor + faixas de fundo + rótulos**.
   Saída: `outputs/figures/estatisticas_tempo_categorias_2023.png`.

Os 3 gráficos (KM, extrapolação Weibull, gráfico-caixa) + 1 divisor também entram como
**Apêndice B** do deck (`gerar_apresentacao.py`, slides `B`/`B1`–`B3`): B1 = teoria KM + curvas;
B2 = teoria Weibull + extrapolação; B3 = gráfico-caixa + tabela Q1/med/méd/Q3 monotonizada.

**Deck em HTML** (`gerar_apresentacao_html.py` → `outputs/apresentacao_risco_desligamento.html`):
roda o deck via `runpy` exportando 1 **SVG vetorial** por slide (env `DECK_DUMP_PNG` + `DECK_DUMP_FMT=svg`,
`svg.fonttype=none`), embute cada slide como **SVG inline** (não PNG; texto em `<text>` **selecionável**,
com a fonte **DejaVu embutida via `@font-face`** base64 — métricas idênticas às do matplotlib; ids
prefixados por slide `sNN_` p/ não colidir entre os 24 SVGs estáticos) e troca os slides **B1/B2/B3** por
gráficos SVG **interativos**: B1/B2 = curvas de sobrevivência (seleção por categoria/grupo, escala-Y
dinâmica, tooltip); **B3 = gráfico-caixa** Q1/mediana/média/Q3 (caixa=IQR, hover na caixa OU na linha
da tabela realça e mostra os dados num box). Reaproveita o motor de `sobrevivencia_interativa.html`.
Fontes/controles em `--u` (1% da largura do palco) p/ escalar com o slide; navegação fixa ao canto com
`100dvh` (não some no tablet em paisagem). Autossuficiente/offline; navegação por ←/→.
O **slide 5 (desempenho)** também é nativo/interativo: calibração em SVG + **importância clicável**
(clique numa variável → explicação + exemplos de valores do dicionário `FEATINFO`, com de-paras
da RAIS). Índices: desempenho=`4`, B1/B2/B3=`NP-3`/`NP-2`/`NP-1`. **Deck 100% vetorial (0 PNG raster).**
Ao final há ainda **2 slides HTML-only de aplicação ao crédito consignado** (anexados após o loop,
não estão no PDF): (1) explicação de como S(t) define o prazo do consignado; (2) duas tabelas por
categoria — **prazo máximo** (confiança 95/90/85/80%, via `t=λ·(−ln c)^(1/p)`, inteiros, cap "120+")
e **cobertura esperada de parcelas** (% pagas em folha por prazo T=6..60, `Σ S(m)/T`; S=KM≤12m + Weibull).
Total: 30 slides.

Aprendizados (não reintroduzir):
- **Sazonalidade de dezembro** no hazard (spike ×1,3–4,3) + **rampa nos meses 1–3**; o Weibull
  liso NÃO captura. Logo **extrapolação >12m é SUPOSIÇÃO**; padrão-ouro = coorte sintética com
  RAIS 2024/2025 (recém-publicadas — "Caminho C", ainda não implementado).
- **Frailty**: cats de risco alto (17–20) têm shape p<1 (hazard decrescente) → média/Q3 saem fora
  de ordem (a cauda extrapolada infla); **use mediana e Q1 para ranquear** (robustas, monotônicas).
- Grupos de risco/personas (iguais ao deck): Mínimo[1,2], Baixo[3–6], Médio-Baixo[7–11],
  Médio[12–17], Alto[18–23]. **Se mudar K/modelo e as categorias mudarem, reescreva esses ranges**
  em `gerar_html_sobrevivencia.py` e `grafico_estatisticas_categorias.py`.

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
