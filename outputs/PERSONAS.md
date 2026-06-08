# Personas das 23 categorias de risco de desligamento

Análise de "persona" de cada uma das **23 categorias de risco** (1 = menor risco …
23 = maior risco) criadas pela discretização ótima da probabilidade prevista pelo
**ensemble base** (holdout 2023, 82,96M vínculos). As categorias vêm de
`tune_bins_infogain.py` (maximização do ganho de informação) e foram materializadas
em `predicoes_2023_ensemble_base_categorizado.parquet` (coluna `categoria_risco`).

Dados completos: [`tables/persona_categorias.csv`](tables/persona_categorias.csv).
Gerado por [`../persona_categorias.py`](../persona_categorias.py).

---

## Como funciona o algoritmo de perfilagem

O objetivo é descrever **o que caracteriza cada categoria** a partir das colunas do
dataset, sem olhar uma a uma as 83M linhas. Para cada feature, combinamos duas visões:

1. **Composição interna** — *"do que esta categoria é feita?"*
   Para cada feature categórica, conta-se um **crosstab `categoria × valor`** e
   normaliza-se por categoria (% de cada valor dentro da categoria). Códigos afins são
   agrupados em buckets interpretáveis (ex.: tipo de vínculo → `CLT indeterminado` /
   `estatutário` / `temporário-determinado`; setor → `público` / `privado` / `sem fins`;
   remuneração → `≤1 SM` / `1–5 SM` / `>5 SM`).

2. **Distintividade (lift)** — *"o que é desproporcionalmente comum aqui?"*
   Para CBO/CNAE/UF (alta cardinalidade), calcula-se
   `lift = (share do valor na categoria) / (share do valor na base inteira)`.
   `lift > 1` indica sobre-representação. Filtra-se um **share mínimo de 5%** para não
   capturar valores raros com lift inflado, e pegam-se os 2 valores de maior lift.
   É isso que revela, por exemplo, que a categoria 23 é **7,6× mais "construção de
   edifícios" (CNAE 41)** do que a média.

3. **Numéricas** — média por categoria de `idade`, `tempo_vinculo_meses`,
   `qtd_dias_afastamento`.

**Eficiência:** tudo é feito com `pyarrow.group_by` (agregação em C++, baixa memória) —
uma passada leve por feature lendo só 2 colunas de cada vez, em vez de carregar o
dataset inteiro. Os códigos são então traduzidos pelo **dicionário oficial da RAIS**.

A "persona" final é a leitura conjunta desses indicadores ao longo das categorias
**ordenadas por risco** — como os perfis mudam do menor para o maior risco.

---

## Legenda dos códigos (dicionário RAIS)
- **CBO grande grupo (`cbo1`):** `0`=Forças Armadas · `2`=profissionais de nível superior ·
  `4`=trab. de serviços administrativos · `5`=serviços/vendedores do comércio ·
  `6`=agropecuários · `7`=operários da produção/**construção** industrial
- **CNAE divisão (`cnae2`):** `84`=Administração pública · `85`=Educação · `86`=Saúde ·
  `78`=Agências de **trabalho temporário** · `47`=Comércio varejista · `46`=atacado ·
  `82`=Serviços de apoio/escritório (limpeza, terceirização) · `56`=Alimentação
  (bares/restaurantes) · `10`=Indústria de alimentos · **`41`=Construção de edifícios ·
  `42`=Obras de infraestrutura · `43`=Serviços p/ construção**
- **Remuneração** em salários mínimos (SM); **tamanho** = nº de empregados do estabelecimento.

---

## A história em uma frase

O risco sobe ao longo de um eixo claro: **setor público estatutário (estável) → CLT
consolidado na indústria/saúde → comércio e serviços em pequenas empresas → construção
civil em micro/pequenas construtoras**. Em paralelo, o **tempo de vínculo cai de ~16 anos
para ~2 anos**, o **setor migra de público para privado**, o **porte cai de grande para
micro** e a **escolaridade de superior para fundamental**.

---

## Perfil resumido por categoria

| cat | taxa_y | prob_média | idade | tempo (anos) | CLT indet | temp/det | estatut | público | Simples | superior | micro/peq | CBO/CNAE distintivos |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0,6% | 0,2% | 46 | 15,9 | 12% | 0% | **87%** | **89%** | 2% | 55% | 5% | militares; adm. pública (84) |
| 2 | 1,4% | 0,6% | 47 | 14,6 | 26% | 1% | **63%** | 79% | 1% | 52% | 8% | nível superior; adm.púb./educ. |
| 3 | 3,0% | 2,3% | 38 | 7,8 | 41% | 21% | 23% | 41% | 3% | 40% | 12% | agências temporárias (78), adm.púb. |
| 4 | 4,1% | 4,5% | 37 | 6,2 | 67% | 18% | 7% | 15% | 11% | 34% | 23% | ind. alimentos (10), saúde (86) |
| 5 | 5,4% | 5,7% | 37 | 5,5 | 71% | 16% | 5% | 14% | 12% | 32% | 26% | ind. alimentos/saúde |
| 6 | 6,6% | 7,0% | 37 | 4,7 | 74% | 14% | 3% | 12% | 16% | 30% | 30% | ind. alimentos/saúde |
| 7 | 8,1% | 8,6% | 37 | 3,6 | 76% | 14% | 2% | 9% | 21% | 26% | 37% | administrativo; varejo (47) |
| 8 | 9,6% | 10,3% | 36 | 3,0 | 78% | 15% | 2% | 8% | 24% | 23% | 42% | comércio; atacado (46), apoio (82) |
| 9 | 11,0% | 11,9% | 37 | 2,6 | 80% | 14% | 2% | 7% | 27% | 21% | 45% | serviços de apoio (82) |
| 10 | 12,7% | 13,5% | 37 | 2,4 | 82% | 13% | 2% | 6% | 30% | 19% | 49% | serviços de apoio (82) |
| 11 | 14,5% | 15,3% | 36 | 2,1 | 84% | 12% | 2% | 5% | 33% | 17% | 53% | comércio/serviços (82) |
| 12 | 16,4% | 16,9% | 36 | 2,0 | 84% | 11% | 2% | 5% | 35% | 15% | 57% | alimentação (56), varejo |
| 13 | 18,3% | 18,6% | 36 | 1,8 | 84% | 11% | 3% | 5% | 38% | 14% | 60% | alimentação (56), varejo (47) |
| 14 | 20,4% | 20,4% | 36 | 1,7 | 84% | 10% | 3% | 6% | 39% | 13% | 62% | alimentação (56) |
| 15 | 22,9% | 22,6% | 36 | 1,6 | 83% | 10% | 4% | 7% | 40% | 13% | 64% | produção (7); alimentação (56) |
| 16 | 26,1% | 25,6% | 36 | 1,6 | 83% | 10% | 5% | 8% | 39% | 13% | 65% | produção; **construção (43,41)** |
| 17 | 29,1% | 29,2% | 36 | 1,6 | 82% | 9% | 6% | 9% | 35% | 14% | 64% | produção; **construção (41,42)** |
| 18 | 31,9% | 34,0% | 37 | 1,8 | 83% | 8% | 6% | 10% | 31% | 15% | 61% | **construção (42,41) lift ~5** |
| 19 | 35,4% | 40,4% | 37 | 1,9 | 86% | 6% | 7% | 9% | 31% | 16% | 62% | **construção (42,41) lift ~6** |
| 20 | 39,9% | 47,6% | 37 | 1,9 | 86% | 4% | 9% | 10% | 30% | 16% | 68% | **construção (42,41)** |
| 21 | 46,1% | 59,1% | 37 | 1,8 | 87% | 3% | 10% | 11% | 30% | 19% | 77% | **construção (42,41)** |
| 22 | 57,1% | 70,1% | 37 | 1,9 | **93%** | 1% | 6% | 6% | 50% | 15% | **90%** | **construção (42,41)** |
| 23 | **66,7%** | 80,9% | 38 | 2,2 | **98%** | 0% | 1% | 2% | 54% | 8% | **98%** | **construção de edifícios (41) lift 7,6** |

*(taxa_y = % realmente desligado sem justa causa em 2023; prob_média = previsão do modelo.)*

---

## Personas

**🟢 Risco mínimo — o serviço público (cat 1–2)**
- **Cat 1** (0,6% · lift 0,05×): **O servidor público concursado.** 87% estatutários,
  89% setor público, administração pública/defesa (CNAE 84, lift 5,2); militares
  fortemente sobre-representados (lift 10,6). 46 anos, ~16 anos de casa, nível superior.
- **Cat 2** (1,4%): **Servidor/profissional do setor público** (educação e administração),
  62% estatutário, profissionais de nível superior, 47 anos, ~15 anos de casa.

**🟢 Risco baixo — CLT consolidado (cat 3–6)**
- **Cat 3** (3,0%): **Transição** — mistura de estatutário (23%), CLT (41%) e
  **temporário via agências de mão de obra** (CNAE 78, lift 2,5). ~8 anos de vínculo.
- **Cat 4–6** (4–6,6%): **CLT estável na indústria de alimentos (CNAE 10) e saúde (86)**,
  67–74% CLT indeterminado, empresa média/grande, 5–6 anos de casa, ensino médio/superior.

**🟡 Risco médio-baixo — comércio e serviços (cat 7–11)**
- **Cat 7–8** (8–9,6%): **CLT do comércio (varejo, atacado) e administrativo**, empresa
  pequena, Simples subindo, 3–4 anos.
- **Cat 9–11** (11–14,5%): **CLT recente em serviços de apoio/terceirização** (limpeza,
  escritório – CNAE 82) e comércio, empresa pequena/Simples (27–33%), 2–2,6 anos, jovem.

**🟡 Risco médio — alimentação, varejo e início da construção (cat 12–17)**
- **Cat 12–14** (16–20%): **Trabalhador de bares/restaurantes (CNAE 56) e varejo**,
  vínculo curto (~2 anos), empresa pequena.
- **Cat 15–17** (23–29%): **Operário da produção e início da construção civil**
  (CNAE 41/43 lift 2,4–3,5; cbo1=7 em 30–35%), escolaridade caindo (ensino fundamental
  subindo), ~1,5 ano de casa.

**🔴 Risco alto — a construção civil (cat 18–23)**
- **Cat 18–21** (32–46%): **Operário da construção civil** — edifícios e obras de
  infraestrutura (CNAE 41/42, lift 5–6), micro/pequena empresa, ensino fundamental
  (12–13%). Os **temporários quase somem** aqui (8%→3%): não é contrato temporário, é
  **CLT "permanente" num setor de rotatividade estrutural**.
- **Cat 22** (57% · micro 90%): **Operário de obra em microempresa**, 93% CLT
  indeterminado, 50% optante do Simples.
- **Cat 23** (66,7% · lift 5,0×): **O perfil de MAIOR risco** — operário da **construção
  de edifícios** (CNAE 41 lift **7,6**), 98% CLT indeterminado, micro construtora (98%),
  54% Simples, ensino fundamental (14%; superior só 8%), remuneração baixa (22% ≤1 SM).
  **2 em cada 3 são desligados sem justa causa no ano.**

---

## Insights

1. **As duas âncoras são institucionais:** servidores públicos estatutários (estabilidade
   legal) no fundo do risco; operários da construção civil em micro construtoras no topo.
2. **O modelo capta rotatividade estrutural, não só contrato temporário:** no topo, os
   vínculos são majoritariamente **CLT indeterminado** — a construção demite muito sem
   justa causa mesmo em contratos "permanentes". Os temporários concentram-se no risco
   médio-baixo.
3. **Os eixos batem com a importância das features** do modelo: tempo de vínculo (16→2
   anos), tipo de vínculo (estatutário→CLT), porte (grande→micro), setor (público→
   construção) e escolaridade (superior→fundamental).

> ⚠️ **Ética/LGPD:** estas personas são descritivas e agregadas; não devem ser usadas
> para decisão automatizada sobre indivíduos sem revisão humana e cuidado com vieses
> (setor, escolaridade, região). Ver seção de LGPD no README.
