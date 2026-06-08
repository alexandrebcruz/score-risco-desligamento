"""Gera os notebooks .ipynb do projeto a partir de células definidas em Python.

Executar uma vez: `python _build_notebooks.py`. Pode ser apagado depois.
Mantém o conteúdo dos notebooks versionável/editável de forma legível.
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).parent / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text}


# Preâmbulo comum: adiciona a raiz do projeto ao sys.path para importar src.
PREAMBULO = (
    "# Preâmbulo: torna o pacote src importável a partir do notebook\n"
    "import sys, pathlib\n"
    "ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path.cwd()\n"
    "sys.path.insert(0, str(ROOT))\n"
    "import pandas as pd, numpy as np\n"
    "from src.config import load_config, anos_validos\n"
    "cfg = load_config()\n"
    "print('Raiz:', cfg['root'], '| modo sintético:', cfg['synthetic_mode'])"
)


def write_nb(name, cells):
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                       "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = NB_DIR / name
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", path.name)


# ===========================================================================
# 00 — Setup e dicionários
# ===========================================================================
write_nb("00_setup_e_dicionarios.ipynb", [
    md("# 00 · Setup e Dicionários de Variáveis\n"
       "\n"
       "**Objetivo:** validar o ambiente, carregar a configuração central e materializar\n"
       "as tabelas de-para (motivos de desligamento, escolaridade) em `data/dicts/`.\n"
       "\n"
       "**Entradas:** `config.yaml`.  \n"
       "**Saídas:** `data/dicts/*.csv` (dicionários/de-para).\n"
       "\n"
       "**Premissas:** códigos CBO 2002 e CNAE 2.0 estáveis no período.  \n"
       "**Decisões:** motivos crus RAIS/CAGED unificados em 6 categorias.  \n"
       "**Limitações:** mudanças de layout entre anos exigem revisar os de-para (TODOs)."),
    code(PREAMBULO),
    code("# De-para de motivos e escolaridade (definidos em src/cleaning.py)\n"
         "from src.cleaning import MAPA_MOTIVO, MAPA_ESCOLARIDADE\n"
         "dicts_dir = cfg['abs']['dicts']\n"
         "\n"
         "df_motivo = pd.DataFrame(\n"
         "    [{'codigo_cru': k, 'categoria_unificada': v} for k, v in MAPA_MOTIVO.items()])\n"
         "df_motivo.to_csv(dicts_dir / 'depara_motivo.csv', index=False)\n"
         "\n"
         "df_escol = pd.DataFrame(\n"
         "    [{'grau_instrucao': k, 'faixa_canonica': v} for k, v in MAPA_ESCOLARIDADE.items()])\n"
         "df_escol.to_csv(dicts_dir / 'depara_escolaridade.csv', index=False)\n"
         "\n"
         "print('Categorias de motivo configuradas:', cfg['motivos'])\n"
         "display(df_motivo)"),
    md("### TODOs para dados reais\n"
       "- [ ] Baixar e versionar o **layout RAIS Vínculos** (PDET) em `data/dicts/`.\n"
       "- [ ] Baixar o **layout do Novo CAGED** (PDET).\n"
       "- [ ] Obter as tabelas oficiais **CBO 2002** e **CNAE 2.0** (níveis 6/4/2 dígitos)\n"
       "  para validar os recortes hierárquicos usados em `src/cells.py`."),
])

# ===========================================================================
# 01 — Ingestão / download
# ===========================================================================
write_nb("01_ingestao_download.ipynb", [
    md("# 01 · Ingestão / Download dos Microdados\n"
       "\n"
       "**Objetivo:** obter os microdados RAIS (vínculos) e Novo CAGED para os anos do escopo.\n"
       "\n"
       "**Entradas:** `config.yaml` (anos, URLs, modo sintético).  \n"
       "**Saídas:** arquivos em `data/raw/{RAIS,CAGED}/{ano}/` (parquet).\n"
       "\n"
       "**Premissas:** acesso público estável.  \n"
       "**Decisões:** download idempotente com cache; *fallback* para amostra sintética.  \n"
       "**Limitações:** volume grande (dezenas de GB em 5 anos × 27 UF) — use `ufs_subset`\n"
       "ou `synthetic_mode` para protótipo."),
    code(PREAMBULO),
    code("from src import io_utils\n"
         "raw = cfg['abs']['raw']\n"
         "anos = cfg['anos']\n"
         "\n"
         "if cfg['synthetic_mode']:\n"
         "    # Modo sintético: gera amostras com o schema bruto e salva como parquet.\n"
         "    s = cfg['synthetic']\n"
         "    for ano in anos:\n"
         "        rais = io_utils.gen_rais_ano(ano, s['n_vinculos_por_ano'], s['seed'])\n"
         "        caged = io_utils.gen_caged_ano(ano, s['n_movs_caged_por_ano'], s['seed'])\n"
         "        (raw / 'RAIS' / str(ano)).mkdir(parents=True, exist_ok=True)\n"
         "        (raw / 'CAGED' / str(ano)).mkdir(parents=True, exist_ok=True)\n"
         "        rais.to_parquet(raw / 'RAIS' / str(ano) / 'rais.parquet', index=False)\n"
         "        caged.to_parquet(raw / 'CAGED' / str(ano) / 'caged.parquet', index=False)\n"
         "        print(f'[sintético] {ano}: RAIS={len(rais):,} CAGED={len(caged):,}')\n"
         "else:\n"
         "    # Modo REAL: baixa os vínculos RAIS por região do FTP do PDET/MTE.\n"
         "    # Arquivos: {rais_base}/{ano}/RAIS_VINC_PUB_<REGIAO>.7z  (CSV ',' latin-1).\n"
         "    base = cfg['urls']['rais_base']\n"
         "    regioes = cfg['rais']['regioes']\n"
         "    for ano in anos:\n"
         "        for reg in regioes:\n"
         "            nome = f'RAIS_VINC_PUB_{reg}.7z'\n"
         "            url = f'{base}/{ano}/{nome}'\n"
         "            dest = raw / 'RAIS' / str(ano) / nome\n"
         "            print(f'[real] baixando {ano}/{reg} ...', flush=True)\n"
         "            io_utils.download_ftp(url, dest)            # idempotente (cache)\n"
         "            print(f'        ok {dest.name} ({dest.stat().st_size/1e6:.0f} MB)', flush=True)\n"
         "    print('Download RAIS concluído. (A extração ocorre no nb02, sob demanda.)')\n"
         "    # CAGED real (componente recente) — opcional; o cálculo de taxas usa a RAIS.\n"
         "    # TODO: decodificar o layout do Novo CAGED (CAGEDMOV{AAAAMM}.7z) se for usar fluxos."),
    code("# Conferência rápida do que foi gravado\n"
         "if cfg['synthetic_mode']:\n"
         "    for fonte in ['RAIS', 'CAGED']:\n"
         "        files = sorted((raw / fonte).rglob('*.parquet'))\n"
         "        print(fonte, '->', len(files), 'arquivo(s)')\n"
         "        if files: display(pd.read_parquet(files[0]).head(3))\n"
         "else:\n"
         "    z = sorted((raw / 'RAIS').rglob('*.7z'))\n"
         "    print(len(z), 'arquivos .7z baixados:')\n"
         "    for c in z:\n"
         "        print(' ', c.relative_to(raw), f'-> {c.stat().st_size/1e6:.0f} MB')"),
])

# ===========================================================================
# 02 — Limpeza e padronização
# ===========================================================================
write_nb("02_limpeza_padronizacao.ipynb", [
    md("# 02 · Limpeza e Padronização\n"
       "\n"
       "**Objetivo:** ler os brutos, padronizar tipos e harmonizar RAIS×CAGED num schema comum.\n"
       "\n"
       "**Entradas:** `data/raw/...`.  \n"
       "**Saídas:** `data/interim/rais.parquet` e `data/interim/caged.parquet`.\n"
       "\n"
       "**Decisões:** motivos crus → categorias unificadas; escolaridade/idade em faixas canônicas.  \n"
       "**Limitações:** pequenas incompatibilidades de categorização entre fontes são aproximadas;\n"
       "registros inconsistentes são *logados e quantificados*, não descartados em silêncio."),
    code(PREAMBULO),
    code("import pyarrow as pa, pyarrow.parquet as pq\n"
         "from src import io_utils, cleaning\n"
         "raw, interim = cfg['abs']['raw'], cfg['abs']['interim']\n"
         "out_root = interim / 'rais'   # interim particionado: rais/ano=YYYY/<fonte>.parquet\n"
         "out_root.mkdir(parents=True, exist_ok=True)\n"
         "ufs = cfg.get('ufs_subset')\n"
         "\n"
         "def escreve_particao(dest, chunks):\n"
         "    '''Escreve chunks limpos num parquet via ParquetWriter (memória baixa).'''\n"
         "    writer = None; n = 0\n"
         "    for ch in chunks:\n"
         "        t = pa.Table.from_pandas(ch, preserve_index=False)\n"
         "        if writer is None:\n"
         "            writer = pq.ParquetWriter(dest, t.schema)\n"
         "        writer.write_table(t); n += len(ch)\n"
         "    if writer: writer.close()\n"
         "    return n"),
    code("# Ingestão incremental -> interim canônico particionado por ano.\n"
         "# Modo REAL: extrai cada .7z, lê em chunks, limpa, grava parquet e APAGA o\n"
         "# .COMT/.txt (poupa disco). Idempotente: pula partições já gravadas.\n"
         "for ano in cfg['anos']:\n"
         "    pdir = out_root / f'ano={ano}'\n"
         "    pdir.mkdir(parents=True, exist_ok=True)\n"
         "    if cfg['synthetic_mode']:\n"
         "        for pq_in in sorted((raw / 'RAIS' / str(ano)).glob('*.parquet')):\n"
         "            dest = pdir / 'synthetic.parquet'\n"
         "            if dest.exists(): continue\n"
         "            rais = cleaning.clean_rais(pd.read_parquet(pq_in))\n"
         "            n = escreve_particao(dest, [rais]); print(f'{ano} sintético: {n:,}')\n"
         "    else:\n"
         "        for z in sorted((raw/'RAIS'/str(ano)).glob('*.7z')):\n"
         "            regiao = z.stem.replace('RAIS_VINC_PUB_', '')\n"
         "            dest = pdir / f'{regiao}.parquet'\n"
         "            if dest.exists(): print(f'{ano}/{regiao}: já processado'); continue\n"
         "            print(f'{ano}/{regiao}: extraindo ...', flush=True)\n"
         "            extraidos = io_utils.extract_7z(z, raw/'RAIS'/str(ano))\n"
         "            arq = next(p for p in extraidos if p.suffix.upper() in ('.COMT', '.TXT'))\n"
         "            print(f'{ano}/{regiao}: lendo+limpando {arq.name} ...', flush=True)\n"
         "            n = escreve_particao(dest, cleaning.iter_rais_clean_chunks(arq, ano, ufs))\n"
         "            arq.unlink()   # libera o extraído (mantém o .7z como cache de download)\n"
         "            print(f'{ano}/{regiao}: {n:,} vínculos -> {dest.name} (extraído removido)')\n"
         "print('Interim particionado em', out_root)"),
    code("# Conferência: linhas e distribuição de motivos sobre o interim (amostra leve)\n"
         "files = sorted(out_root.rglob('*.parquet'))\n"
         "print('partições:', len(files))\n"
         "tot = 0; vc = None\n"
         "for f in files:\n"
         "    s = pd.read_parquet(f, columns=['motivo_unificado'])\n"
         "    tot += len(s)\n"
         "    vc = s['motivo_unificado'].value_counts() if vc is None else vc.add(\n"
         "         s['motivo_unificado'].value_counts(), fill_value=0)\n"
         "print('Total de vínculos:', f'{tot:,}')\n"
         "display((vc / vc.sum()).round(4).sort_values(ascending=False))"),
])

# ===========================================================================
# 03 — Células e taxas
# ===========================================================================
write_nb("03_celulas_e_taxas.ipynb", [
    md("# 03 · Células de Perfil e Taxas de Desligamento\n"
       "\n"
       "**Objetivo:** binning das dimensões, construção das células e cálculo das taxas por\n"
       "motivo em cada nível de backoff, com suavização Empirical Bayes.\n"
       "\n"
       "**Entradas:** `data/interim/rais/ano=*/**.parquet` (interim particionado).  \n"
       "**Saídas:** `data/processed/rates/level_*.parquet` + `levels.json`;\n"
       "`data/interim/val_<ano>.parquet` (agregado por ano p/ validação no nb06).\n"
       "\n"
       "**Decisões centrais:**\n"
       "- **Agregação incremental (map-reduce):** lê cada partição em lotes e SOMA as\n"
       "  contagens por célula (`rates.Accumulator`). Nunca carrega tudo em memória —\n"
       "  escala para cobertura nacional × múltiplos anos com RAM limitada.\n"
       "- Exposição = nº de vínculos observados na célula no ano (denominador de risco).\n"
       "- Taxa anual por motivo = desligamentos do motivo / exposição.\n"
       "- Backoff hierárquico (`src/cells.py`) do mais geral ao mais específico.\n"
       "- Shrinkage Beta-Binomial aninhado resolve células pouco populosas.\n"
       "\n"
       "**Limitações:** viés de composição; taxa é risco *médio* da célula, não individual."),
    code(PREAMBULO),
    code("import pyarrow.parquet as pq\n"
         "from src import binning, cells, rates\n"
         "interim = cfg['abs']['interim']\n"
         "motivos = cfg['motivos']\n"
         "anos_ok = set(anos_validos(cfg))   # respeita exclusão de anos atípicos\n"
         "\n"
         "# Nível usado na validação temporal (populoso e estável)\n"
         "VAL_LVL = 'cbo2_cnae2_tempo_uf'\n"
         "val_cols = [l['cols'] for l in cells.BACKOFF_LEVELS if l['name']==VAL_LVL][0]\n"
         "\n"
         "niveis = cells.active_levels(cfg)           # exclui níveis ultra-granulares (config)\n"
         "print('níveis materializados:', [l['name'] for l in niveis])\n"
         "acc = rates.Accumulator(motivos, levels=niveis)   # acumulador global (todos os anos)\n"
         "val_by_year = {}                            # acumulador do nível de validação por ano\n"
         "files = sorted((interim / 'rais').rglob('*.parquet'))\n"
         "print('partições a processar:', len(files))"),
    code("# Passe incremental: lote a lote, sem materializar os microdados inteiros\n"
         "for f in files:\n"
         "    ano = int(f.parent.name.split('=')[1])\n"
         "    if ano not in anos_ok: continue\n"
         "    pf = pq.ParquetFile(f)\n"
         "    for batch in pf.iter_batches(batch_size=3_000_000):\n"
         "        df = batch.to_pandas()\n"
         "        df = cells.add_cell_keys(binning.add_bins(df, cfg))\n"
         "        acc.add(df)                                       # soma em todos os níveis\n"
         "        v = rates.count_single_level(df, val_cols, motivos)\n"
         "        val_by_year[ano] = rates._sum_two_tables(val_by_year.get(ano), v, val_cols, motivos)\n"
         "    print(f'  ok {f.parent.name}/{f.name}', flush=True)\n"
         "tables = acc.tables()\n"
         "for name, tab in tables.items():\n"
         "    print(f'{name:32s} células={len(tab):>9,}  exposição_total={int(tab[\"n\"].sum()):,}')"),
    code("# Persistir tabelas de taxa (scoring) + agregados por ano (validação)\n"
         "rates.save_level_tables(tables, motivos, cfg['abs']['rates'])\n"
         "for ano, v in val_by_year.items():\n"
         "    v.to_parquet(interim / f'val_{ano}.parquet', index=False)\n"
         "print('Tabelas salvas em', cfg['abs']['rates'], '| anos validação:', sorted(val_by_year))\n"
         "\n"
         "# Espiar o nível mais específico com a taxa involuntária bruta (k/n)\n"
         "comp = tables['completo'].copy()\n"
         "comp['taxa_sjc_bruta'] = comp['k_involuntario_sjc'] / comp['n']\n"
         "display(comp.sort_values('n', ascending=False).head(8))"),
    md("> **Nota metodológica:** a taxa *bruta* (k/n) acima é apenas diagnóstica.\n"
       "> A taxa usada no score é a versão **suavizada** (Empirical Bayes aninhado),\n"
       "> calculada sob demanda em `src/rates.eb_annual_hazard` durante o scoring —\n"
       "> assim cada consulta recua na hierarquia exatamente até onde há suporte estatístico."),
])

# ===========================================================================
# 04 — EDA das taxas
# ===========================================================================
write_nb("04_eda_taxas.ipynb", [
    md("# 04 · Análise Exploratória das Taxas\n"
       "\n"
       "**Objetivo:** explorar as taxas e identificar perfis de maior/menor risco.\n"
       "\n"
       "**Entradas:** `data/processed/rates/level_*.parquet`.  \n"
       "**Saídas:** figuras em `outputs/figures/`, tabelas em `outputs/tables/`.\n"
       "\n"
       "**Limitações:** associações descritivas, não causais."),
    code(PREAMBULO),
    code("from src import rates, viz\n"
         "tables, meta = rates.load_level_tables(cfg['abs']['rates'])\n"
         "fig_dir, tab_dir = cfg['abs']['figures'], cfg['abs']['tables']\n"
         "\n"
         "# Dois recortes: nível RICO (muitas dimensões, células pequenas) para as\n"
         "# marginais ponderadas; nível mais AGREGADO (células populosas) para a\n"
         "# distribuição e o ranking, onde queremos taxas estáveis por célula.\n"
         "lvl_rico, lvl_agg = 'cbo4_cnae2_tempo_tam_uf', 'cbo2_cnae2_tempo_uf'\n"
         "df_rico = tables[lvl_rico].copy()\n"
         "df_rico['taxa_sjc'] = df_rico['k_involuntario_sjc'] / df_rico['n']\n"
         "df_agg = tables[lvl_agg].copy()\n"
         "df_agg['taxa_sjc'] = df_agg['k_involuntario_sjc'] / df_agg['n']\n"
         "exp_min = cfg['suavizacao']['exposicao_minima']\n"
         "print(f'rico={len(df_rico):,} células (n médio {df_rico.n.mean():.0f}) | '\n"
         "      f'agg={len(df_agg):,} células (n médio {df_agg.n.mean():.0f})')"),
    code("# Distribuição das taxas no nível agregado (células com exposição mínima)\n"
         "df_ok = df_agg[df_agg['n'] >= exp_min]\n"
         "fig = viz.plot_rate_distribution(df_ok['taxa_sjc'],\n"
         "       f'Distribuição da taxa anual (dispensa s/ justa causa) — n>={exp_min}')\n"
         "viz.save_fig(fig, fig_dir / 'dist_taxa_sjc.png'); fig"),
    code("# Efeitos marginais por dimensão (média ponderada por exposição no nível rico)\n"
         "for dim in ['tempo_faixa', 'tamanho_faixa', 'uf']:\n"
         "    fig = viz.plot_marginal(df_rico, dim, 'taxa_sjc',\n"
         "            f'Risco involuntário médio por {dim}')\n"
         "    viz.save_fig(fig, fig_dir / f'marginal_{dim}.png')\n"
         "    display(fig)"),
    code("# Ranking de perfis de MAIOR risco (com suporte) -> tabela de output.\n"
         "# Exclui CBO/CNAE não-identificados ('00') e exige exposição mais robusta.\n"
         "rk = df_ok[(df_ok['cbo2'] != '00') & (df_ok['cnae2'] != '00') & (df_ok['n'] >= 200)]\n"
         "rank = (rk.sort_values('taxa_sjc', ascending=False)\n"
         "          .head(25)[['cbo2','cnae2','tempo_faixa','uf','n','taxa_sjc']])\n"
         "rank.to_csv(tab_dir / 'perfis_maior_risco.csv', index=False)\n"
         "display(rank)"),
    code("# Ranking de perfis de MENOR risco (com suporte) -> tabela de output\n"
         "rank_min = (df_ok.sort_values('taxa_sjc', ascending=True)\n"
         "                 .head(25)[['cbo2','cnae2','tempo_faixa','uf','n','taxa_sjc']])\n"
         "rank_min.to_csv(tab_dir / 'perfis_menor_risco.csv', index=False)\n"
         "display(rank_min)"),
    code("# Cobertura do backoff: exposição por nível\n"
         "cob = pd.DataFrame([\n"
         "    {'nivel': name, 'n_celulas': len(t), 'exposicao_total': int(t['n'].sum())}\n"
         "    for name, t in tables.items()])\n"
         "cob.to_csv(tab_dir / 'cobertura_niveis.csv', index=False)\n"
         "display(cob)"),
])

# ===========================================================================
# 05 — Função de scoring
# ===========================================================================
write_nb("05_funcao_scoring.ipynb", [
    md("# 05 · Função de Scoring\n"
       "\n"
       "**Objetivo:** demonstrar `score_pessoa(...)` e `score_lote(...)` de `src/scoring.py`.\n"
       "\n"
       "**Entradas:** `data/processed/rates/...`.  \n"
       "**Saídas:** exemplos de score (em tela) e CSV de lote em `outputs/tables/`.\n"
       "\n"
       "**Decisões:** binning idêntico ao do treino (sem *train/serve skew*);\n"
       "EB aninhado + backoff resolvem células ausentes.  \n"
       "**Limitações:** o score é a taxa da célula, não predição individual calibrada;\n"
       "perfis idênticos recebem score idêntico."),
    code(PREAMBULO),
    code("from src import scoring\n"
         "import json\n"
         "\n"
         "# Pessoa exemplo (atributos brutos)\n"
         "res = scoring.score_pessoa(\n"
         "    cbo='252105', cnae='4711301', uf='PA', idade=33, escolaridade='superior',\n"
         "    tempo_vinculo_meses=8, tamanho_empresa=2,\n"
         "    motivos=['involuntario_sjc'], horizontes=[3, 6, 12])\n"
         "print(json.dumps(res, ensure_ascii=False, indent=2, default=float))"),
    code("# Comparação: tempo de vínculo curto vs longo (espera-se risco maior p/ curto)\n"
         "base = dict(cbo='521110', cnae='4711301', uf='PA', idade=40,\n"
         "            escolaridade='medio_completo', tamanho_empresa=1)\n"
         "for t in [2, 8, 18, 72]:\n"
         "    r = scoring.score_pessoa(**base, tempo_vinculo_meses=t)\n"
         "    r12 = r['risco']['involuntario_sjc'][12]\n"
         "    print(f'tempo={t:>3}m -> risco 12m (sjc) = {r12:.3f} | nível={r[\"nivel_usado\"][\"involuntario_sjc\"]}')"),
    code("# Score em lote\n"
         "pessoas = pd.DataFrame([\n"
         "    dict(cbo='252105', cnae='4711301', uf='PA', idade=33, escolaridade='superior',\n"
         "         tempo_vinculo_meses=8, tamanho_empresa=2),\n"
         "    dict(cbo='715210', cnae='4120400', uf='AM', idade=51, escolaridade='fundamental',\n"
         "         tempo_vinculo_meses=40, tamanho_empresa=4),\n"
         "    dict(cbo='621005', cnae='0111301', uf='TO', idade=27, escolaridade='medio_completo',\n"
         "         tempo_vinculo_meses=3, tamanho_empresa=0),\n"
         "])\n"
         "scored = scoring.score_lote(pessoas, motivos=['involuntario_sjc','fim_contrato'])\n"
         "scored.to_csv(cfg['abs']['tables'] / 'exemplo_score_lote.csv', index=False)\n"
         "display(scored)"),
])

# ===========================================================================
# 06 — Validação / sanidade
# ===========================================================================
write_nb("06_validacao_sanidade.ipynb", [
    md("# 06 · Validação e Sanidade do Score\n"
       "\n"
       "**Objetivo:** checar estabilidade temporal, calibração agregada e monotonicidades.\n"
       "Não há rótulo individual → validação **agregada** (consistência interna), não acurácia individual.\n"
       "\n"
       "**Entradas:** `data/interim/val_<ano>.parquet` (agregados por ano do nb03).  \n"
       "**Saídas:** `outputs/tables/validacao*.csv`, figuras.\n"
       "\n"
       "**Limitações:** sem *ground truth* individual; confirma estabilidade/consistência, não acurácia."),
    code(PREAMBULO),
    code("import numpy as np\n"
         "from src import cells, rates, viz\n"
         "interim = cfg['abs']['interim']\n"
         "motivos = cfg['motivos']\n"
         "VAL_LVL = 'cbo2_cnae2_tempo_uf'\n"
         "cols = [l['cols'] for l in cells.BACKOFF_LEVELS if l['name']==VAL_LVL][0]\n"
         "anos = sorted(int(f.stem.split('_')[1]) for f in interim.glob('val_*.parquet'))\n"
         "assert len(anos) >= 2, 'Validação temporal requer >=2 anos (ajuste cfg.anos).'\n"
         "treino_anos, holdout_ano = anos[:-1], anos[-1]\n"
         "print('Treino:', treino_anos, '| Holdout:', holdout_ano)"),
    code("# Estabilidade temporal: taxa (treino) vs taxa observada (holdout) por célula,\n"
         "# a partir dos agregados por ano (nível populoso VAL_LVL).\n"
         "load = lambda a: pd.read_parquet(interim / f'val_{a}.parquet')\n"
         "tr = None\n"
         "for a in treino_anos:\n"
         "    tr = rates._sum_two_tables(tr, load(a), cols, motivos)\n"
         "ho = load(holdout_ano)\n"
         "def add_taxa(df):\n"
         "    df = df.copy(); df['taxa'] = df['k_involuntario_sjc'] / df['n']; return df\n"
         "tr, ho = add_taxa(tr), add_taxa(ho)\n"
         "merged = tr.merge(ho, on=cols, suffixes=('_tr','_ho'))\n"
         "merged = merged[merged['n_tr'] >= cfg['suavizacao']['exposicao_minima']]\n"
         "corr = merged['taxa_tr'].corr(merged['taxa_ho'])\n"
         "mae = (merged['taxa_tr']-merged['taxa_ho']).abs().mean()\n"
         "print(f'Estabilidade célula a célula: corr={corr:.3f}  MAE={mae:.4f}  (n_células={len(merged):,})')"),
    code("# Calibração agregada: agrupa células por decil de taxa prevista (treino)\n"
         "merged['decil'] = pd.qcut(merged['taxa_tr'], 10, duplicates='drop')\n"
         "calib = merged.groupby('decil', observed=True).apply(\n"
         "    lambda d: pd.Series({'prevista': np.average(d['taxa_tr'], weights=d['n_tr']),\n"
         "                         'observada': np.average(d['taxa_ho'], weights=d['n_ho'])}))\n"
         "calib.to_csv(cfg['abs']['tables'] / 'validacao_calibracao.csv')\n"
         "fig, ax = __import__('matplotlib.pyplot', fromlist=['x']).subplots(figsize=(5,5))\n"
         "ax.plot([0, calib['prevista'].max()], [0, calib['prevista'].max()], '--', color='gray')\n"
         "ax.scatter(calib['prevista'], calib['observada'])\n"
         "ax.set_xlabel('Taxa prevista (treino)'); ax.set_ylabel('Taxa observada (holdout)')\n"
         "ax.set_title('Calibração agregada por decil'); viz.save_fig(fig, cfg['abs']['figures']/'calibracao.png'); fig"),
    code("# Sanidade de monotonicidade: risco deve cair com o tempo de vínculo\n"
         "from src import scoring\n"
         "base = dict(cbo='521110', cnae='4711301', uf='PA', idade=40,\n"
         "            escolaridade='medio_completo', tamanho_empresa=1)\n"
         "riscos = [scoring.score_pessoa(**base, tempo_vinculo_meses=t)['risco']['involuntario_sjc'][12]\n"
         "          for t in [2, 8, 18, 72]]\n"
         "print('Riscos 12m por tempo [2,8,18,72]m:', [round(x,3) for x in riscos])\n"
         "print('Monotonicamente não-crescente?', all(a>=b-1e-9 for a,b in zip(riscos, riscos[1:])))"),
    code("# Sensibilidade ao shrinkage (m) — reusa UM único Scorer (memória),\n"
         "# variando m diretamente em rates.eb_annual_hazard.\n"
         "from src.scoring import Scorer\n"
         "sc = Scorer(cfg=cfg)\n"
         "keys = sc._keys_from_attrs(cbo='223505', cnae='8610101', uf='AM', idade=62,\n"
         "                           escolaridade='superior', tempo_vinculo_meses=5, tamanho_empresa=0)\n"
         "mot = 'involuntario_sjc'\n"
         "for m in [10, 50, 200]:\n"
         "    haz, nivel, n = rates.eb_annual_hazard(sc.indexes, sc.meta, keys, mot, m)\n"
         "    print(f'm={m:>3} -> risco12m={rates.horizon_risk(haz,12):.3f} nível={nivel} exp={n:,}')"),
])

# ===========================================================================
# 07 — Ajuste conjuntural com o Novo CAGED
# ===========================================================================
write_nb("07_ajuste_caged.ipynb", [
    md("# 07 · Ajuste Conjuntural com o Novo CAGED\n"
       "\n"
       "**Objetivo:** complementar o risco *estrutural* da RAIS com a *conjuntura recente*\n"
       "dos fluxos mensais do Novo CAGED, via um **fator de ajuste** por célula\n"
       "L = (CBO 2díg × CNAE 2díg × UF).\n"
       "\n"
       "**Entradas:** `data/processed/rates/level_completo.parquet` (estoque RAIS) e\n"
       "microdados Novo CAGED (`config.caged`).  \n"
       "**Saídas:** `data/processed/caged_fator_<motivo>.parquet`; figura/tabela em `outputs/`.\n"
       "\n"
       "**Método:** a RAIS não tem estoque infra-anual; o CAGED não tem estoque nem tempo\n"
       "de vínculo. Combinamos: `hazard_recente_L = (desligamentos_CAGED / meses)·12 / estoque_RAIS_L`,\n"
       "e `fator_L = hazard_recente_L / hazard_estrutural_L` (suavizado p/ 1 em células ralas).\n"
       "No scoring, `risco_ajustado = risco_estrutural × fator_L`.\n"
       "\n"
       "**Premissas/Limitações:** o ajuste é *agregado* em L (não usa tempo de vínculo,\n"
       "que o CAGED não traz); o mapa de `tipomovimentação` aproxima os motivos; UF e\n"
       "CNAE/CBO devem ser comparáveis entre RAIS e CAGED (CNAE 2.0 subclasse, CBO 2002)."),
    code(PREAMBULO),
    code("from src import io_utils, caged, rates, scoring\n"
         "raw = cfg['abs']['raw']; motivos = cfg['motivos']; ufs = cfg.get('ufs_subset')\n"
         "ano = cfg['caged']['ano']; meses = cfg['caged']['meses']\n"
         "base = cfg['urls']['caged_base']\n"
         "\n"
         "# Baixa, extrai, conta desligamentos e descarta o .txt — mês a mês.\n"
         "acc = None; meses_proc = 0\n"
         "for mm in meses:\n"
         "    comp = f'{ano}{mm:02d}'\n"
         "    z = raw / 'CAGED' / str(ano) / f'CAGEDMOV{comp}.7z'\n"
         "    url = f'{base}/{ano}/{comp}/CAGEDMOV{comp}.7z'\n"
         "    try:\n"
         "        io_utils.download_ftp(url, z)\n"
         "    except Exception as e:\n"
         "        print(f'  {comp}: indisponível ({e}); pulando'); continue\n"
         "    extr = io_utils.extract_7z(z, raw / 'CAGED' / str(ano))\n"
         "    txt = next(p for p in extr if p.suffix.lower() == '.txt')\n"
         "    tab, _ = caged.count_caged_deslig([txt], motivos, ufs)\n"
         "    txt.unlink()\n"
         "    acc = tab if acc is None else (pd.concat([acc, tab])\n"
         "          .groupby(['cbo2','cnae2','uf'], as_index=False).sum())\n"
         "    meses_proc += 1\n"
         "    print(f'  {comp}: ok (acumulado: {len(acc):,} células L)', flush=True)\n"
         "print('meses processados:', meses_proc)"),
    code("# Estoque estrutural RAIS por L=(cbo2,cnae2,uf) a partir do nível mais\n"
         "# granular disponível que contenha cnae2 (robusto à exclusão de níveis).\n"
         "tables, meta = rates.load_level_tables(cfg['abs']['rates'])\n"
         "fonte_L = 'cbo2_cnae2_tempo_uf' if 'cbo2_cnae2_tempo_uf' in tables else meta['order'][-1]\n"
         "rais_L = caged.rais_estoque_por_L(tables[fonte_L], motivos)\n"
         "print('células L na RAIS:', f'{len(rais_L):,}')\n"
         "\n"
         "# Fator conjuntural para o motivo default (dispensa s/ justa causa)\n"
         "MOT = cfg['motivo_default'][0]\n"
         "fat = caged.fator_ajuste_conjuntural(acc, meses_proc, rais_L, MOT,\n"
         "                                     m_suav=cfg['caged']['shrinkage_fator'],\n"
         "                                     n_anos_rais=len(cfg['anos']))\n"
         "fat[['cbo2','cnae2','uf','fator']].to_parquet(\n"
         "    cfg['abs']['processed'] / f'caged_fator_{MOT}.parquet', index=False)\n"
         "print('fator salvo. Resumo do fator (peso conjuntural):')\n"
         "display(fat['fator'].describe().round(3))"),
    code("# Onde a conjuntura recente mais AGRAVOU o risco vs a estrutura (fator alto)\n"
         "alto = fat[fat['n'] >= 500].sort_values('fator', ascending=False).head(12)\n"
         "alto.to_csv(cfg['abs']['tables'] / 'caged_fatores_top.csv', index=False)\n"
         "display(alto[['cbo2','cnae2','uf','hazard_estrut','hazard_recente','fator','n']].round(4))"),
    code("# Demonstração: score SEM vs COM ajuste conjuntural (recarrega fatores)\n"
         "sc = scoring.Scorer(cfg=cfg)   # nova instância já carrega caged_fator_*\n"
         "exemplo = dict(cbo='521110', cnae='4711301', uf='SP', idade=35,\n"
         "               escolaridade='medio_completo', tempo_vinculo_meses=8, tamanho_empresa=3)\n"
         "for ajuste in (False, True):\n"
         "    r = sc.score_pessoa(**exemplo, motivos=[cfg['motivo_default'][0]], ajuste_conjuntural=ajuste)\n"
         "    m = cfg['motivo_default'][0]\n"
         "    print(f'ajuste={ajuste!s:5} -> risco12m={r[\"risco\"][m][12]:.3f} '\n"
         "          f'fator={r[\"fator_conjuntural\"][m]:.3f} hazard={r[\"hazard_anual\"][m]:.3f}')"),
    md("> **Leitura:** `fator > 1` significa que o CAGED recente aponta **mais**\n"
       "> desligamentos involuntários naquela célula L do que a média estrutural da RAIS\n"
       "> (conjuntura piorando); `fator < 1`, o oposto. O ajuste é deliberadamente\n"
       "> conservador (suavizado p/ 1 e limitado ao intervalo [0,2; 5,0])."),
])

write_nb("08_pgfn_lista_empresas.ipynb", [
    md("# 08 · PGFN — Lista de Empresas com Dívida Previdenciária ou FGTS\n\n"
       "**Objetivo:** a partir dos Dados Abertos da Dívida Ativa (PGFN) baixados em\n"
       "`data/raw/pgfn/` (CLI `python -m src.pgfn`), consolidar **uma linha por\n"
       "empresa (CNPJ)** que apareceu como devedora **previdenciária** ou de **FGTS**\n"
       "em qualquer trimestre de 2020 até o mais recente.\n\n"
       "**Saída:** `outputs/tables/pgfn_empresas_devedoras.{csv,parquet}`.\n\n"
       "**Como funciona:** `src.pgfn.agregar_empresas` lê os CSVs **em streaming de\n"
       "dentro dos .zip** (não extrai pro disco), filtra `TIPO_PESSOA == 'Pessoa\n"
       "jurídica'` e agrega por `CPF_CNPJ`.\n\n"
       "**Limitações:** reflete o *estoque* publicado em cada trimestre (se a empresa\n"
       "quitou, some nos trimestres seguintes — por isso olhamos o período todo)."),
    code("# Preâmbulo: torna o pacote src importável a partir do notebook\n"
         "import sys, pathlib\n"
         "ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path.cwd()\n"
         "sys.path.insert(0, str(ROOT))\n"
         "import pandas as pd\n"
         "from src.config import load_config\n"
         "from src import pgfn\n"
         "cfg = load_config()\n"
         "destino = pathlib.Path(cfg.get('pgfn', {}).get('dir', 'data/raw/pgfn'))\n"
         "if not destino.is_absolute():\n"
         "    destino = cfg['root'] / destino\n"
         "print('Lendo de:', destino)"),
    code("# Confere o que foi baixado (um .zip por trimestre x tipo)\n"
         "zips = sorted(destino.rglob('Dados_abertos_*.zip'))\n"
         "print(len(zips), 'arquivos .zip encontrados')\n"
         "for z in zips[:4] + zips[-2:]:\n"
         "    print(' ', z.relative_to(destino), f'{z.stat().st_size/1e6:.0f} MB')"),
    code("# Agrega TODAS as empresas (PJ) com dívida previdenciária ou FGTS no período.\n"
         "# Streaming; leva alguns minutos. Teste rápido: tipos=['FGTS'].\n"
         "linhas = pgfn.agregar_empresas(destino, tipos=['Previdenciario', 'FGTS'], apenas_pj=True)\n"
         "df = pd.DataFrame(linhas)\n"
         "print(f'\\n{len(df):,} empresas únicas com alguma dívida previdenciária ou FGTS')\n"
         "df.head(10)"),
    code("# Visão geral + recorte de dívida EXIGÍVEL (em cobrança) vs suspensa/parcelada\n"
         "print('Com dívida PREVIDENCIÁRIA :', (df.TEVE_PREVIDENCIARIO == 'S').sum())\n"
         "print('Com dívida FGTS          :', (df.TEVE_FGTS == 'S').sum())\n"
         "print('Com AMBAS                :', ((df.TEVE_PREVIDENCIARIO=='S') & (df.TEVE_FGTS=='S')).sum())\n"
         "print('Com dívida EXIGÍVEL      :', (df.DIVIDA_EXIGIVEL == 'S').sum(), '(em cobrança no últ. trim.)')\n"
         "print()\n"
         "print('Soma por situação (R$ bi, snapshot do últ. trim. de cada empresa):')\n"
         "for c in ['VALOR_EM_COBRANCA','VALOR_PARCELADO_BENEF','VALOR_GARANTIDO','VALOR_SUSPENSO_JUD']:\n"
         "    print(f'  {c:<22} {df[c].sum()/1e9:>10,.2f}')"),
    code("# Exemplo: empresas grandes que NÃO têm dívida exigível (tudo parcelado/garantido)\n"
         "cols = ['CPF_CNPJ','NOME_DEVEDOR','UF','VALOR_EM_COBRANCA','VALOR_PARCELADO_BENEF','VALOR_GARANTIDO','DIVIDA_EXIGIVEL']\n"
         "df[df.DIVIDA_EXIGIVEL=='N'].sort_values('VALOR_TOTAL_REF', ascending=False)[cols].head(10)"),
    code("# Salva a lista consolidada\n"
         "out = cfg['abs']['tables']\n"
         "csv_path = out / 'pgfn_empresas_devedoras.csv'\n"
         "df.to_csv(csv_path, index=False, encoding='utf-8')\n"
         "try:\n"
         "    df.to_parquet(out / 'pgfn_empresas_devedoras.parquet', index=False)\n"
         "except Exception as e:\n"
         "    print('(parquet opcional falhou:', e, ')')\n"
         "print('Salvo em:', csv_path)"),
])

# ===========================================================================
# 08 — Benchmark CatBoost vs score agregado
# ===========================================================================
write_nb("08_catboost.ipynb", [
    md("# 08 · CatBoost (supervisionado) vs Score Agregado de Células\n"
       "\n"
       "**Objetivo:** treinar um classificador individual (CatBoost) nos mesmos microdados e\n"
       "compará-lo ao score de células no MESMO holdout temporal e métricas.\n"
       "\n"
       "**Entradas:** `data/interim/rais/ano=*/*.parquet`; tabelas de taxa (score de células).  \n"
       "**Saídas:** `outputs/tables/benchmark_catboost.csv`, figuras de calibração e importância.\n"
       "\n"
       "**Setup:** alvo binário = teve dispensa s/ justa causa no ano; features CBO, CNAE, UF,\n"
       "escolaridade, tamanho (categóricas nativas do CatBoost) + idade, tempo de vínculo.\n"
       "Split TEMPORAL: treino = anos anteriores, holdout = último ano.\n"
       "\n"
       "**Premissas/Limitações:** treina em AMOSTRA (cabe em memória); horizonte = 'dentro do\n"
       "ano' (não 3/6/12m); é modelo individual (contrasta com a premissa agregada do projeto).\n"
       "Sem leakage: vinculo_ativo/mes_deslig/motivo/separado NÃO são features."),
    code(PREAMBULO),
    code("import time; t0=time.time()\n"
         "from src import ml, scoring\n"
         "MOT = cfg['motivo_default'][0]\n"
         "anos = cfg['anos']; treino_anos, holdout = anos[:-1], anos[-1]\n"
         "fit_anos, val_ano = treino_anos[:-1], treino_anos[-1]   # split TEMPORAL p/ early stopping\n"
         "mlp = cfg['ml']\n"
         "print('fit:', fit_anos, '| validação:', val_ano, '| holdout:', holdout, '| alvo:', MOT)\n"
         "fit = ml.sample_microdata(fit_anos, frac=mlp['frac_treino'], seed=mlp['seed'], motivo=MOT, cfg=cfg)\n"
         "val = ml.sample_microdata([val_ano], frac=mlp['frac_treino'], seed=mlp['seed'], motivo=MOT, cfg=cfg)\n"
         "te  = ml.sample_microdata([holdout], frac=mlp['frac_teste'], seed=mlp['seed'], motivo=MOT, cfg=cfg)\n"
         "print(f'fit={len(fit):,} val={len(val):,} holdout={len(te):,} [{time.time()-t0:.0f}s]')"),
    code("# Treino do CatBoost com EARLY STOPPING na validação (2022) — holdout (2023)\n"
         "# fica intocado p/ a métrica final. O nº de árvores é escolhido pela validação.\n"
         "Xf, yf = ml.prepare_xy(fit)\n"
         "Xv, yv = ml.prepare_xy(val)\n"
         "Xte, yte = ml.prepare_xy(te)\n"
         "cb = mlp['catboost']\n"
         "model = ml.train_catboost(Xf, yf, Xv, yv, depth=cb['depth'],\n"
         "                          learning_rate=cb['learning_rate'], iterations=cb['max_iterations'],\n"
         "                          early_stopping_rounds=cb['early_stopping_rounds'], verbose=200)\n"
         "p_cb = model.predict_proba(Xte)[:, 1]\n"
         "print(f'CatBoost treinado | best_iteration={model.get_best_iteration()} (teto {cb[\"max_iterations\"]}) [{time.time()-t0:.0f}s]')"),
    code("# Score de células TREINO-ONLY (sem o holdout 2023) -> comparação JUSTA.\n"
         "# As tabelas de PRODUÇÃO (data/processed/rates, todos os anos) seguem\n"
         "# corretas p/ pontuar pessoas; aqui usamos uma versão out-of-time só p/\n"
         "# avaliar — senão a célula 'veria' o ano de teste (vazamento).\n"
         "import pandas as pd\n"
         "rates_tr = ml.build_train_only_rates(cfg, holdout)   # reusa se já existir\n"
         "sc_tr = scoring.Scorer(rates_dir=rates_tr, cfg=cfg)\n"
         "p_cel = ml.score_celulas_array(te, sc_tr, MOT, cfg)\n"
         "# referência (vazada) p/ ilustrar o efeito do leakage:\n"
         "p_cel_leak = ml.score_celulas_array(te, scoring.Scorer(cfg=cfg), MOT, cfg)\n"
         "bench = pd.DataFrame({'modelo': ['Células treino-only (justo)', 'CatBoost', 'Células c/ 2023 (vazado)'],\n"
         "                      **{k: [ml.eval_scores(yte, p_cel)[k], ml.eval_scores(yte, p_cb)[k],\n"
         "                             ml.eval_scores(yte, p_cel_leak)[k]] for k in ['AUC','Brier','LogLoss']}})\n"
         "bench.to_csv(cfg['abs']['tables'] / 'benchmark_catboost.csv', index=False)\n"
         "print(f'[{time.time()-t0:.0f}s]'); display(bench)"),
    code("# Curvas de calibração (previsto vs observado por decil)\n"
         "import matplotlib.pyplot as plt\n"
         "from src import viz\n"
         "fig, ax = plt.subplots(figsize=(5,5))\n"
         "for nome, p in [('Células', p_cel), ('CatBoost', p_cb)]:\n"
         "    r = ml.reliability_table(yte, p, 10)\n"
         "    ax.plot(r['prevista'], r['observada'], 'o-', label=nome)\n"
         "mx = max(p_cel.max(), p_cb.max())\n"
         "ax.plot([0,mx],[0,mx],'--',color='gray',label='ideal')\n"
         "ax.set_xlabel('Risco previsto'); ax.set_ylabel('Frequência observada')\n"
         "ax.set_title(f'Calibração no holdout {holdout}'); ax.legend()\n"
         "viz.save_fig(fig, cfg['abs']['figures']/'benchmark_calibracao.png'); fig"),
    code("# Importância das variáveis no CatBoost\n"
         "imp = pd.DataFrame({'feature': ml.FEATURES,\n"
         "                    'importancia': model.get_feature_importance()}\n"
         "                  ).sort_values('importancia', ascending=False)\n"
         "imp.to_csv(cfg['abs']['tables'] / 'catboost_importancia.csv', index=False)\n"
         "display(imp)"),
    md("## Leitura e ressalva metodológica (importante)\n"
       "\n"
       "**Comparação justa (todos vendo só 2020-2022, testados em 2023):** o **CatBoost\n"
       "supera o score de células** (ex.: AUC ~0,70 vs ~0,685). O modelo supervisionado\n"
       "generaliza melhor para o ano seguinte; a taxa da célula 'envelhece' quando o ano muda.\n"
       "\n"
       "**Cuidado com vazamento (out-of-time):** a 3ª linha do benchmark mostra as células\n"
       "*com 2023 embutido* (AUC ~0,738) — número **inflado**, porque a taxa da célula já\n"
       "continha os desligamentos do próprio ano de teste. Para avaliar poder preditivo é\n"
       "obrigatório o holdout temporal.\n"
       "\n"
       "**Produção ≠ avaliação:** para *pontuar uma pessoa hoje*, o score de células usa\n"
       "TODOS os anos (`data/processed/rates/`) — correto, usa toda a informação. O recorte\n"
       "treino-only (`rates_treino/`) existe só para esta avaliação honesta.\n"
       "\n"
       "**Quando ainda preferir as células:** transparência total, defensabilidade (LGPD) e\n"
       "operar sem treinar um modelo individual."),
])

print("\nNotebooks gerados em", NB_DIR)
