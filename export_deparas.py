"""Exporta para data/dicts/ um CSV por de-para REAL usado no código (fonte de verdade =
os mapas vivos em src/ e nos scripts). Re-executável: roda sempre que algum mapa mudar.

  MPLCONFIGDIR=/tmp/mpl /tmp/consig_venv/bin/python export_deparas.py
"""
import sys, csv, os
sys.path.insert(0, ".")
from src.cleaning import (MAPA_MOTIVO_RAIS, MAPA_MOTIVO, MAPA_ESCOLARIDADE, MAPA_UF_IBGE)
from src.caged import MAPA_TIPOMOV_CAGED
from src.cells import UF_REGIAO

OUT = "data/dicts"
os.makedirs(OUT, exist_ok=True)

# De-paras definidos inline em persona_categorias.py (espelhados aqui; mantidos estáveis).
NATUREZA_SETOR = {"1": "publico", "2": "privado", "3": "sem_fins"}   # 1º díg. da natureza jurídica
TAMANHO_ESTAB = {"1": "0", "2": "1-4", "3": "5-9", "4": "10-19", "5": "20-49",
                 "6": "50-99", "7": "100-249", "8": "250-499", "9": "500-999", "10": "1000+"}

# (nome do arquivo, [cabeçalho], dict, ordena_por_inteiro?, fonte)
TABELAS = [
    ("depara_motivo.csv",          ["codigo_cru", "categoria_unificada"], MAPA_MOTIVO_RAIS, True,
     "src/cleaning.py: MAPA_MOTIVO_RAIS — motivo de desligamento RAIS real (clean_rais_real)"),
    ("depara_motivo_sintetico.csv", ["codigo_cru", "categoria_unificada"], MAPA_MOTIVO, True,
     "src/cleaning.py: MAPA_MOTIVO — motivo (caminho sintético/_map_motivo); NÃO usado no RAIS real"),
    ("depara_escolaridade.csv",    ["grau_instrucao", "faixa_canonica"], MAPA_ESCOLARIDADE, True,
     "src/cleaning.py: MAPA_ESCOLARIDADE — grau de instrução (1..11) -> faixa"),
    ("depara_uf.csv",              ["prefixo_ibge", "uf"], MAPA_UF_IBGE, False,
     "src/cleaning.py: MAPA_UF_IBGE — 2 primeiros díg. do município (IBGE) -> UF"),
    ("depara_uf_regiao.csv",       ["uf", "regiao"], UF_REGIAO, False,
     "src/cells.py: UF_REGIAO — UF -> região (N/NE/CO/SE/S)"),
    ("depara_tipomov_caged.csv",   ["tipo_movimentacao", "categoria_unificada"], MAPA_TIPOMOV_CAGED, True,
     "src/caged.py: MAPA_TIPOMOV_CAGED — tipo de movimentação Novo CAGED -> categoria"),
    ("depara_natureza_setor.csv",  ["natureza_setor_1dig", "setor"], NATUREZA_SETOR, False,
     "persona_categorias.py: 1º díg. da natureza jurídica -> setor (default: outro_setor)"),
    ("depara_tamanho_estab.csv",   ["codigo", "faixa_vinculos"], TAMANHO_ESTAB, True,
     "persona_categorias.py: tamanho do estabelecimento (1..10) -> faixa de nº de vínculos"),
]

print("Exportando de-paras -> data/dicts/\n")
for fname, header, mapa, ord_int, fonte in TABELAS:
    itens = sorted(mapa.items(), key=(lambda kv: int(kv[0])) if ord_int else (lambda kv: str(kv[0])))
    with open(os.path.join(OUT, fname), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header)
        for k, v in itens:
            w.writerow([k, v])
    print(f"  {fname:30s} {len(itens):>3} linhas   ({fonte})")
print("\nFIM.")
