"""Gera a lista consolidada de empresas devedoras (PGFN) sem depender de pandas.

Roda src.pgfn.agregar_empresas sobre todos os .zip baixados e escreve o CSV.
Usado para produzir o artefato fora do notebook (ambiente sem pandas).
    python3 _run_pgfn_lista.py
"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.pgfn import agregar_empresas  # noqa: E402

DESTINO = Path("data/raw/pgfn")
OUT = Path("outputs/tables/pgfn_empresas_devedoras.csv")


def main():
    t0 = time.time()
    linhas = agregar_empresas(DESTINO, tipos=["Previdenciario", "FGTS"], apenas_pj=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    campos = list(linhas[0].keys())
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)
    n_prev = sum(1 for r in linhas if r["TEVE_PREVIDENCIARIO"] == "S")
    n_fgts = sum(1 for r in linhas if r["TEVE_FGTS"] == "S")
    n_ambas = sum(1 for r in linhas if r["TEVE_PREVIDENCIARIO"] == "S" and r["TEVE_FGTS"] == "S")
    n_exig = sum(1 for r in linhas if r["DIVIDA_EXIGIVEL"] == "S")
    print(f"\n=== RESUMO ({time.time()-t0:.0f}s) ===")
    print(f"Empresas únicas (PJ)      : {len(linhas):,}")
    print(f"  com previdenciário      : {n_prev:,}")
    print(f"  com FGTS                : {n_fgts:,}")
    print(f"  com ambas               : {n_ambas:,}")
    print(f"  com dívida EXIGÍVEL      : {n_exig:,}  (em cobrança no último trim.)")
    print(f"  só parcelada/garant/susp: {len(linhas)-n_exig:,}")
    print(f"Arquivo: {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
