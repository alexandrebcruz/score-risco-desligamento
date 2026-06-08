#!/bin/bash
# Baixa Dados Abertos PGFN (Previdenciario + FGTS), 2020 T1 -> 2026 T1.
# Idempotente: pula zip já existente com tamanho == Content-Length.
# Layout compatível com src/pgfn.py: data/raw/pgfn/{ano}_trimestre_TT/Dados_abertos_{TIPO}.zip
set -u
BASE="https://dadosabertos.pgfn.gov.br"
ROOT="/mnt/d/Projetos/Consignado/data/raw/pgfn"
LOG="/mnt/d/Projetos/Consignado/_dl_pgfn.log"
TIPOS=(Previdenciario FGTS)
echo "INICIO $(date)" > "$LOG"

for ano in 2020 2021 2022 2023 2024 2025 2026; do
  for tri in 01 02 03 04; do
    # Limite superior: 2026 T1
    [ "$ano" = "2026" ] && [ "$tri" != "01" ] && continue
    rot="${ano}_trimestre_${tri}"
    for tipo in "${TIPOS[@]}"; do
      url="$BASE/$rot/Dados_abertos_${tipo}.zip"
      dir="$ROOT/$rot"; mkdir -p "$dir"
      dest="$dir/Dados_abertos_${tipo}.zip"
      # tamanho remoto (Content-Length)
      rem=$(curl -sS -I --max-time 60 "$url" 2>/dev/null | tr -d '\r' \
            | awk -F': ' 'tolower($1)=="content-length"{print $2}')
      if [ -z "$rem" ]; then
        echo "SKIP  $rot $tipo (indisponivel/404)" >> "$LOG"; continue
      fi
      if [ -s "$dest" ] && [ "$(stat -c%s "$dest")" = "$rem" ]; then
        echo "cache $rot $tipo ($(du -h "$dest"|cut -f1))" >> "$LOG"; continue
      fi
      if curl -sS --max-time 1800 -o "$dest.part" "$url"; then
        got=$(stat -c%s "$dest.part" 2>/dev/null || echo 0)
        if [ "$got" = "$rem" ]; then
          mv "$dest.part" "$dest"
          echo "ok    $rot $tipo ($(du -h "$dest"|cut -f1))" >> "$LOG"
        else
          rm -f "$dest.part"
          echo "FALHA $rot $tipo (tamanho $got != $rem)" >> "$LOG"
        fi
      else
        rm -f "$dest.part"
        echo "FALHA $rot $tipo (curl erro)" >> "$LOG"
      fi
    done
  done
done
echo "FIM $(date)" >> "$LOG"
echo "TOTAL_ZIPS $(find "$ROOT" -name '*.zip' | wc -l)" >> "$LOG"
