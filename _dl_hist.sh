#!/bin/bash
set -u
B="ftp://ftp.mtps.gov.br/pdet/microdados/RAIS"
ROOT="/mnt/d/Projetos/Consignado/data/raw/RAIS"
LOG="/mnt/d/Projetos/Consignado/_dl_hist.log"; echo "INICIO $(date)" > "$LOG"
UFS="AC AL AM AP BA CE DF ES GO MA MG MS MT PA PB PE PI PR RJ RN RO RR RS SC SE SP TO"
REG="NORTE NORDESTE CENTRO_OESTE MG_ES_RJ SP SUL"
dl(){ f="$2"; [ -s "$f" ] && { echo "cache $(basename $f)" >>"$LOG"; return; }
      curl -sS -o "$f.part" "$1" && mv "$f.part" "$f" && echo "ok $(basename $f) $(du -h "$f"|cut -f1)" >>"$LOG" || echo "FALHOU $(basename $f)" >>"$LOG"; }
for ano in 2016 2017; do mkdir -p "$ROOT/$ano"; for uf in $UFS; do dl "$B/$ano/${uf}${ano}.7z" "$ROOT/$ano/${uf}${ano}.7z"; done; done
for ano in 2018 2019; do mkdir -p "$ROOT/$ano"; for r in $REG; do dl "$B/$ano/RAIS_VINC_PUB_${r}.7z" "$ROOT/$ano/RAIS_VINC_PUB_${r}.7z"; done; done
echo "FIM $(date)" >> "$LOG"
