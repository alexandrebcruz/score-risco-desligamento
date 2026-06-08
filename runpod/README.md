# runpod/ — helpers de execução no RunPod (via HubService)

## Transferência de dados (LIÇÃO IMPORTANTE)

Os uploads "travavam a cada ano" porque **uma tarefa de background do bash era
encerrada quando eu disparava o próximo comando** (o sandbox mantém ~1 background
vivo; cada novo `Bash` derruba o anterior). Como eu checava o progresso a cada
~ano, meu próprio check matava o upload na fronteira de ano. Houve também **1 queda
real de rede** (Connection reset) no meio de um arquivo grande.

### Regra de ouro
- Use **`sync_interim.py`** (um único processo, reconecta e retoma sozinho até
  tudo bater) e **NÃO emita outros comandos** até a notificação de conclusão dele.

```bash
# tudo:
python runpod/sync_interim.py data/interim/rais /workspace/data/rais
# só alguns anos (ensemble base/lags só precisam de 2019-2023):
python runpod/sync_interim.py data/interim/rais /workspace/data/rais 2019 2020 2021 2022 2023
```
Idempotente (pula por tamanho) e atômico (`.uploading`+rename → não corrompe em queda).
Alternativas ainda melhores p/ o futuro: volume de rede persistente no RunPod, ou
puxar os dados de dentro do pod (rclone/wget) em vez de empurrar do local.

## Scripts
- `remote.py` — SSH/SFTP (paramiko) via `pod.json`. `exec|put|get|putdir|getdir`.
- `sync_interim.py` — upload robusto/retomável (ver acima).
- `build_aggs_pod.py` — build paralelo das agregações de lag (CPU, multiprocessing).
- `finish_aggs.py` — finaliza/merge dos aggs sequencialmente (sem Pool, robusto).
- `prep_lags.py` — enriquece o interim com as 114 colunas de lag (normalize rápido).
- `train_model.py` — ensemble BASE (sem lags), 1 fase/processo: `A | B | eval`.
- `train_model_lags.py` — ensemble COM lags, pools quantizados em disco p/ caber na RAM:
  `poolfit <M> | poolval <M> | fit <M> | eval`.

## Memória (o que aprendemos)
- Container RAM ~188 GB: construir os 2 Pools (132M+148M × 136 feat) estoura.
  Solução: **construir cada pool em processo isolado**, quantizar e salvar em disco
  (`quantized://`), depois carregar leve e treinar.
- VRAM do fit com 136 features ≈ **173 GB** → exige **B200 180GB** (H200 141GB dá OOM).
- Ensemble base (22 feat) cabe em A100 80GB (~77 GB de VRAM).
