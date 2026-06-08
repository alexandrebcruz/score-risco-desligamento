"""Sync robusto de um diretório local -> pod (RunPod), tolerante a quedas.

Resolve os dois problemas que travavam o upload:
  1) roda como UM ÚNICO processo (não fragmenta em vários comandos);
  2) RECONECTA sozinho em queda de conexão e RETOMA de onde parou
     (idempotente por tamanho + upload atômico via .uploading + rename),
     repetindo em ciclos até TODOS os arquivos baterem (tamanho local==remoto).

Uso:
    python runpod/sync_interim.py <dir_local> <dir_remoto> [ano1 ano2 ...]
Ex.:
    python runpod/sync_interim.py data/interim/rais /workspace/data/rais
    python runpod/sync_interim.py data/interim/rais /workspace/data/rais 2019 2020 2021 2022 2023

Idempotente: pode rodar quantas vezes quiser; só envia o que falta/diverge.
"""
import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import remote as R

LOCAL = Path(sys.argv[1])
REMOTE = sys.argv[2].rstrip("/")
ANOS = set(sys.argv[3:])                 # vazio = todos

MAX_CICLOS = 50                          # backstop
RETRY_CONN = 8                           # tentativas de (re)conexão por ciclo


def _connect():
    for i in range(RETRY_CONN):
        try:
            c = R.conn(); return c, c.open_sftp()
        except Exception as e:
            print(f"  conexão falhou ({type(e).__name__}: {e}); retry {i+1}/{RETRY_CONN} em 10s", flush=True)
            time.sleep(10)
    raise SystemExit("não consegui conectar após várias tentativas")


def _want(fp: Path) -> bool:
    """Filtra por ano se ANOS foi passado (espera caminho .../ano=YYYY/...)."""
    if not ANOS:
        return True
    parts = fp.as_posix()
    return any(f"ano={a}/" in parts or parts.endswith(f"ano={a}") for a in ANOS)


def main():
    files = [p for p in sorted(LOCAL.rglob("*")) if p.is_file() and _want(p)]
    total = len(files)
    print(f"{total} arquivos candidatos em {LOCAL} (filtro anos={sorted(ANOS) or 'TODOS'})", flush=True)
    t0 = time.time()

    for ciclo in range(1, MAX_CICLOS + 1):
        c, s = _connect()
        enviados = pulados = 0
        faltam = []
        try:
            for p in files:
                rel = p.relative_to(LOCAL).as_posix()
                rp = f"{REMOTE}/{rel}"
                lsz = p.stat().st_size
                # já existe igual? pula
                try:
                    if s.stat(rp).st_size == lsz:
                        pulados += 1; continue
                except IOError:
                    pass
                # upload atômico (não corrompe em queda)
                R._mkdirs(s, str(Path(rp).parent.as_posix()))
                tmp = rp + ".uploading"
                t = time.time()
                try:
                    s.put(str(p), tmp)
                    try: s.remove(rp)
                    except IOError: pass
                    s.rename(tmp, rp)
                    enviados += 1
                    print(f"  [{enviados}] {rel} ({lsz/1e6:.1f}MB, {time.time()-t:.0f}s)", flush=True)
                except Exception as e:
                    # queda no meio do arquivo -> deixa pro próximo ciclo (reconecta)
                    print(f"  queda em {rel} ({type(e).__name__}); vou reconectar e retomar", flush=True)
                    faltam.append(rel)
                    break
        finally:
            try: c.close()
            except Exception: pass

        # confere o que ainda falta (tamanho) com uma conexão nova e limpa
        c, s = _connect()
        pend = []
        for p in files:
            rel = p.relative_to(LOCAL).as_posix(); rp = f"{REMOTE}/{rel}"
            try:
                if s.stat(rp).st_size != p.stat().st_size: pend.append(rel)
            except IOError:
                pend.append(rel)
        c.close()

        print(f"ciclo {ciclo}: enviados={enviados} pulados={pulados} pendentes={len(pend)} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if not pend:
            print(f"OK — todos os {total} arquivos sincronizados ({time.time()-t0:.0f}s)", flush=True)
            return
        time.sleep(3)

    raise SystemExit(f"ainda há pendências após {MAX_CICLOS} ciclos")


if __name__ == "__main__":
    main()
