"""Upload resumível de um diretório para o pod: pula arquivos já presentes
(mesmo caminho relativo e mesmo tamanho). Reexecutável quantas vezes precisar."""
import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import remote as R

LOCAL = Path(sys.argv[1])       # ex: data/interim/rais
REMOTE = sys.argv[2]            # ex: /workspace/data/rais

c = R.conn()
s = c.open_sftp()

def rsize(path):
    try:
        return s.stat(path).st_size
    except IOError:
        return -1

files = [p for p in sorted(LOCAL.rglob("*")) if p.is_file()]
sent = skipped = 0
t0 = time.time()
for p in files:
    rel = p.relative_to(LOCAL).as_posix()
    rp = f"{REMOTE}/{rel}"
    lsz = p.stat().st_size
    if rsize(rp) == lsz:
        skipped += 1
        continue
    R._mkdirs(s, str(Path(rp).parent.as_posix()))
    t = time.time()
    tmp = rp + ".uploading"
    s.put(str(p), tmp)                 # grava em temp...
    try:
        s.remove(rp)
    except IOError:
        pass
    s.rename(tmp, rp)                  # ...e renomeia atômico (evita arquivo parcial/colisão)
    sent += 1
    print(f"  [{sent}] {rel} ({lsz/1e6:.1f}MB, {time.time()-t:.0f}s)", flush=True)

print(f"FIM: enviados={sent} pulados={skipped} total={len(files)} ({time.time()-t0:.0f}s)", flush=True)
c.close()
