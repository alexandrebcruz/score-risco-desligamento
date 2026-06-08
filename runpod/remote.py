"""Helper de operações no Pod RunPod via SSH/SFTP (paramiko).

Reutilizável e resiliente: os dados do pod ficam em runpod/pod.json, então
reconecto mesmo após reinício do ambiente local. Uso:
    python remote.py exec "comando"
    python remote.py put <local> <remoto>
    python remote.py get <remoto> <local>
    python remote.py putdir <local_dir> <remoto_dir>
    python remote.py getdir <remoto_dir> <local_dir>
"""
import sys, json, os, stat, time
from pathlib import Path
import paramiko

POD = json.load(open(Path(__file__).parent / "pod.json"))
KEYFILE = "/mnt/d/Projetos/HubService/secrets/runpod_ssh_key"


def conn():
    k = paramiko.Ed25519Key.from_private_key_file(KEYFILE)
    c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(POD["host"], port=POD["port"], username="root", pkey=k, timeout=40,
              banner_timeout=40, auth_timeout=40)
    return c


def run(c, cmd, echo=True):
    _, o, e = c.exec_command(cmd, timeout=POD.get("cmd_timeout", 7200))
    out = o.read().decode(); err = e.read().decode()
    if echo:
        if out.strip(): print(out.rstrip())
        if err.strip(): print("[stderr]", err.rstrip())
    return out, err


def _put(sftp, lp, rp):
    sz = os.path.getsize(lp); t = time.time()
    sftp.put(lp, rp)
    print(f"  put {os.path.basename(lp)} ({sz/1e6:.1f}MB, {time.time()-t:.0f}s)", flush=True)


def putdir(sftp, ld, rd):
    ld = Path(ld)
    _mkdirs(sftp, rd)
    for p in sorted(ld.rglob("*")):
        rel = p.relative_to(ld)
        target = f"{rd}/{rel.as_posix()}"
        if p.is_dir():
            _mkdirs(sftp, target)
        else:
            _mkdirs(sftp, str(Path(target).parent.as_posix()))
            _put(sftp, str(p), target)


def getdir(sftp, rd, ld):
    Path(ld).mkdir(parents=True, exist_ok=True)
    for entry in sftp.listdir_attr(rd):
        rpath = f"{rd}/{entry.filename}"
        lpath = os.path.join(ld, entry.filename)
        if stat.S_ISDIR(entry.st_mode):
            getdir(sftp, rpath, lpath)
        else:
            sftp.get(rpath, lpath)
            print(f"  got {entry.filename} ({entry.st_size/1e6:.1f}MB)", flush=True)


def _mkdirs(sftp, path):
    parts = path.strip("/").split("/")
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur)


if __name__ == "__main__":
    cmd = sys.argv[1]
    c = conn()
    if cmd == "exec":
        run(c, sys.argv[2])
    elif cmd == "put":
        s = c.open_sftp(); _mkdirs(s, str(Path(sys.argv[3]).parent.as_posix())); _put(s, sys.argv[2], sys.argv[3])
    elif cmd == "get":
        s = c.open_sftp(); s.get(sys.argv[2], sys.argv[3]); print("ok")
    elif cmd == "putdir":
        s = c.open_sftp(); putdir(s, sys.argv[2], sys.argv[3])
    elif cmd == "getdir":
        s = c.open_sftp(); getdir(s, sys.argv[2], sys.argv[3])
    c.close()
