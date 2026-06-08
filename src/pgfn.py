"""Download dos Dados Abertos da Dívida Ativa da União e do FGTS (PGFN).

A PGFN publica, trimestralmente, a base completa de créditos inscritos em
dívida ativa em formato aberto (.csv dentro de .zip), segmentada por sistema
de origem do débito e por UF. Os três grupos:

    - Previdenciario      (~80 MB/trimestre)   contribuições previdenciárias (INSS)
    - FGTS                (~15 MB/trimestre)   débitos de FGTS inscritos
    - Nao_Previdenciario  (~1,2 GB/trimestre)  demais tributos federais

URL (confirmada): {base}/{ano}_trimestre_{TT}/Dados_abertos_{TIPO}.zip
    ex.: https://dadosabertos.pgfn.gov.br/2026_trimestre_01/Dados_abertos_FGTS.zip

Responsabilidades deste módulo:
    - iterar os trimestres do escopo (de 2020 T1 até o mais recente publicado);
    - descobrir automaticamente o último trimestre disponível (HEAD);
    - baixar de forma idempotente (cache + .part + validação de tamanho);
    - opcionalmente extrair os .zip;
    - registrar um manifesto (CSV) do que foi baixado.

Uso como script (a partir da raiz do projeto):
    python -m src.pgfn                      # usa o config.yaml (tipos: Previdenciario, FGTS)
    python -m src.pgfn --tipos FGTS         # só FGTS
    python -m src.pgfn --tipos all          # inclui Nao_Previdenciario (~30 GB no total!)
    python -m src.pgfn --ano-fim 2024 --trimestre-fim 4 --sem-extrair
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import zipfile

# Os três grupos publicados pela PGFN (nomes exatos usados na URL).
TIPOS_VALIDOS = ("Previdenciario", "Nao_Previdenciario", "FGTS")
# Padrão do projeto: o foco (pergunta de negócio) é previdenciário + FGTS.
# Nao_Previdenciario é gigantesco (~1,2 GB/trimestre) e fica fora por padrão.
TIPOS_PADRAO = ("Previdenciario", "FGTS")

BASE_URL = "https://dadosabertos.pgfn.gov.br"


@dataclass(frozen=True)
class Trimestre:
    ano: int
    tri: int  # 1..4

    @property
    def rotulo(self) -> str:
        return f"{self.ano}_trimestre_{self.tri:02d}"

    def proximo(self) -> "Trimestre":
        return Trimestre(self.ano + 1, 1) if self.tri == 4 else Trimestre(self.ano, self.tri + 1)


def iter_trimestres(inicio: Trimestre, fim: Trimestre):
    """Gera trimestres de `inicio` até `fim` (inclusive)."""
    if (fim.ano, fim.tri) < (inicio.ano, inicio.tri):
        return
    t = inicio
    while (t.ano, t.tri) <= (fim.ano, fim.tri):
        yield t
        t = t.proximo()


def build_url(tri: Trimestre, tipo: str, base: str = BASE_URL) -> str:
    return f"{base}/{tri.rotulo}/Dados_abertos_{tipo}.zip"


# ---------------------------------------------------------------------------
# Rede
# ---------------------------------------------------------------------------
def _head(url: str, session=None, timeout: int = 60):
    """HEAD request; retorna (existe: bool, content_length: int | None)."""
    import requests

    sess = session or requests.Session()
    try:
        r = sess.head(url, timeout=timeout, allow_redirects=True)
    except requests.RequestException:
        return False, None
    if r.status_code != 200:
        return False, None
    cl = r.headers.get("Content-Length")
    return True, (int(cl) if cl and cl.isdigit() else None)


def descobrir_ultimo_trimestre(inicio: Trimestre, tipo: str = "FGTS",
                               base: str = BASE_URL, session=None,
                               limite_falhas: int = 2) -> Trimestre:
    """Sonda HEAD a partir de `inicio` até achar o último trimestre publicado.

    Avança enquanto a URL responder 200. Para após `limite_falhas` ausências
    consecutivas (cobre o caso de um trimestre demorar a sair para um dos tipos).
    Usa o tipo mais leve (FGTS) como sentinela.
    """
    ultimo = None
    t = inicio
    falhas = 0
    while falhas < limite_falhas:
        existe, _ = _head(build_url(t, tipo, base), session=session)
        if existe:
            ultimo = t
            falhas = 0
        else:
            falhas += 1
        t = t.proximo()
    if ultimo is None:
        raise RuntimeError(f"Nenhum trimestre publicado a partir de {inicio.rotulo} (tipo {tipo}).")
    return ultimo


def baixar_zip(url: str, dest: Path, session=None, chunk: int = 1 << 20,
               timeout: int = 120, esperado: int | None = None) -> tuple[Path, str]:
    """Baixa `url` para `dest` de forma idempotente.

    Retorna (caminho, status) onde status ∈ {"cache", "baixado"}.
    Valida tamanho contra Content-Length quando disponível; um .part só vira
    arquivo final após o download completo (download atômico).
    """
    import requests

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if esperado is None:
        _, esperado = _head(url, session=session)

    # Cache hit: existe e (sem tamanho esperado OU tamanho confere).
    if dest.exists() and dest.stat().st_size > 0:
        if esperado is None or dest.stat().st_size == esperado:
            return dest, "cache"

    sess = session or requests.Session()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with sess.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for bloco in resp.iter_content(chunk_size=chunk):
                if bloco:
                    fh.write(bloco)
    if esperado is not None and tmp.stat().st_size != esperado:
        tmp.unlink(missing_ok=True)
        raise IOError(f"Tamanho divergente em {url}: "
                      f"{tmp.stat().st_size} != {esperado} (esperado)")
    tmp.replace(dest)
    return dest, "baixado"


def extrair_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extrai um .zip para `dest_dir` (idempotente: pula o que já existe)."""
    zip_path, dest_dir = Path(zip_path), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    extraidos = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            alvo = dest_dir / info.filename
            if not (alvo.exists() and alvo.stat().st_size == info.file_size):
                z.extract(info, dest_dir)
            extraidos.append(alvo)
    return extraidos


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------
def baixar_periodo(inicio: Trimestre, fim: Trimestre, tipos, destino: Path,
                   *, base: str = BASE_URL, extrair: bool = True,
                   session=None) -> list[dict]:
    """Baixa (e opcionalmente extrai) todos os (trimestre × tipo) do período.

    Estrutura em disco:
        {destino}/{ano}_trimestre_{TT}/Dados_abertos_{TIPO}.zip
        {destino}/{ano}_trimestre_{TT}/{TIPO}/...csv   (se extrair=True)

    Retorna a lista de registros do manifesto.
    """
    import requests

    sess = session or requests.Session()
    destino = Path(destino)
    manifesto: list[dict] = []
    for tri in iter_trimestres(inicio, fim):
        for tipo in tipos:
            url = build_url(tri, tipo, base)
            existe, tamanho = _head(url, session=sess)
            pasta = destino / tri.rotulo
            zip_dest = pasta / f"Dados_abertos_{tipo}.zip"
            reg = {"trimestre": tri.rotulo, "tipo": tipo, "url": url,
                   "bytes": tamanho or "", "status": "", "arquivos": ""}
            if not existe:
                reg["status"] = "indisponivel"
                print(f"  [..] {tri.rotulo} {tipo:<18} indisponível", flush=True)
                manifesto.append(reg)
                continue
            try:
                _, status = baixar_zip(url, zip_dest, session=sess, esperado=tamanho)
                reg["status"] = status
                mb = (zip_dest.stat().st_size) / 1048576
                msg = f"  [{'ok' if status == 'baixado' else '=='}] {tri.rotulo} {tipo:<18} {mb:7.1f} MB ({status})"
                if extrair:
                    arqs = extrair_zip(zip_dest, pasta / tipo)
                    reg["arquivos"] = len(arqs)
                    msg += f" -> {len(arqs)} csv"
                print(msg, flush=True)
            except Exception as exc:  # rede/integridade: registra e segue
                reg["status"] = f"erro: {exc}"
                print(f"  [!!] {tri.rotulo} {tipo:<18} ERRO: {exc}", flush=True)
            manifesto.append(reg)
    return manifesto


def _tipo_de_zip(zip_path: Path) -> str:
    """Infere o tipo (Previdenciario/FGTS/Nao_Previdenciario) pelo nome do .zip."""
    nome = zip_path.name
    for t in TIPOS_VALIDOS:
        if t in nome:
            return t
    return "?"


def _ord_trim(rotulo: str) -> int:
    """Converte '2020_trimestre_03' -> 20203 para ordenar trimestres."""
    try:
        ano, _, tri = rotulo.split("_")
        return int(ano) * 10 + int(tri)
    except Exception:
        return 0


# Mapa TIPO_SITUACAO_INSCRICAO -> bucket (0=cobrança/exigível, 1=parcelado/
# benefício fiscal, 2=garantida, 3=suspensa por decisão judicial). Valores não
# previstos caem em "cobrança" (conservador: tratado como exigível).
_SITUACAO_BUCKET = {
    "Em cobrança": 0,
    "Benefício Fiscal": 1,
    "Em negociação": 1,
    "Garantia": 2,
    "Suspenso por decisão judicial": 3,
}


def _bucket_situacao(tipo_situacao: str) -> int:
    return _SITUACAO_BUCKET.get((tipo_situacao or "").strip(), 0)


def agregar_empresas(destino: Path, *, tipos=None, apenas_pj: bool = True,
                     progress=print) -> "list[dict]":
    """Varre os .zip baixados e consolida UMA linha por devedor (CPF/CNPJ).

    Lê os CSVs em streaming de dentro de cada .zip (não extrai pro disco).
    Cada .zip corresponde a um (trimestre × tipo); dentro dele há um CSV por UF.

    Para cada devedor acumula, ao longo do período:
        nome, uf, n_trimestres_prev, n_trimestres_fgts,
        primeiro/ultimo trimestre em que apareceu;
    e, NO ÚLTIMO TRIMESTRE em que a empresa consta (snapshot mais recente,
    combinando previdenciário + FGTS), a dívida quebrada por situação:
        em cobrança (exigível), parcelada/benefício fiscal, garantida,
        suspensa por decisão judicial.
    `DIVIDA_EXIGIVEL = S` quando há valor "Em cobrança" nesse snapshot.

    `apenas_pj=True` mantém só 'Pessoa jurídica' (empresas). Retorna lista de
    dicts ordenada por valor em cobrança (desc).
    """
    import io as _io

    destino = Path(destino)
    tipos = tuple(tipos) if tipos else None
    zips = sorted(destino.rglob("Dados_abertos_*.zip"))
    if tipos:
        zips = [z for z in zips if _tipo_de_zip(z) in tipos]
    if not zips:
        raise FileNotFoundError(f"Nenhum .zip em {destino} (rode `python -m src.pgfn`).")

    # devedor -> [nome, uf, n_prev, n_fgts, first_ord, last_ord,
    #             ref_ord, cob, parc, gar, susp]  (buckets = snapshot de ref_ord)
    acc: dict[str, list] = {}

    for iz, zp in enumerate(zips, 1):
        tipo = _tipo_de_zip(zp)
        rot = zp.parent.name
        ordem = _ord_trim(rot)
        eh_prev = (tipo == "Previdenciario")
        # por devedor neste trimestre: [cob, parc, gar, susp]
        local: dict[str, list] = {}
        nomes: dict[str, str] = {}
        ufs: dict[str, str] = {}
        with zipfile.ZipFile(zp) as zf:
            membros = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            for m in membros:
                with zf.open(m) as fh:
                    txt = _io.TextIOWrapper(fh, encoding="latin-1", errors="replace")
                    rd = csv.reader(txt, delimiter=";")
                    header = next(rd, None)
                    if not header:
                        continue
                    h = {c: i for i, c in enumerate(header)}
                    i_doc = h.get("CPF_CNPJ")
                    i_tp = h.get("TIPO_PESSOA")
                    i_nome = h.get("NOME_DEVEDOR")
                    i_val = h.get("VALOR_CONSOLIDADO")
                    i_sit = h.get("TIPO_SITUACAO_INSCRICAO")
                    i_uf = h.get("UF_DEVEDOR", h.get("UF_UNIDADE_RESPONSAVEL"))
                    if i_doc is None:
                        continue
                    for row in rd:
                        if len(row) <= i_doc:
                            continue
                        if apenas_pj and i_tp is not None and row[i_tp] != "Pessoa jurídica":
                            continue
                        doc = row[i_doc].strip()
                        if not doc:
                            continue
                        try:
                            val = float(row[i_val]) if i_val is not None and row[i_val] else 0.0
                        except ValueError:
                            val = 0.0
                        sit = row[i_sit] if i_sit is not None and len(row) > i_sit else ""
                        b = _bucket_situacao(sit)  # 0=cob 1=parc 2=gar 3=susp
                        v = local.get(doc)
                        if v is None:
                            v = local[doc] = [0.0, 0.0, 0.0, 0.0]
                        v[b] += val
                        if doc not in nomes and i_nome is not None and len(row) > i_nome:
                            nomes[doc] = row[i_nome]
                            ufs[doc] = row[i_uf] if i_uf is not None and len(row) > i_uf else ""
        # merge do trimestre no acumulador global
        for doc, bk in local.items():
            soma = bk[0] + bk[1] + bk[2] + bk[3]  # total deste devedor neste zip
            e = acc.get(doc)
            if e is None:
                acc[doc] = [nomes.get(doc, ""), ufs.get(doc, ""),
                            1 if eh_prev else 0, 0 if eh_prev else 1,
                            ordem, ordem, ordem, bk[0], bk[1], bk[2], bk[3], soma]
                continue
            if eh_prev:
                e[2] += 1
            else:
                e[3] += 1
            if ordem < e[4]:
                e[4] = ordem
            if ordem > e[5]:
                e[5] = ordem
            # snapshot da dívida: mantém só o trimestre mais recente (combina prev+fgts)
            if ordem > e[6]:
                e[6], e[7], e[8], e[9], e[10] = ordem, bk[0], bk[1], bk[2], bk[3]
            elif ordem == e[6]:
                e[7] += bk[0]; e[8] += bk[1]; e[9] += bk[2]; e[10] += bk[3]
            # maior dívida total observada num único trimestre (período inteiro)
            if soma > e[11]:
                e[11] = soma
        progress(f"  [{iz:>2}/{len(zips)}] {rot} {tipo:<18} "
                 f"devedores no trim: {len(local):>7,} | total único: {len(acc):>9,}")

    def _rot(o: int) -> str:
        return f"{o // 10}_trimestre_{o % 10:02d}"

    linhas = []
    for doc, e in acc.items():
        nome, uf, n_prev, n_fgts, fo, lo, _ro, cob, parc, gar, susp, vmax = e
        total = cob + parc + gar + susp
        linhas.append({
            "CPF_CNPJ": doc,
            "NOME_DEVEDOR": nome,
            "UF": uf,
            "TEVE_PREVIDENCIARIO": "S" if n_prev else "N",
            "TEVE_FGTS": "S" if n_fgts else "N",
            "N_TRIMESTRES_PREV": n_prev,
            "N_TRIMESTRES_FGTS": n_fgts,
            "PRIMEIRO_TRIMESTRE": _rot(fo),
            "ULTIMO_TRIMESTRE": _rot(lo),
            "MAIOR_DIVIDA_TRIMESTRE": round(vmax, 2),
            "VALOR_EM_COBRANCA": round(cob, 2),
            "VALOR_PARCELADO_BENEF": round(parc, 2),
            "VALOR_GARANTIDO": round(gar, 2),
            "VALOR_SUSPENSO_JUD": round(susp, 2),
            "VALOR_TOTAL_REF": round(total, 2),
            "DIVIDA_EXIGIVEL": "S" if cob > 0 else "N",
        })
    linhas.sort(key=lambda r: (r["VALOR_EM_COBRANCA"], r["VALOR_TOTAL_REF"]),
                reverse=True)
    return linhas


def escrever_manifesto(manifesto: list[dict], caminho: Path) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    campos = ["trimestre", "tipo", "url", "bytes", "status", "arquivos"]
    with open(caminho, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(manifesto)
    return caminho


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _resolver_config():
    """Lê a seção `pgfn` do config.yaml (com defaults se ausente)."""
    try:
        from src.config import load_config
    except ImportError:  # execução fora do pacote
        from config import load_config  # type: ignore
    cfg = load_config()
    p = cfg.get("pgfn", {}) or {}
    raw = cfg["abs"]["raw"]
    destino = Path(p.get("dir")) if p.get("dir") else raw / "pgfn"
    if not destino.is_absolute():
        destino = cfg["root"] / destino
    return cfg, p, destino


def main(argv=None) -> int:
    cfg, p, destino_cfg = _resolver_config()

    ap = argparse.ArgumentParser(description="Baixa os Dados Abertos da Dívida Ativa (PGFN).")
    ap.add_argument("--ano-inicio", type=int, default=p.get("ano_inicio", 2020))
    ap.add_argument("--trimestre-inicio", type=int, default=p.get("trimestre_inicio", 1))
    ap.add_argument("--ano-fim", type=int, default=p.get("ano_fim"),
                    help="Se omitido, detecta automaticamente o último trimestre publicado.")
    ap.add_argument("--trimestre-fim", type=int, default=p.get("trimestre_fim"))
    ap.add_argument("--tipos", nargs="+", default=list(p.get("tipos", TIPOS_PADRAO)),
                    help="Subconjunto de %s, ou 'all' para os três." % (TIPOS_VALIDOS,))
    ap.add_argument("--destino", type=Path, default=destino_cfg)
    ap.add_argument("--base-url", default=p.get("base_url", BASE_URL))
    ap.add_argument("--sem-extrair", action="store_true",
                    default=not p.get("extrair", True))
    args = ap.parse_args(argv)

    tipos = list(TIPOS_VALIDOS) if args.tipos == ["all"] else args.tipos
    invalidos = [t for t in tipos if t not in TIPOS_VALIDOS]
    if invalidos:
        ap.error(f"tipos inválidos {invalidos}; use {TIPOS_VALIDOS} ou 'all'.")

    inicio = Trimestre(args.ano_inicio, args.trimestre_inicio)

    if args.ano_fim:
        fim = Trimestre(args.ano_fim, args.trimestre_fim or 4)
    else:
        print(f"Detectando último trimestre publicado a partir de {inicio.rotulo} ...", flush=True)
        fim = descobrir_ultimo_trimestre(inicio, base=args.base_url)

    qtd = sum(1 for _ in iter_trimestres(inicio, fim))
    print(f"Período: {inicio.rotulo} → {fim.rotulo}  ({qtd} trimestres)")
    print(f"Tipos:   {', '.join(tipos)}")
    print(f"Destino: {args.destino}")
    print(f"Extrair: {not args.sem_extrair}\n")

    manifesto = baixar_periodo(inicio, fim, tipos, args.destino,
                               base=args.base_url, extrair=not args.sem_extrair)

    man_path = escrever_manifesto(manifesto, args.destino / "_manifesto.csv")
    ok = sum(1 for m in manifesto if m["status"] in ("baixado", "cache"))
    print(f"\nConcluído: {ok}/{len(manifesto)} arquivos disponíveis. Manifesto: {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
