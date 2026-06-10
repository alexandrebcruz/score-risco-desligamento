"""Personas das 14 categorias do MODELO NOVO (retreino 2021–2024), com referência
APENAS nos anos 2021–2024 das predições categorizadas. Gera, numa única passada:

  outputs/tables/persona_categorias_2124.csv           (base completa)
  outputs/tables/persona_categorias_2124_privado.csv   (sem setor público, natureza_setor!='1')

Mesmos indicadores do persona_categorias.py (composição + distintividade/lift de
cbo1/cnae2/uf com piso de 5%, médias numéricas), adaptados ao schema novo:
- escolaridade/faixas vêm como CÓDIGO int (agrupadas na leitura);
- cbo1/cnae2 derivados de cbo/cnae (não existem no parquet);
- tempo_vinculo_meses = antiguidade NA ENTRADA (leak-free);
- qtd_dias_afastamento = dias POR MÊS observado (leak-free).

Uso: /tmp/consig_venv/bin/python persona_categorias_2124.py
"""
import glob, time
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PRED = "data/processed/predicoes_2124"
ANOS_REF = [2021, 2022, 2023, 2024]
KCOL = "categoria_risco"
COLS = [KCOL, "y", "prob_desligamento", "idade", "tempo_vinculo_meses",
        "qtd_dias_afastamento", "tipo_vinculo", "faixa_remuneracao", "escolaridade",
        "natureza_setor", "simples", "intermitente", "tamanho_estab",
        "cbo", "cnae", "uf"]

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

# ---------- mapeadores de bucket (iguais ao persona_categorias.py) ----------
def _s(v):
    """Normaliza p/ string de inteiro: 9.0 -> '9' (ordinais salvas como float32
    no parquet de predições); não-numérico fica como str(v)."""
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        return str(v)

def map_tipo(v):
    return {"10": "clt_indet", "15": "clt_indet", "20": "clt_indet", "25": "clt_indet",
            "30": "estatutario", "31": "estatutario", "35": "estatutario",
            "50": "temp_determinado", "55": "temp_determinado", "60": "temp_determinado",
            "65": "temp_determinado", "90": "temp_determinado", "95": "temp_determinado",
            "97": "verde_amarelo"}.get(str(v), "outro")

def map_rem(v):
    v = _s(v)
    if v in {"0", "1", "2"}: return "rem_baixa<=1SM"
    if v in {"3", "4", "5", "6", "7"}: return "rem_media_1a5SM"
    if v in {"8", "9", "10", "11", "12"}: return "rem_alta>5SM"
    return "rem_ignorada"

def map_esc(v):
    return {"1": "ate_fund_incompleto", "2": "ate_fund_incompleto", "3": "ate_fund_incompleto",
            "4": "ate_fund_incompleto", "5": "fundamental", "6": "medio_incompleto",
            "7": "medio_completo", "8": "superior_incompleto", "9": "superior",
            "10": "superior", "11": "superior"}.get(_s(v), "nao_informado")

def map_setor(v):
    return {"1": "publico", "2": "privado", "3": "sem_fins"}.get(str(v), "outro_setor")

def map_tam(v):
    s = _s(v)
    if s in {"1", "2", "3", "4", "5"}: return "micro_peq(<=49)"
    if s in {"6", "7"}: return "media(50-249)"
    return "grande(250+)"

BUCKETS = {"tv": ("tipo_vinculo", map_tipo), "rem": ("faixa_remuneracao", map_rem),
           "esc": ("escolaridade", map_esc), "setor": ("natureza_setor", map_setor),
           "simples": ("simples", lambda v: "simples" if _s(v) == "1" else "nao_simples"),
           "interm": ("intermitente", lambda v: "intermitente" if _s(v) == "1" else "nao_interm"),
           "tam": ("tamanho_estab", map_tam)}
DISTINTIVOS = ["cbo1", "cnae2", "uf"]
NUMS = ["y", "prob_desligamento", "idade", "tempo_vinculo_meses", "qtd_dias_afastamento"]


def novo_acc():
    return {"num": defaultdict(lambda: np.zeros(len(NUMS) + 1)),       # cat -> [n, sums...]
            "buck": {b: defaultdict(int) for b in BUCKETS},            # (cat,bucket) -> n
            "dist": {f: defaultdict(int) for f in DISTINTIVOS}}        # (cat,valor) -> n

ACC = {"": novo_acc(), "_privado": novo_acc()}

# ---------- 1 passada pelas partições 2021–2024 ----------
for a in ANOS_REF:
    for fp in sorted(glob.glob(f"{PRED}/ano={a}/*.parquet")):
        d = pq.ParquetFile(fp).read(columns=COLS).to_pandas()
        d["cbo1"] = d["cbo"].astype(str).str[:1]
        d["cnae2"] = d["cnae"].astype(str).str[:2]
        # buckets mapeados 1x (compartilhados entre os 2 modos)
        for b, (c, fn) in BUCKETS.items():
            d[f"__{b}"] = d[c].map(fn)
        masks = {"": np.ones(len(d), bool), "_privado": (d["natureza_setor"].astype(str) != "1").values}
        for suf, mk in masks.items():
            dd = d[mk]
            acc = ACC[suf]
            g = dd.groupby(KCOL)[NUMS].sum(); gn = dd.groupby(KCOL).size()
            for cat, row in g.iterrows():
                acc["num"][int(cat)] += np.concatenate([[gn[cat]], row.values])
            for b in BUCKETS:
                vc = dd.groupby([KCOL, f"__{b}"]).size()
                for (cat, val), n in vc.items():
                    acc["buck"][b][(int(cat), val)] += int(n)
            for f in DISTINTIVOS:
                vc = dd.groupby([KCOL, f]).size()
                for (cat, val), n in vc.items():
                    acc["dist"][f][(int(cat), val)] += int(n)
        log(f"{a}/{fp.split('/')[-1]} ({len(d):,})")
        del d

# ---------- monta as tabelas ----------
def share_table(buckdict):
    s = pd.Series(buckdict)
    df = s.rename_axis([KCOL, "grp"]).reset_index(name="n")
    tot = df.groupby(KCOL)["n"].transform("sum")
    df["pct"] = 100 * df["n"] / tot
    return df.pivot_table(index=KCOL, columns="grp", values="pct", fill_value=0.0)

def distinctive_table(distdict, topn=2):
    s = pd.Series(distdict)
    df = s.rename_axis([KCOL, "valor"]).reset_index(name="n")
    glob = df.groupby("valor")["n"].sum(); glob = glob / glob.sum()
    out = {}
    for k, sub in df.groupby(KCOL):
        sh = sub.set_index("valor")["n"]; sh = sh / sh.sum()
        lift = sh / glob.reindex(sh.index)
        cand = [(v, sh[v], lift[v]) for v in sh.index if sh[v] >= 0.05]
        cand.sort(key=lambda x: -x[2])
        out[int(k)] = cand[:topn]
    return out

for suf, acc in ACC.items():
    cats = sorted(acc["num"].keys())
    M = np.array([acc["num"][c] for c in cats])
    base = pd.DataFrame(M, columns=["n"] + NUMS)
    base[KCOL] = cats
    for i, c in enumerate(NUMS):
        base[c + "_mean"] = base[c] / base["n"]
    base["pct_total"] = 100 * base["n"] / base["n"].sum()
    base["tempo_anos"] = base["tempo_vinculo_meses_mean"] / 12

    tv = share_table(acc["buck"]["tv"]); rem = share_table(acc["buck"]["rem"])
    esc = share_table(acc["buck"]["esc"]); setor = share_table(acc["buck"]["setor"])
    simples = share_table(acc["buck"]["simples"]); interm = share_table(acc["buck"]["interm"])
    tam = share_table(acc["buck"]["tam"])
    cbo1_d = distinctive_table(acc["dist"]["cbo1"])
    cnae2_d = distinctive_table(acc["dist"]["cnae2"])
    uf_d = distinctive_table(acc["dist"]["uf"])

    def col(df, c): return df[c] if c in df.columns else pd.Series(0.0, index=df.index)
    prof = pd.DataFrame({"categoria": base[KCOL].astype(int)})
    prof["n"] = base["n"].astype(int); prof["pct_total"] = base["pct_total"].round(2)
    prof["taxa_y"] = (base["y_mean"] * 100).round(2)
    prof["prob_media"] = (base["prob_desligamento_mean"] * 100).round(2)
    prof["idade_media"] = base["idade_mean"].round(1)
    prof["tempo_anos"] = base["tempo_anos"].round(2)
    prof["dias_afast_mes"] = base["qtd_dias_afastamento_mean"].round(2)
    for k, df_, c in [("clt_indet%", tv, "clt_indet"), ("temp_det%", tv, "temp_determinado"),
                      ("estatut%", tv, "estatutario"), ("verde_amarelo%", tv, "verde_amarelo"),
                      ("rem_baixa%", rem, "rem_baixa<=1SM"), ("rem_alta%", rem, "rem_alta>5SM"),
                      ("superior%", esc, "superior"), ("ate_fund%", esc, "ate_fund_incompleto"),
                      ("publico%", setor, "publico"), ("semfins%", setor, "sem_fins"),
                      ("simples%", simples, "simples"), ("interm%", interm, "intermitente"),
                      ("micro_peq%", tam, "micro_peq(<=49)"), ("grande%", tam, "grande(250+)")]:
        prof[k] = col(df_, c).reindex(prof["categoria"].values).round(1).values
    prof["cbo1_distintivo"] = ["; ".join(f"{v}({s*100:.0f}%/lift{l:.1f})" for v, s, l in cbo1_d.get(k, [])) for k in prof["categoria"]]
    prof["cnae2_distintivo"] = ["; ".join(f"{v}({s*100:.0f}%/lift{l:.1f})" for v, s, l in cnae2_d.get(k, [])) for k in prof["categoria"]]
    prof["uf_distintiva"] = ["; ".join(f"{v}({s*100:.0f}%/lift{l:.1f})" for v, s, l in uf_d.get(k, [])) for k in prof["categoria"]]

    out = f"outputs/tables/persona_categorias_2124{suf}.csv"
    prof.to_csv(out, index=False)
    pd.set_option("display.width", 260, "display.max_columns", 60)
    print(f"\n===== MODO {suf or 'geral'} | total={int(prof['n'].sum()):,} -> {out} =====")
    print(prof.to_string(index=False))

print("FIM_PERSONAS_2124", flush=True)
