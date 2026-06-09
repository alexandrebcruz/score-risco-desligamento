"""Perfila cada categoria de risco (1..23) -> base para 'personas'.

Calcula, por categoria, indicadores interpretáveis (traduzidos pelo dicionário RAIS)
e os valores mais DISTINTIVOS (sobre-representados vs a base) de CBO/CNAE/UF.
Usa pyarrow group_by (baixa memória). Saída: outputs/tables/persona_categorias.csv
"""
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc

import sys
SRC = "outputs/predicoes_2023_ensemble_base_categorizado.parquet"
KCOL = "categoria_risco"
# modo "privado": exclui o setor público (natureza_setor=='1') — público-alvo do
# consignado PRIVADO. Saída com sufixo _privado. Default: toda a base.
MODO = sys.argv[1] if len(sys.argv) > 1 else "todos"
FILTRO = [("natureza_setor", "!=", "1")] if MODO == "privado" else None
SUF = "_privado" if MODO == "privado" else ""

def ct(feat):
    """crosstab counts (categoria x valor) -> DataFrame [K, valor, n]."""
    t = pq.read_table(SRC, columns=[KCOL, feat], filters=FILTRO)
    g = t.group_by([KCOL, feat]).aggregate([(KCOL, "count")])
    return g.to_pandas().rename(columns={f"{KCOL}_count": "n", feat: "valor"})

def num_mean(feats):
    t = pq.read_table(SRC, columns=[KCOL] + feats, filters=FILTRO)
    g = t.group_by([KCOL]).aggregate([(f, "mean") for f in feats] + [(KCOL, "count")])
    return g.to_pandas()

# base numérica
base = num_mean(["idade", "tempo_vinculo_meses", "qtd_dias_afastamento", "y", "prob_desligamento"])
base = base.rename(columns={f"{KCOL}_count": "n"}).sort_values(KCOL).reset_index(drop=True)
base["pct_total"] = 100 * base["n"] / base["n"].sum()
base["tempo_anos"] = base["tempo_vinculo_meses_mean"] / 12

def share(feat, mapfn):
    """retorna dict[K] -> {rótulo: % dentro da categoria} usando agregação de valores."""
    c = ct(feat); c["grp"] = c["valor"].map(mapfn)
    tot = c.groupby(KCOL)["n"].transform("sum")
    c["pct"] = 100 * c["n"] / tot
    return c.groupby([KCOL, "grp"])["pct"].sum().unstack(fill_value=0.0)

# --- tipo de vínculo: indeterminado (10/15/20/25) vs determinado/temporário ---
def map_tipo(v):
    return {"10":"clt_indet","15":"clt_indet","20":"clt_indet","25":"clt_indet",
            "30":"estatutario","31":"estatutario","35":"estatutario",
            "50":"temp_determinado","55":"temp_determinado","60":"temp_determinado",
            "65":"temp_determinado","90":"temp_determinado","95":"temp_determinado",
            "97":"verde_amarelo"}.get(str(v),"outro")
tv = share("tipo_vinculo", map_tipo)

# --- faixa remuneração (SM): baixa(<=1SM: 0,1,2) media(3-7) alta(8-12) ---
def map_rem(v):
    v=str(v)
    if v in {"0","1","2"}: return "rem_baixa<=1SM"
    if v in {"3","4","5","6","7"}: return "rem_media_1a5SM"
    if v in {"8","9","10","11","12"}: return "rem_alta>5SM"
    return "rem_ignorada"
rem = share("faixa_remuneracao", map_rem)

# --- escolaridade: vem CRUA do interim (código 1..11) -> agrupa aqui p/ leitura ---
def map_esc(v):
    return {"1":"ate_fund_incompleto","2":"ate_fund_incompleto","3":"ate_fund_incompleto",
            "4":"ate_fund_incompleto","5":"fundamental","6":"medio_incompleto",
            "7":"medio_completo","8":"superior_incompleto","9":"superior",
            "10":"superior","11":"superior"}.get(str(v),"nao_informado")
esc = share("escolaridade", map_esc)

# --- setor (1º dígito natureza jurídica) ---
setor = share("natureza_setor", lambda v: {"1":"publico","2":"privado","3":"sem_fins"}.get(str(v),"outro_setor"))

# --- simples / intermitente ---
simples = share("simples", lambda v: "simples" if str(v)=="1" else "nao_simples")
interm = share("intermitente", lambda v: "intermitente" if str(v)=="1" else "nao_interm")

# --- tamanho estab (faixa empregados) ---
TAM={"1":"0","2":"1-4","3":"5-9","4":"10-19","5":"20-49","6":"50-99","7":"100-249",
     "8":"250-499","9":"500-999","10":"1000+"}
tam = share("tamanho_estab", lambda v: "micro_peq(<=49)" if str(v) in {"1","2","3","4","5"}
            else ("media(50-249)" if str(v) in {"6","7"} else "grande(250+)"))

# --- distintivos (top lift) p/ cbo1, cnae2, uf ---
def distinctive(feat, topn=2):
    c = ct(feat)
    glob = c.groupby("valor")["n"].sum(); glob = glob/glob.sum()
    out={}
    for k,sub in c.groupby(KCOL):
        s = sub.set_index("valor")["n"]; s = s/s.sum()
        lift = (s/glob).reindex(s.index)
        # exige share mínimo p/ não pegar raros
        cand = [(v, s[v], lift[v]) for v in s.index if s[v]>=0.05]
        cand.sort(key=lambda x:-x[2])
        out[k]=cand[:topn]
    return out
cbo1_d = distinctive("cbo1"); cnae2_d = distinctive("cnae2"); uf_d = distinctive("uf")

# --- monta tabela ---
def col(df,c): return df[c] if c in df.columns else pd.Series(0.0, index=df.index)
prof = pd.DataFrame({"categoria": base[KCOL].astype(int)})
prof["n"]=base["n"].astype(int); prof["pct_total"]=base["pct_total"].round(2)
prof["taxa_y"]=(base["y_mean"]*100).round(2); prof["prob_media"]=(base["prob_desligamento_mean"]*100).round(2)
prof["idade_media"]=base["idade_mean"].round(1); prof["tempo_anos"]=base["tempo_anos"].round(2)
prof["dias_afast"]=base["qtd_dias_afastamento_mean"].round(1)
for k,df,c in [("clt_indet%",tv,"clt_indet"),("temp_det%",tv,"temp_determinado"),
               ("estatut%",tv,"estatutario"),("verde_amarelo%",tv,"verde_amarelo"),
               ("rem_baixa%",rem,"rem_baixa<=1SM"),("rem_alta%",rem,"rem_alta>5SM"),
               ("superior%",esc,"superior"),("ate_fund%",esc,"ate_fund_incompleto"),
               ("publico%",setor,"publico"),("semfins%",setor,"sem_fins"),
               ("simples%",simples,"simples"),("interm%",interm,"intermitente"),
               ("micro_peq%",tam,"micro_peq(<=49)"),("grande%",tam,"grande(250+)")]:
    prof[k]=col(df,c).reindex(prof["categoria"].values).round(1).values
prof["cbo1_distintivo"]=[ "; ".join(f"{v}({s*100:.0f}%/lift{l:.1f})" for v,s,l in cbo1_d.get(k,[])) for k in prof["categoria"]]
prof["cnae2_distintivo"]=[ "; ".join(f"{v}({s*100:.0f}%/lift{l:.1f})" for v,s,l in cnae2_d.get(k,[])) for k in prof["categoria"]]
prof["uf_distintiva"]=[ "; ".join(f"{v}({s*100:.0f}%/lift{l:.1f})" for v,s,l in uf_d.get(k,[])) for k in prof["categoria"]]

prof.to_csv(f"outputs/tables/persona_categorias{SUF}.csv", index=False)
print(f"MODO={MODO} | total vínculos={int(prof['n'].sum()):,}")
pd.set_option("display.width",250, "display.max_columns",60)
print(prof.to_string(index=False))
print(f"\nsalvo: outputs/tables/persona_categorias{SUF}.csv")
