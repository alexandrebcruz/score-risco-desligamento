import glob, pandas as pd, pyarrow.parquet as pq, numpy as np
PRED="data/processed/predicoes_2124"; ANOS=[2021,2022,2023,2024]
acc={}  # cat -> [soma_prob, n]
for a in ANOS:
    for fp in sorted(glob.glob(f"{PRED}/ano={a}/*.parquet")):
        d=pq.ParquetFile(fp).read(columns=["categoria_risco","prob_desligamento"]).to_pandas()
        g=d.groupby("categoria_risco")["prob_desligamento"].agg(["sum","count"])
        for c,row in g.iterrows():
            v=acc.get(int(c),[0.0,0]); v[0]+=float(row["sum"]); v[1]+=int(row["count"]); acc[int(c)]=v
    print("ok",a,flush=True)
pm={c:(v[0]/v[1]) for c,v in acc.items()}
esc=pd.read_csv("outputs/tables/binning_infogain_escolhido_2124.csv")
esc["prob_media"]=esc["categoria"].map(pm)
esc.to_csv("outputs/tables/binning_infogain_escolhido_2124.csv",index=False)
print("prob_media gravado:", {c:round(pm[c]*100,2) for c in sorted(pm)})
print("FIM_PROBMEDIA")
