import os; os.environ["POLARS_MAX_THREADS"]="2"
import polars as pl, numpy as np
D="/home/ubuntu/amlc/data/cands"
cols=["s1_id","cand_id","rbits","label","name_tset","addr_tset","name_rank_in_s1","addr_rank_in_s1","name_gap_s1","n_cand_s1","cos_name","kmask"]
def recompute(df):
    return df.with_columns(
        pl.col("name_tset").rank("min",descending=True).over("s1_id").cast(pl.Float32).alias("nr2"),
        pl.col("addr_tset").rank("min",descending=True).over("s1_id").cast(pl.Float32).alias("ar2"),
        (pl.col("name_tset").max().over("s1_id")-pl.col("name_tset")).alias("ng2"),
        pl.len().over("s1_id").alias("nc2"))
for tag in ["train","mini","fold0x"]:
    for v in ["v1_n1","v1_n2"]:
        df=pl.read_parquet(f"{D}/{v}/{tag}.parquet",columns=cols+(["role"] if tag=="train" else []))
        dup=df.height-df.select("s1_id","cand_id").unique().height
        r=recompute(df)
        mm=r.select(((pl.col("nr2")-pl.col("name_rank_in_s1")).abs()>1e-6).sum().alias("nr"),
                    ((pl.col("ar2")-pl.col("addr_rank_in_s1")).abs()>1e-6).sum().alias("ar"),
                    ((pl.col("ng2")-pl.col("name_gap_s1")).abs()>1e-4).sum().alias("ng"),
                    (pl.col("nc2")!=pl.col("n_cand_s1")).sum().alias("nc")).row(0)
        b=df.group_by(pl.col("rbits")==512).agg(pl.len(),pl.col("label").mean().alias("pos"),pl.col("cos_name").is_null().mean().alias("cosnull"),(pl.col("kmask")==0).mean().alias("k0")).sort("rbits")
        print(tag,v,"rows",df.height,"dup",dup,"S1",df["s1_id"].n_unique(),"rank-mismatch(nr,ar,ng,nc)",mm)
        print(b)
        if v=="v1_n2":
            print("rbits has bit512 but !=512:", df.filter(((pl.col("rbits")&512)>0)&(pl.col("rbits")!=512)).height)
            print("n_cand_s1 >100 frac", (df["n_cand_s1"]>100).mean(), "max", df["n_cand_s1"].max())
            if tag=="train": print(df.filter(pl.col("rbits")==512).group_by("role").agg(pl.len(),pl.col("label").mean()))
