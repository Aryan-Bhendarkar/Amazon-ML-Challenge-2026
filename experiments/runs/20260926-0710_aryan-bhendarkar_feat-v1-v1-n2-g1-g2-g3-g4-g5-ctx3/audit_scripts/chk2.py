import os; os.environ["POLARS_MAX_THREADS"]="2"
import polars as pl, pyarrow.parquet as pq, collections, time
D="/home/ubuntu/amlc/data/cands"
cm=pl.read_parquet("/home/ubuntu/amlc/data/features/norm_v1_test_s1.parquet",columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
cols=["s1_id","cand_id","rbits","name_tset","addr_tset","name_rank_in_s1","addr_rank_in_s1","name_gap_s1","n_cand_s1","cos_name","kmask"]
for v in ["v1_n1","v1_n2"]:
    pf=pq.ParquetFile(f"{D}/{v}/test.parquet"); t0=time.time()
    seen=set(); span=0; dup=0; mm=[0,0,0,0]; agg=[]; n=0
    for rg in range(pf.num_row_groups):
        df=pl.from_arrow(pf.read_row_group(rg,columns=cols))
        s=set(df["s1_id"].unique().to_list()); span+=len(s&seen); seen|=s
        dup+=df.height-df.select("s1_id","cand_id").unique().height
        r=df.with_columns(pl.col("name_tset").rank("min",descending=True).over("s1_id").cast(pl.Float32).alias("nr2"),
            pl.col("addr_tset").rank("min",descending=True).over("s1_id").cast(pl.Float32).alias("ar2"),
            (pl.col("name_tset").max().over("s1_id")-pl.col("name_tset")).alias("ng2"),pl.len().over("s1_id").alias("nc2"))
        x=r.select(((pl.col("nr2")-pl.col("name_rank_in_s1")).abs()>1e-6).sum(),((pl.col("ar2")-pl.col("addr_rank_in_s1")).abs()>1e-6).sum(),
                   ((pl.col("ng2")-pl.col("name_gap_s1")).abs()>1e-4).sum(),(pl.col("nc2")!=pl.col("n_cand_s1")).sum()).row(0)
        mm=[a+b for a,b in zip(mm,x)]
        agg.append(df.join(cm,on="s1_id",how="left").group_by("country").agg(pl.len().alias("pairs"),pl.col("s1_id").n_unique().alias("s1"),
            (pl.col("rbits")==512).sum().alias("n512"),((pl.col("rbits")&512)>0).sum().alias("b512"),pl.col("cos_name").is_null().sum().alias("cosnull"),
            (pl.col("n_cand_s1")>100).sum().alias("gt100")))
        n+=df.height
    a=pl.concat(agg).group_by("country").sum().with_columns((pl.col("n512")/pl.col("s1")).alias("n512_per_s1"),(pl.col("pairs")/pl.col("s1")).alias("pairs_per_s1"))
    print(v,"rgs",pf.num_row_groups,"rows",n,"S1",len(seen),"S1 spanning rgs",span,"dup",dup,"rank mismatch",mm, f"{time.time()-t0:.0f}s")
    print(a)
