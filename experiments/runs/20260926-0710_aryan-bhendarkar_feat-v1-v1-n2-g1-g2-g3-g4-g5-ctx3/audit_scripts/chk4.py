import os; os.environ["POLARS_MAX_THREADS"]="2"
import polars as pl, pyarrow.parquet as pq, sys
sys.path.insert(0,"/home/ubuntu/amlc/src")
from ber import io
A="/home/ubuntu/amlc/artifacts/"
NEW=A+"20260926-0710_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3"; OLD=A+"20260925-2121_aryan-bhendarkar_feat-v1-v1-n1-g1-g2-g3-g4-g5"
cm=pl.read_parquet("/home/ubuntu/amlc/data/features/norm_v1_test_s1.parquet",columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
pf=pq.ParquetFile("/home/ubuntu/amlc/data/cands/v1_n2/test.parquet")
n512=pl.concat([pl.from_arrow(pf.read_row_group(rg,columns=["s1_id","cand_id","rbits","name_tset","addr_tset","n_cand_s1"])).filter(pl.col("rbits")==512) for rg in range(pf.num_row_groups)])
print("512 pairs",n512.height)
mn=pl.read_parquet(NEW+"/test_matches.parquet"); mo=pl.read_parquet(OLD+"/test_matches.parquet")
pr=pl.read_parquet(NEW+"/test_pred.parquet")
m5=mn.join(n512.rename({"cand_id":"match_id"}),on=["s1_id","match_id"],how="inner").join(cm,on="s1_id")
s1n=cm.group_by("country").len("n_s1")
print("test matches from nkey_num-only pairs by country:"); print(m5.group_by("country").len("m512").join(s1n,on="country").with_columns((pl.col("m512")/pl.col("n_s1")).alias("per_s1")))
p5=pr.join(n512,on=["s1_id","cand_id"],how="inner").join(cm,on="s1_id")
print(p5.group_by("country").agg(pl.len(),(pl.col("prob")>=0.775).mean().alias("frac>=t"),pl.col("prob").quantile(0.99).alias("p99"),((pl.col("prob")>=0.3)&(pl.col("prob")<0.775)).sum().alias("grey")))
# val: mini
V=pl.read_parquet(NEW+"/val_pred.parquet"); vm=pl.read_parquet("/home/ubuntu/amlc/data/cands/v1_n2/mini.parquet",columns=["s1_id","cand_id","rbits","label"]).filter(pl.col("rbits")==512)
vv=V.join(vm,on=["s1_id","cand_id"])
f=io.load_folds(); fc=pl.from_pandas(f[["s1_id","country"]])
vv=vv.join(fc,on="s1_id")
print("val 512:"); print(vv.group_by("country").agg(pl.len(),(pl.col("prob")>=0.775).sum().alias("fired"),pl.col("label").sum().alias("pos"),((pl.col("prob")>=0.775)&pl.col("label")).sum().alias("tp")))
print("mini S1 per country", fc.join(pl.DataFrame({"s1_id":V["s1_id"].unique()}),on="s1_id").group_by("country").len())
# diff vs sub3 by country
k=["s1_id","match_id"]
add=mn.join(mo,on=k,how="anti").join(cm,on="s1_id"); rem=mo.join(mn,on=k,how="anti").join(cm,on="s1_id")
print("added vs sub3"); print(add.group_by("country").len().join(s1n,on="country").with_columns((pl.col("len")/pl.col("n_s1")).alias("per_s1")))
print("removed vs sub3"); print(rem.group_by("country").len().join(s1n,on="country").with_columns((pl.col("len")/pl.col("n_s1")).alias("per_s1")))
# sample 12 FR 512 matches + 6 FR 512 grey
fr=m5.filter(pl.col("country")=="France").sample(12,seed=42)
s1=pl.from_pandas(io.load_source("test",1)).rename({"entity_id":"s1_id"})
pool=pl.concat([pl.from_pandas(io.load_source("test",s)) for s in (2,3)]).rename({"entity_id":"match_id"})
pd_=fr.join(s1,on="s1_id").join(pool,on="match_id",suffix="_c").join(pr.rename({"cand_id":"match_id"}),on=["s1_id","match_id"])
with pl.Config(tbl_rows=30,fmt_str_lengths=60,tbl_width_chars=250):
    print(pd_.select("prob","business_name","business_name_c","business_address","business_address_c"))
    g=p5.filter((pl.col("country")=="France")&(pl.col("prob")>=0.3)&(pl.col("prob")<0.775)).sample(6,seed=1).rename({"cand_id":"match_id"})
    print(g.join(s1,on="s1_id").join(pool,on="match_id",suffix="_c").select("prob","business_name","business_name_c","business_address","business_address_c"))
