import os; os.environ["POLARS_MAX_THREADS"]="2"
import polars as pl, sys
sys.path.insert(0,"/home/ubuntu/amlc/src")
from ber import io
A="/home/ubuntu/amlc/artifacts/20260926-0710_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3"
C=["s1_id","cand_id","rbits","street_tset","addr_tset","house_rel","s1_name_c","s1_name_self","sib_n","sib_house_agree","hs_key_equal","a_key_equal"]
cm=pl.read_parquet("/home/ubuntu/amlc/data/features/norm_v1_test_s1.parquet",columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
mn=pl.read_parquet(A+"/test_matches.parquet").rename({"match_id":"cand_id"})
T=pl.scan_parquet("/home/ubuntu/amlc/data/cands/v1_n2/test_feats_g15_ctx3.parquet").select(C).filter(pl.col("rbits")==512).collect()
T=T.join(mn,on=["s1_id","cand_id"],how="semi").join(cm,on="s1_id")
V=pl.read_parquet(A+"/val_pred.parquet").filter(pl.col("prob")>=0.775)
M=pl.read_parquet("/home/ubuntu/amlc/data/cands/v1_n2/mini.parquet",columns=["s1_id","cand_id","rbits","label","street_tset","addr_tset","house_rel"]).filter(pl.col("rbits")==512)
F=pl.read_parquet("/home/ubuntu/amlc/data/cands/v1_n2/ctx3_mini.parquet",columns=["s1_id","cand_id","s1_name_c","s1_name_self","sib_n","sib_house_agree","hs_key_equal","a_key_equal"])
VV=M.join(V.select("s1_id","cand_id"),on=["s1_id","cand_id"],how="semi").join(F,on=["s1_id","cand_id"])
def summ(d,tag):
    print(tag, d.height, d.select(pl.col("street_tset").median().alias("street_med"),(pl.col("street_tset")<60).mean().alias("street<60"),
        pl.col("addr_tset").median().alias("addr_med"),(pl.col("addr_tset")<60).mean().alias("addr<60"),(pl.col("s1_name_self")<=1).mean().alias("uniqname"),
        pl.col("hs_key_equal").mean().alias("hs_eq"),pl.col("a_key_equal").mean().alias("a_eq")).to_dicts())
for c in ["France","India","US"]: summ(T.filter(pl.col("country")==c),"test fired512 "+c)
summ(VV.filter(pl.col("label")),"val fired512 TP"); summ(VV.filter(~pl.col("label")),"val fired512 FP")
# precision of val fired-512 by street bucket
print(VV.with_columns((pl.col("street_tset")<60).alias("lowstreet")).group_by("lowstreet").agg(pl.len(),pl.col("label").mean()))
fr=T.filter(pl.col("country")=="France").with_columns((pl.col("street_tset")<60).alias("lowstreet"))
print("FR fired512 lowstreet share", fr["lowstreet"].mean(), fr.height)
# sample 12 FR fired-512 with low street sim, eyeball
s=fr.filter(pl.col("lowstreet")).sample(10,seed=7)
s1=pl.from_pandas(io.load_source("test",1)).rename({"entity_id":"s1_id"})
pool=pl.concat([pl.from_pandas(io.load_source("test",k)) for k in (2,3)]).rename({"entity_id":"cand_id"})
with pl.Config(tbl_rows=30,fmt_str_lengths=55,tbl_width_chars=250):
    print(s.join(s1,on="s1_id").join(pool,on="cand_id",suffix="_c").select("street_tset","s1_name_self","business_name","business_name_c","business_address","business_address_c"))
    s=fr.sample(20,seed=11)
    print(s.join(s1,on="s1_id").join(pool,on="cand_id",suffix="_c").select("street_tset","business_name","business_name_c","business_address","business_address_c"))
