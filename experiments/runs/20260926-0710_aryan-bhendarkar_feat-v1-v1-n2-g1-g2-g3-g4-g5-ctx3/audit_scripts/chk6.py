import os; os.environ["POLARS_MAX_THREADS"]="2"
import polars as pl, numpy as np, sys
from rapidfuzz import process, fuzz
sys.path.insert(0,"/home/ubuntu/amlc/src")
from ber import io
FD="/home/ubuntu/amlc/data/features"; A="/home/ubuntu/amlc/artifacts/20260926-0710_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3"
def load(split):
    s1=pl.read_parquet(f"{FD}/norm_v1_{split}_s1.parquet",columns=["entity_id","country","a_street"]).rename({"entity_id":"s1_id","a_street":"st"})
    c=pl.concat([pl.read_parquet(f"{FD}/norm_v1_{split}_s{k}.parquet",columns=["entity_id","a_street"]) for k in (2,3)]).rename({"entity_id":"cand_id","a_street":"st_c"})
    toks=s1.select("country",pl.col("st").fill_null("").str.split(" ").list.unique().alias("t")).explode("t").filter(pl.col("t")!="")
    n=s1.group_by("country").len("n")
    common=toks.group_by("country","t").len("df").join(n,on="country").filter(pl.col("df")>0.003*pl.col("n")).select("country","t")
    return s1,c,common
def sim(P,s1,c,common):
    P=P.join(s1,on="s1_id").join(c,on="cand_id")
    cs={}
    for ctry,g in common.group_by("country"): cs[ctry[0]]=set(g["t"].to_list())
    def sig(st,ctry): return " ".join(sorted(t for t in (st or "").split() if t not in cs[ctry] and not t.isdigit()))
    a=[sig(x,k) for x,k in zip(P["st"].to_list(),P["country"].to_list())]
    b=[sig(x,k) for x,k in zip(P["st_c"].to_list(),P["country"].to_list())]
    r=process.cpdist(a,b,scorer=fuzz.token_set_ratio,workers=2)
    ok=np.array([bool(x) and bool(y) for x,y in zip(a,b)])
    return P.with_columns(pl.Series("ssim",r),pl.Series("both",ok))
def rep(P,tag):
    d=P.filter(pl.col("both")); print(f"{tag:40s} n={P.height:>8} both={d.height:>8} diffstreet(<50)={ (d['ssim']<50).mean():.4f}  (<35)={(d['ssim']<35).mean():.4f}")
# --- val (mini), train split stats
s1,c,common=load("train")
V=pl.read_parquet(A+"/val_pred.parquet")
M=pl.read_parquet("/home/ubuntu/amlc/data/cands/v1_n2/mini.parquet",columns=["s1_id","cand_id","rbits","label"])
VM=V.join(M,on=["s1_id","cand_id"])
fired=VM.filter(pl.col("prob")>=0.775)
pos=M.filter(pl.col("label")).sample(150000,seed=1)
P=sim(pos,s1,c,common)
for k in ["US","India"]: rep(P.filter(pl.col("country")==k),f"mini GT positives {k}")
F=sim(fired,s1,c,common)
for k in ["US","India"]:
    rep(F.filter((pl.col("country")==k)&pl.col("label")),f"mini fired TP {k}")
    rep(F.filter((pl.col("country")==k)&~pl.col("label")),f"mini fired FP {k}")
rep(F.filter((pl.col("rbits")==512)&pl.col("label")),"mini fired512 TP"); rep(F.filter((pl.col("rbits")==512)&~pl.col("label")),"mini fired512 FP")
# precision of fired pairs with diffstreet in val
d=F.filter(pl.col("both")&(pl.col("ssim")<50)); print("val fired diffstreet precision", d["label"].mean(), d.height, " by country", d.group_by("country").agg(pl.len(),pl.col("label").mean()).to_dicts())
del s1,c
# --- test
s1,c,common=load("test")
mn=pl.read_parquet(A+"/test_matches.parquet").rename({"match_id":"cand_id"})
import pyarrow.parquet as pq
pf=pq.ParquetFile("/home/ubuntu/amlc/data/cands/v1_n2/test.parquet")
n512=pl.concat([pl.from_arrow(pf.read_row_group(rg,columns=["s1_id","cand_id","rbits"])).filter(pl.col("rbits")==512) for rg in range(pf.num_row_groups)])
m=mn.join(n512,on=["s1_id","cand_id"],how="left").with_columns(pl.col("rbits").fill_null(0))
T=sim(pl.concat([m.filter(pl.col("rbits")==512),m.filter(pl.col("rbits")!=512).sample(400000,seed=2)]),s1,c,common)
for k in ["France","India","US"]:
    rep(T.filter((pl.col("country")==k)&(pl.col("rbits")==512)),f"test fired512 {k}")
    rep(T.filter((pl.col("country")==k)&(pl.col("rbits")!=512)),f"test fired other {k}")
T.filter((pl.col("country")=="France")&(pl.col("rbits")==512)).select("s1_id","cand_id","ssim","both").write_parquet("/tmp/audit_scratch/fr512.parquet")
