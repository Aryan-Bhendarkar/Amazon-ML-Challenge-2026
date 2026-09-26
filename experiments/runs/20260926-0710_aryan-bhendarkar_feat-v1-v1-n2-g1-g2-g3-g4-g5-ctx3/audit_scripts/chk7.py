import os; os.environ["POLARS_MAX_THREADS"]="2"
import polars as pl, numpy as np, pyarrow.parquet as pq
from rapidfuzz import process, fuzz
SP="/tmp/audit_scratch"
FD="/home/ubuntu/amlc/data/features"; A="/home/ubuntu/amlc/artifacts/20260926-0710_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3"
fr=pl.read_parquet(SP+"/fr512.parquet").with_columns((pl.col("both")&(pl.col("ssim")<50)).alias("fp_h"))
mn=pl.read_parquet(A+"/test_matches.parquet")
s1c=pl.read_parquet(f"{FD}/norm_v1_test_s1.parquet",columns=["entity_id","country","a_street"]).rename({"entity_id":"s1_id"})
FRs=s1c.filter(pl.col("country")=="France")
# per-S1 impact under "fp_h => FP, else TP; other preds all correct; truth complete"
m=mn.join(FRs.select("s1_id"),on="s1_id").rename({"match_id":"cand_id"}).join(fr.select("s1_id","cand_id","fp_h"),on=["s1_id","cand_id"],how="left")
g=m.group_by("s1_id").agg(pl.col("fp_h").is_null().sum().alias("n_other"),(pl.col("fp_h")==False).sum().alias("tp5"),(pl.col("fp_h")==True).sum().alias("fp5")).filter((pl.col("tp5")+pl.col("fp5"))>0)
def f05(tp,fp,nt):
    if nt==0: return 1.0 if tp+fp==0 else 0.0
    if tp+fp==0: return 0.0
    p=tp/(tp+fp); r=tp/nt
    return 0 if p+r==0 else 1.25*p*r/(0.25*p+r)
d=[f05(o+t,f,o+t)-f05(o,0,o+t) for o,t,f in g.select("n_other","tp5","fp5").iter_rows()]
d=np.array(d); n=FRs.height
print("FR S1 with 512 matches", g.height, "sum delta", d.sum(), "FR F0.5 delta", d.sum()/n, "overall (x0.15)", d.sum()/n*0.15)
print("breakdown: gain S1s", (d>0).sum(), "loss S1s", (d<0).sum(), "only-512 S1s", g.filter(pl.col("n_other")==0).height, " of which all-fp", g.filter((pl.col("n_other")==0)&(pl.col("tp5")==0)).height)
# alt: what if 512 matches are 100% correct / fp rate from eyeball 30%
d2=[f05(o+t+f,0,o+t+f)-f05(o,0,o+t+f) for o,t,f in g.select("n_other","tp5","fp5").iter_rows()]; print("if all correct: FR delta", sum(d2)/n, "overall", sum(d2)/n*.15)
# competitor owners for fp_h pairs
fp=fr.filter(pl.col("fp_h")).select("s1_id","cand_id")
pr=pl.read_parquet(A+"/test_pred.parquet")
comp=pr.join(fp.select("cand_id"),on="cand_id").join(fp.rename({"s1_id":"winner"}),on="cand_id").filter(pl.col("s1_id")!=pl.col("winner"))
c=pl.concat([pl.read_parquet(f"{FD}/norm_v1_test_s{k}.parquet",columns=["entity_id","a_street"]) for k in (2,3)]).rename({"entity_id":"cand_id","a_street":"st_c"})
comp=comp.join(s1c.select("s1_id","a_street"),on="s1_id").join(c,on="cand_id")
comp=comp.with_columns(pl.Series("csim",process.cpdist(comp["a_street"].fill_null("").to_list(),comp["st_c"].fill_null("").to_list(),scorer=fuzz.token_set_ratio,workers=2)))
best=comp.sort("prob",descending=True).group_by("cand_id").first()
print("fp_h pairs", fp.height, "with any competitor S1 (p>=0.01)", best.height, "competitor p>=0.775 (stolen true match?)", best.filter(pl.col("prob")>=0.775).height,
      "competitor street sim>=90", best.filter(pl.col("csim")>=90).height)
print(best.select(pl.col("prob").quantile(0.5).alias("p50"),pl.col("csim").median().alias("csim_med")))
