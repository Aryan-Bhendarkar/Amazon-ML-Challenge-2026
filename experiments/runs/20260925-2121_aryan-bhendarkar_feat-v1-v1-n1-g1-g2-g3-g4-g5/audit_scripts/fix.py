import os; os.environ["POLARS_MAX_THREADS"]="2"
import sys, json, time; sys.path.insert(0,"src")
import numpy as np, pandas as pd, polars as pl, lightgbm as lgb
from ber import ctx_features as cf, harness, io, metric
from ber.decision import assign_best_s1, threshold_matches, tune_threshold
RUN="20260925-2121_aryan-bhendarkar_feat-v1-v1-n1-g1-g2-g3-g4-g5"; art=f"artifacts/{RUN}"
feats=json.load(open(f"{art}/features.json")); model=lgb.Booster(model_file=f"{art}/model.lgb"); T=0.75
ctx=harness.EvalContext.load("mini"); ids=np.sort(np.array(list(ctx.ids)))
rng=np.random.default_rng(7); samp=set(rng.choice(ids,15000,replace=False).tolist())
base=pl.read_parquet("data/cands/v1_n1/mini.parquet").filter(pl.col("s1_id").is_in(list(samp)))
c1=pl.read_parquet("data/cands/v1_n1/ctx1_mini.parquet").filter(pl.col("s1_id").is_in(list(samp)))
newc=[c for c in c1.columns if c not in ("s1_id","cand_id")]
have=[f for f in feats if f in base.columns]
truth={k:ctx.truth[k] for k in samp}; cty={k:ctx.country[k] for k in samp}
def score(X, tag):
    Xp=X.select(["s1_id","cand_id"]+feats).to_pandas()
    p=Xp[["s1_id","cand_id"]].assign(prob=model.predict(Xp[feats],num_threads=2))
    a=assign_best_s1(p); m=threshold_matches(a,T); r=metric.report(m,truth,cty)
    t2,f2,_=tune_threshold(a,truth)
    fired=(a.prob>=T).sum()
    print(f"{tag}: F@.75={r['f05_macro']:.5f} byc={ {k:round(v,5) for k,v in r['f05_by_country'].items()} } P={r['pair_precision']:.4f} R={r['pair_recall']:.4f} fired={fired} retuned t={t2} F={f2:.5f}", flush=True)
    return p
X0=base.select(["s1_id","cand_id"]+have).join(c1,on=["s1_id","cand_id"],how="left")
p0=score(X0,"cached ctx1 (as in run)")
# helpers
s1n=(pl.scan_parquet(cf._norm_file("train",1)).select("entity_id","country","n_core","a_full","a_street","a_house").collect())
pooln=pl.concat([pl.scan_parquet(cf._norm_file("train",s)).select("entity_id","country","n_core","a_full").collect() for s in (2,3)])
gt=pl.from_pandas(io.load_gt_pairs()[["s1_id","match_id"]])
def build(s1, pool, nref=None):
    s1=s1.with_columns(cf.key_exprs()); pool=pool.with_columns(cf.key_exprs()[:2]).select("country","n_key","a_key")
    cnt=lambda df,k: df.filter(pl.col(k)!="").group_by("country",k).len("cnt")
    n_s1=s1.group_by("country").len("n_s1")
    df=(s1.select("country",cf._tokens("n_core").list.unique().alias("tok")).explode("tok").drop_nulls("tok").group_by("country","tok").len("df"))
    n_s1=n_s1 if nref is None else n_s1.with_columns(pl.lit(float(nref)).alias("n_s1"))
    idf=df.join(n_s1,on="country").select("country","tok",(pl.col("n_s1")/pl.col("df")).log().cast(pl.Float32).alias("idf"))
    return cf.SplitContext(cnt(s1,"n_key"),cnt(s1,"a_key"),cnt(s1,"hs_key"),cnt(pool,"n_key"),cnt(pool,"a_key"),idf)
def feat(sctx, B=None):
    B=base if B is None else B
    P=B.select("s1_id","cand_id","name_tset")
    F=cf.add_features(cf.attach_norm(P,"train"),sctx,workers=2)
    return B.select(["s1_id","cand_id"]+have).join(F.select(["s1_id","cand_id"]+newc),on=["s1_id","cand_id"],how="left")
t0=time.time()
Xr=X0
d=X0.select(["s1_id","cand_id"]+newc).join(Xr.select(["s1_id","cand_id"]+newc),on=["s1_id","cand_id"],suffix="_r")
bad=[c for c in newc if (d[c].cast(pl.Float64).fill_null(-999)-d[c+"_r"].cast(pl.Float64).fill_null(-999)).abs().max()>1e-5]
print("recompute==cache mismatched cols:", bad, f"{time.time()-t0:.0f}s", flush=True)

allmini=set(ctx.ids)
ntr=s1n.group_by("country").len("n")
NREF=float(ntr["n"].mean()); print("train n_s1", dict(zip(ntr["country"],ntr["n"])), "NREF", NREF, flush=True)
# (0) full-density with NREF idf (what the model would see on India/US test if the fix is applied at test only)
score(feat(build(s1n,pooln,nref=NREF)),"r=1.0 idf NREF (post-hoc fix at full density)")
for r in (0.5,0.2):
    rg=np.random.default_rng(42)
    nq=s1n.filter(~pl.col("entity_id").is_in(list(allmini)))
    keep_nq=nq.filter(pl.Series(rg.random(nq.height)<r))
    s1k=pl.concat([s1n.filter(pl.col("entity_id").is_in(list(allmini))),keep_nq])
    dropped=set(nq["entity_id"].to_list())-set(keep_nq["entity_id"].to_list())
    dm=set(gt.filter(pl.col("s1_id").is_in(list(dropped)))["match_id"].to_list())
    pk=pooln.filter(~pl.col("entity_id").is_in(list(dm)))
    score(feat(build(s1k,pk,nref=NREF)),f"r={r} idf NREF post-hoc")
    Bc=base.filter(~pl.col("cand_id").is_in(list(dm)))
    print(f"r={r} clean world: removed {base.height-Bc.height:,} orphaned candidate rows", flush=True)
    score(feat(build(s1k,pk),Bc),f"r={r} CLEAN (orphans removed) idf as-is")
    score(feat(build(s1k,pk,nref=NREF),Bc),f"r={r} CLEAN + idf NREF")
print("done")
