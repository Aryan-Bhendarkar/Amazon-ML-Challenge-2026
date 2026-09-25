import os; os.environ["POLARS_MAX_THREADS"]="2"
import sys, json, time; sys.path.insert(0,"src")
import numpy as np, pandas as pd, polars as pl, lightgbm as lgb, pyarrow.parquet as pq
from ber import ctx_features as cf, io
from ber.decision import assign_best_s1, threshold_matches
NEW="20260925-2121_aryan-bhendarkar_feat-v1-v1-n1-g1-g2-g3-g4-g5"; PAR="20260925-2015_aryan-bhendarkar_gate-blocking-v1-n1"
M={r:(lgb.Booster(model_file=f"artifacts/{r}/model.lgb"),json.load(open(f"artifacts/{r}/features.json")),json.load(open(f"artifacts/{r}/decision.json"))["threshold"]) for r in (NEW,PAR)}
src="data/cands/v1_n1/test.parquet"; pf=pq.ParquetFile(src)
cty=dict(zip(*[io.load_source("test",1,columns=["entity_id","country"])[c] for c in ("entity_id","country")]))
have=set(pf.schema_arrow.names); base=[f for f in M[NEW][1] if f in have]
t0=time.time(); sctx=cf.SplitContext.build("test")
rows=[]; feats_dump=[]
for rg in (1,12,38):
    X=pl.from_arrow(pf.read_row_group(rg,columns=["s1_id","cand_id"]+sorted(set(base)|set(M[PAR][1]))))
    ids=np.sort(X["s1_id"].unique().to_numpy()); keep=np.random.default_rng(1).choice(ids,6000,replace=False)
    X=X.filter(pl.col("s1_id").is_in(keep.tolist()))
    F=cf.add_features(cf.attach_norm(X.select("s1_id","cand_id","name_tset"),"test"),sctx,workers=2)
    newc=[f for f in M[NEW][1] if f not in have]
    X=X.join(F.select(["s1_id","cand_id"]+newc),on=["s1_id","cand_id"],how="left")
    c=cty[X["s1_id"][0]]
    Xp=X.to_pandas()
    for r,(m,fe,t) in M.items():
        p=Xp[["s1_id","cand_id"]].assign(prob=m.predict(Xp[fe],num_threads=2))
        a=assign_best_s1(p); mm=threshold_matches(a,t)
        n=pd.Series({k:len(mm.get(k,[])) for k in keep})
        rows.append(dict(country=c,run=r[:24],empty=round((n==0).mean(),4),mean=round(n.mean(),3),ge6=round((n>=6).mean(),4),
                         band=round(((a.prob>0.2)&(a.prob<0.9)).mean(),4)))
    feats_dump.append(Xp.assign(country=c)[["country"]+newc])
    print(c, f"{time.time()-t0:.0f}s", flush=True)
print(pd.DataFrame(rows).to_string())
D=pd.concat(feats_dump)
cols=["s1_name_c","s1_name_self","s1_addr_c","s1_addr_self","s1_hs_self","pool_name_self","pool_addr_c","ex_idf_max","ex_idf_min","ex_biz","sib_n","sib_house_frac","house_lev"]
print(D.groupby("country")[cols].mean().round(3).T.to_string())
D.to_parquet("/tmp/claude-1000/-home-ubuntu-amlc/ef786237-f2e9-448f-b096-0980f0b0ac02/scratchpad/test_ctx_sample.parquet")
