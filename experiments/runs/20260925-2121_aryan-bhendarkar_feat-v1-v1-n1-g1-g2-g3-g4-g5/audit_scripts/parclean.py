import os; os.environ["POLARS_MAX_THREADS"]="2"
import sys, json; sys.path.insert(0,"src")
import numpy as np, polars as pl, lightgbm as lgb
from ber import ctx_features as cf, harness, io, metric
from ber.decision import assign_best_s1, threshold_matches
ctx=harness.EvalContext.load("mini"); ids=np.sort(np.array(list(ctx.ids)))
rng=np.random.default_rng(7); samp=set(rng.choice(ids,15000,replace=False).tolist())
truth={k:ctx.truth[k] for k in samp}
PAR="20260925-2015_aryan-bhendarkar_gate-blocking-v1-n1"
pm=lgb.Booster(model_file=f"artifacts/{PAR}/model.lgb"); pf_=json.load(open(f"artifacts/{PAR}/features.json"))
base=pl.read_parquet("data/cands/v1_n1/mini.parquet").filter(pl.col("s1_id").is_in(list(samp)))
s1n=pl.scan_parquet(cf._norm_file("train",1)).select("entity_id").collect()
gt=pl.from_pandas(io.load_gt_pairs()[["s1_id","match_id"]]); allmini=set(ctx.ids)
for r in (0.5,0.2):
    rg=np.random.default_rng(42)
    nq=s1n.filter(~pl.col("entity_id").is_in(list(allmini)))
    keep_nq=nq.filter(pl.Series(rg.random(nq.height)<r))
    dropped=set(nq["entity_id"].to_list())-set(keep_nq["entity_id"].to_list())
    dm=set(gt.filter(pl.col("s1_id").is_in(list(dropped)))["match_id"].to_list())
    Bc=base.filter(~pl.col("cand_id").is_in(list(dm)))
    X=Bc.select(["s1_id","cand_id"]+pf_).to_pandas(); p=X[["s1_id","cand_id"]].assign(prob=pm.predict(X[pf_],num_threads=2))
    print(f"r={r} PARENT clean world F@.725", round(metric.macro_f05(threshold_matches(assign_best_s1(p),0.725),truth),5), flush=True)
