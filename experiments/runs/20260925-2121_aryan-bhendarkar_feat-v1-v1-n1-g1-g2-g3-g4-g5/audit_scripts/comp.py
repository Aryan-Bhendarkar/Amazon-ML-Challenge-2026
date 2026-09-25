import os; os.environ["POLARS_MAX_THREADS"]="2"
import sys; sys.path.insert(0,"src")
import numpy as np, pandas as pd
from ber import io, harness, metric
from ber.decision import assign_best_s1, threshold_matches, tune_threshold
ctx = harness.EvalContext.load("mini")
gt = io.load_gt_pairs()
owner = dict(zip(gt.match_id, gt.s1_id))
runs = {"parent":("20260925-2015_aryan-bhendarkar_gate-blocking-v1-n1",0.725),
        "new":("20260925-2121_aryan-bhendarkar_feat-v1-v1-n1-g1-g2-g3-g4-g5",0.75)}
ids = set(ctx.ids)
for k,(r,t) in runs.items():
    p = pd.read_parquet(f"artifacts/{r}/val_pred.parquet")
    p = p[p.s1_id.isin(ids)]
    a = assign_best_s1(p)
    own = a.cand_id.map(owner)
    a["own_cls"] = np.where(own.isna(), "none", np.where(own == a.s1_id, "self", np.where(own.isin(ids), "mini_other", "nonquery")))
    m = threshold_matches(a, t); f = metric.macro_f05(m, ctx.truth)
    fired = a[a.prob >= t]
    print(k, "F", round(f,5), "fired by owner class:", fired.own_cls.value_counts().to_dict())
    # oracle competition: nonquery-owned records removed (owner always wins at test)
    b = a[a.own_cls != "nonquery"]
    t2, f2, _ = tune_threshold(b, ctx.truth)
    f2t = metric.macro_f05(threshold_matches(b, t), ctx.truth)
    print(k, "oracle-owner-wins F@t", round(f2t,5), "retuned", round(f2,5), t2)
    # realistic-ish: nonquery-owned record removed only if not ... n/a
