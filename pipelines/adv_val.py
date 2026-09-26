"""D1 adversarial validation (lead 19:00): can a classifier tell val pairs from test pairs on the model's features?
Pairs with stage-1 p >= 0.5 (psemb), US+IN only (country only selects rows; never a feature), N per side.
LightGBM val-vs-test, 3-fold AUC; top features by gain with val/test quantiles. Label-free (no match labels used).

    python pipelines/adv_val.py [--countries US,India] [--n 300000]
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import features_v1 as fv
from ber import ctx_features as cf
from ber import harness, paths
from ber.tracking import Run

RUN = "20260926-1737_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3-psemb"   # default; --run overrides


def main(a):
    os.environ.setdefault("POLARS_MAX_THREADS", str(a.threads))
    art = paths.ART_DIR / a.run
    feats = json.loads((art / "features.json").read_text())
    raw = cf.expand_derived(feats)
    cty = set(a.countries.split(","))
    rng = np.random.default_rng(42)
    # val side
    ctx = harness.EvalContext.load("mini")
    vids = [k for k, c in ctx.country.items() if c in cty]
    vp = pl.read_parquet(art / "val_pred.parquet").filter(pl.col("s1_id").is_in(vids) & (pl.col("prob") >= 0.5))
    fv.CTX_VER = 3
    V = fv.load_cache("v1_n2", "mini").filter(pl.col("s1_id").is_in(vids))      # whole S1 lists (ps is group-level)
    V = V.select(list(dict.fromkeys(["s1_id", "cand_id"] + [f for f in raw if f in V.columns])))
    V = V.join(pl.read_parquet(fv.ctx_path("v1_n2", "mini")).select(["s1_id", "cand_id"] + [f for f in raw if f not in V.columns
                                                                                             and f not in cf.EMB_FEATS]),
               on=["s1_id", "cand_id"], how="left")
    V = cf.add_derived(cf.join_emb(V, "v1_n2", "mini", feats), feats).join(vp.select("s1_id", "cand_id"), on=["s1_id", "cand_id"])
    V = V.sample(min(a.n, V.height), seed=42)
    # test side
    s1t = pl.read_parquet(cf._norm_file("test", 1, 1), columns=["entity_id", "country"]).filter(pl.col("country").is_in(list(cty)))
    tids = s1t["entity_id"].sample(fraction=min(1.0, 4 * a.n / (s1t.height * 3.3)), seed=42)
    tp = pl.read_parquet(art / "test_pred.parquet").filter(pl.col("s1_id").is_in(tids.implode()) & (pl.col("prob") >= 0.5))
    T = (pl.scan_parquet(paths.DATA_DIR / "cands" / "v1_n2" / "test_feats_g15_ctx3.parquet")
         .select(list(dict.fromkeys(["s1_id", "cand_id"] + [f for f in raw if f not in cf.EMB_FEATS])))
         .filter(pl.col("s1_id").is_in(tids.implode())).collect())
    T = cf.add_derived(cf.join_emb(T, "v1_n2", "test", feats), feats).join(tp.select("s1_id", "cand_id"), on=["s1_id", "cand_id"])
    T = T.sample(min(a.n, T.height), seed=42)
    X = pd.concat([V.select(feats).to_pandas(), T.select(feats).to_pandas()], ignore_index=True)
    y = np.r_[np.zeros(V.height), np.ones(T.height)]
    oof = np.zeros(len(y))
    gain = pd.Series(0.0, index=feats)
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=42).split(X, y):
        m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                           verbose=-1, seed=42, num_threads=a.threads),
                      lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=[c for c in fv.CAT_BASE + fv.CAT_NEW if c in feats]),
                      300)
        oof[te] = m.predict(X.iloc[te], num_threads=a.threads)
        gain += pd.Series(m.feature_importance("gain"), index=feats)
    auc = roc_auc_score(y, oof)
    top = gain.sort_values(ascending=False).head(15)
    rows = []
    for f in top.index:
        qv = np.nanquantile(X.loc[y == 0, f].astype(float), [0.1, 0.5, 0.9])
        qt = np.nanquantile(X.loc[y == 1, f].astype(float), [0.1, 0.5, 0.9])
        rows.append({"feature": f, "gain_share": round(top[f] / gain.sum(), 3), "val_q10_50_90": np.round(qv, 3).tolist(),
                     "test_q10_50_90": np.round(qt, 3).tolist(),
                     "val_nan": round(float(X.loc[y == 0, f].isna().mean()), 3), "test_nan": round(float(X.loc[y == 1, f].isna().mean()), 3)})
    with Run(f"adv-val-{'-'.join(sorted(cty))}", hypothesis="D1: val vs test separability on the psemb features (p >= 0.5)",
             params=vars(a), tags=["diagnostic", "adversarial"], parent=a.run) as run:
        run.log(auc=round(auc, 4), n_val=V.height, n_test=T.height, top=rows)
        print(f"AUC {auc:.4f} (n val {V.height:,}, test {T.height:,})")
        print(pd.DataFrame(rows).to_string(index=False))
        run.note(f"{sorted(cty)}: AUC {auc:.4f}; top: {[r['feature'] for r in rows[:8]]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", default="US,India")
    ap.add_argument("--n", type=int, default=300_000)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--run", default=RUN)
    main(ap.parse_args())
