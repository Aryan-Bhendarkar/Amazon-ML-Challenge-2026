"""Stage 2 on FULL-competition stage-1 predictions (test_v2.py --split train / test).

Every train S1 was queried (like test), so record-level competition is test-like. Stage 2 = LightGBM on
prob-derived features only (applies to the saved test_pred.parquet without re-running the test pipeline):
  record side : how many S1 claim the record, best / 2nd prob, margin to best OTHER S1, rank in record
  S1 side     : rank in S1, S1 top-k profile, gaps to previous / next candidate, counts above thresholds
S1 split (stage-1 pruner/matcher S1 are in-sample -> excluded from stage-2 training, kept as competitors):
  train = folds 2-4 minus stage-1 S1 (sample), es = fold 1 minus stage-1 S1 (sample), eval = mini.
Decision: assign_best_s1 on the stage-2 prob -> global threshold tuned on mini.

    python pipelines/stage2_v2.py --run-id <model_v2 run> [--n-train 400000] [--test]
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from ber import harness, io, metric, paths, split
from ber.decision import assign_best_s1, threshold_matches, tune_threshold
from ber.tracking import Run

SEED = 42
CANDS = paths.DATA_DIR / "cands" / "v2"
LGB = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=0,
           deterministic=True, force_row_wise=True)


def s2_features(p: pl.DataFrame) -> pl.DataFrame:
    """p: s1_id, cand_id, prob (all pairs >= floor, every S1 of the split). Returns p + features."""
    p = p.with_columns(pl.col("prob").cast(pl.Float32))
    r, s = "cand_id", "s1_id"
    p = p.with_columns(
        pl.len().over(r).cast(pl.Int16).alias("rec_n"),
        (pl.col("prob") >= 0.5).sum().over(r).cast(pl.Int16).alias("rec_n50"),
        pl.col("prob").rank("ordinal", descending=True).over(r).cast(pl.Int16).alias("rec_rank"),
        pl.col("prob").max().over(r).alias("rec_max"),
        pl.col("prob").sort(descending=True).over(r, mapping_strategy="join").list.get(1, null_on_oob=True)
          .fill_null(0.0).alias("rec_2nd"),
        pl.len().over(s).cast(pl.Int16).alias("s1_n"),
        pl.col("prob").rank("ordinal", descending=True).over(s).cast(pl.Int16).alias("s1_rank"),
        pl.col("prob").max().over(s).alias("s1_top"),
        pl.col("prob").sum().over(s).alias("s1_sum"),
        (pl.col("prob") >= 0.5).sum().over(s).cast(pl.Int16).alias("s1_n50"),
        (pl.col("prob") >= 0.75).sum().over(s).cast(pl.Int16).alias("s1_n75"),
        (pl.col("prob") >= 0.9).sum().over(s).cast(pl.Int16).alias("s1_n90"),
    )
    p = p.with_columns(
        pl.when(pl.col("rec_rank") == 1).then(pl.col("prob") - pl.col("rec_2nd"))
          .otherwise(pl.col("prob") - pl.col("rec_max")).alias("rec_margin"),
        (pl.col("s1_top") - pl.col("prob")).alias("s1_gap_top"),
    )
    # S1 profile + neighbour gaps (sorted within S1)
    p = p.sort([s, "prob"], descending=[False, True])
    p = p.with_columns(
        (pl.col("prob").shift(1).over(s).fill_null(1.0) - pl.col("prob")).alias("gap_prev"),
        (pl.col("prob") - pl.col("prob").shift(-1).over(s).fill_null(0.0)).alias("gap_next"),
    )
    prof = p.group_by(s).agg(pl.col("prob").head(6).alias("_t"))
    prof = prof.select(s, *[pl.col("_t").list.get(k, null_on_oob=True).fill_null(0.0).alias(f"s1_p{k + 1}")
                            for k in range(1, 6)])
    # competition-adjusted S1 profile: count of candidates this S1 wins (rank 1 in record) above thresholds
    p = p.with_columns(((pl.col("rec_rank") == 1) & (pl.col("prob") >= 0.5)).sum().over(s).cast(pl.Int16).alias("s1_win50"),
                       ((pl.col("rec_rank") == 1) & (pl.col("prob") >= 0.75)).sum().over(s).cast(pl.Int16).alias("s1_win75"))
    return p.join(prof, on=s, how="left")


FEATS = ["prob", "rec_n", "rec_n50", "rec_rank", "rec_max", "rec_2nd", "rec_margin", "s1_n", "s1_rank", "s1_top",
         "s1_sum", "s1_n50", "s1_n75", "s1_n90", "s1_gap_top", "gap_prev", "gap_next", "s1_win50", "s1_win75"] + \
        [f"s1_p{k}" for k in range(2, 7)]


def main(a):
    t0 = time.time()
    art = paths.ART_DIR / a.run_id
    P = s2_features(pl.read_parquet(art / "train_pred.parquet"))
    print(f"[features] {P.height:,} pairs, {P['s1_id'].n_unique():,} S1 ({time.time() - t0:.0f}s)", flush=True)
    folds = io.load_folds()
    stage1 = set(pl.read_parquet(CANDS / "pruned_k40_nt.parquet", columns=["s1_id", "role"])
                   .filter(pl.col("role") != "eval")["s1_id"].unique().to_list())
    rng = np.random.default_rng(SEED)
    f2 = folds[(folds.fold >= 2) & ~folds.s1_id.isin(stage1)].s1_id.to_numpy()
    f1 = folds[(folds.fold == 1) & ~folds.s1_id.isin(stage1)].s1_id.to_numpy()
    tr_ids = rng.choice(np.sort(f2), size=min(a.n_train, len(f2)), replace=False)
    es_ids = rng.choice(np.sort(f1), size=min(a.n_train // 4, len(f1)), replace=False)
    ctx = harness.EvalContext.load(a.subset)
    gt = pl.from_pandas(io.load_gt_pairs(set(tr_ids) | set(es_ids) | ctx.ids)[["s1_id", "match_id"]])
    P = P.join(gt.rename({"match_id": "cand_id"}).with_columns(pl.lit(1, dtype=pl.Int8).alias("label")),
               on=["s1_id", "cand_id"], how="left").with_columns(pl.col("label").fill_null(0))
    role = pl.DataFrame({"s1_id": np.concatenate([tr_ids, es_ids, np.array(sorted(ctx.ids))]),
                         "r": ["train"] * len(tr_ids) + ["es"] * len(es_ids) + ["eval"] * len(ctx.ids)})
    D = P.join(role, on="s1_id", how="inner")
    del P
    tr = D.filter(pl.col("r") != "eval").to_pandas()
    ev = D.filter(pl.col("r") == "eval").to_pandas()
    del D
    with Run(a.name, hypothesis="stage-2 on full-competition stage-1 probs (record + S1 profile features)",
             parent=a.run_id, tags=["decision"], params={"n_train": a.n_train, "lgb": LGB, "feats": FEATS}) as run:
        # honest stage-1 reference under full competition
        a1 = assign_best_s1(ev[["s1_id", "cand_id", "prob"]])
        t1, f1_, _ = tune_threshold(a1, ctx.truth)
        f1_at75 = metric.macro_f05(threshold_matches(a1, 0.75), ctx.truth)
        rep1 = metric.report(threshold_matches(a1, 0.75), ctx.truth, ctx.country)
        print(f"[stage-1 full competition] mini F0.5 @0.75 = {f1_at75:.4f} (by country {rep1['f05_by_country']}); "
              f"tuned {f1_:.4f} @t={t1}", flush=True)
        run.log(stage1_full_comp_f05_at_075=f1_at75, stage1_full_comp_tuned=f1_, stage1_full_comp_t=t1,
                stage1_full_comp_report=rep1)
        es = (tr.r == "es").to_numpy()
        dtr = lgb.Dataset(tr.loc[~es, FEATS], tr.loc[~es, "label"])
        des = lgb.Dataset(tr.loc[es, FEATS], tr.loc[es, "label"], reference=dtr)
        m = lgb.train(LGB, dtr, 3000, valid_sets=[des], callbacks=[lgb.early_stopping(100, verbose=False),
                                                                   lgb.log_evaluation(200)])
        m.save_model(str(run.art_dir / "stage2.lgb"))
        imp = dict(sorted(zip(FEATS, m.feature_importance("gain").round(1).tolist()), key=lambda x: -x[1]))
        run.log(best_iter=m.best_iteration, feature_gain=imp)
        pred = ev[["s1_id", "cand_id"]].assign(prob=m.predict(ev[FEATS], num_threads=0).astype(np.float32))
        out = harness.log_predictions(run, pred, ctx)
        t2 = out["val"]["threshold"]
        (run.art_dir / "decision.json").write_text(json.dumps({"threshold": t2}))
        run.note(f"stage-1 under full competition: mini {f1_at75:.4f} @0.75 (tuned {f1_:.4f} @{t1}); "
                 f"stage-2: {out['val']['f05_macro']:.4f} @{t2}. Top feats {list(imp)[:8]}")
        if a.test:
            T = s2_features(pl.read_parquet(art / "test_pred.parquet")).to_pandas()
            T["prob"] = m.predict(T[FEATS], num_threads=0).astype(np.float32)
            mm = threshold_matches(assign_best_s1(T[["s1_id", "cand_id", "prob"]]), t2)
            pd.DataFrame([(s, x) for s, xs in mm.items() for x in xs], columns=["s1_id", "match_id"]) \
              .to_parquet(run.art_dir / "test_matches.parquet")
            print(f"[test] {sum(map(len, mm.values())):,} matches for {len(mm):,} S1 -> {run.art_dir / 'test_matches.parquet'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--subset", default="mini")
    ap.add_argument("--n-train", type=int, default=400_000)
    ap.add_argument("--name", default="stage2-v2-fullcomp")
    ap.add_argument("--test", action="store_true")
    main(ap.parse_args())
