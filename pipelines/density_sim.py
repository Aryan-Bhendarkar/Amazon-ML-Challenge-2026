"""Density-shift stress test for context features (generalised from the v1 audit's dens.py).

Test has fewer S1 per country than train (US ~0.5x, France ~0.2x), so every split-level statistic
(S1 counts, pool counts, token df) is computed at a different density on test. This recomputes ALL
ctx features for a sample of mini S1 with statistics from a thinned train split:
  keep all eval S1, keep a fraction r of the other S1 and drop their GT copies from the pool
  ('clean'), optionally also thin the unmatched pool records ('drop_unm')
and scores the run's frozen model at its frozen threshold.

    python pipelines/density_sim.py --run <features_v1 run> --ctx-ver 2 [--compare <run>] [--n 15000]
Acceptance (audit B1): loss vs full density <= 0.002 at r=0.5.
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from ber import ctx_features as cf
from ber import harness, io, metric, paths
from ber.decision import assign_best_s1, threshold_matches
from ber.tracking import Run

SCEN = ((1.0, False), (0.5, False), (0.5, True), (0.2, True))


def main(a):
    t0 = time.time()
    os.environ.setdefault("POLARS_MAX_THREADS", str(a.threads))
    ctx = harness.EvalContext.load("mini")
    ids = np.sort(np.array(list(ctx.ids)))
    samp = sorted(np.random.default_rng(7).choice(ids, a.n, replace=False).tolist())
    base = pl.read_parquet(paths.DATA_DIR / "cands" / a.cache / "mini.parquet").filter(pl.col("s1_id").is_in(samp))
    truth = {k: ctx.truth[k] for k in samp}
    cty = {k: ctx.country[k] for k in samp}
    runs = [(a.run, a.ctx_ver)] + ([(a.compare, a.compare_ctx_ver)] if a.compare else [])
    models = {}
    for r, v in runs:
        art = paths.ART_DIR / r
        models[r] = (lgb.Booster(model_file=str(art / "model.lgb")), json.loads((art / "features.json").read_text()),
                     json.loads((art / "decision.json").read_text())["threshold"], v)
    s1n = pl.scan_parquet(cf._norm_file("train", 1, cf.norm_of(a.ctx_ver))).select("entity_id", "country", "n_core", "a_full", "a_street",
                                                             "a_house").collect()
    pooln = pl.concat([pl.scan_parquet(cf._norm_file("train", s, cf.norm_of(a.ctx_ver))).select("entity_id", "country", "n_core", "a_full")
                       .collect() for s in (2, 3)])
    gt = pl.from_pandas(io.load_gt_pairs()[["s1_id", "match_id"]])
    matched = set(gt["match_id"].to_list())
    allmini = sorted(ctx.ids)
    Xn = {v: cf.attach_norm(base.select("s1_id", "cand_id", "name_tset"), "train", cf.norm_of(v)) for _, v in runs}
    res = {}
    with Run("density-sim", hypothesis=f"density robustness of {[r for r, _ in runs]}", params=vars(a),
             tags=["robustness"], parent=a.run) as run:
        for r, drop_unm in SCEN:
            rg = np.random.default_rng(42)
            nq = s1n.filter(~pl.col("entity_id").is_in(allmini))
            keep_nq = nq.filter(pl.Series(rg.random(nq.height) < r))
            s1k = pl.concat([s1n.filter(pl.col("entity_id").is_in(allmini)), keep_nq])
            dropped = nq.join(keep_nq.select("entity_id"), on="entity_id", how="anti")["entity_id"]
            dm = gt.filter(pl.col("s1_id").is_in(dropped.to_list()))["match_id"]
            pk = pooln.filter(~pl.col("entity_id").is_in(dm.to_list()))
            if drop_unm:
                um = pk.filter(~pl.col("entity_id").is_in(list(matched)))
                pk = pk.filter(~pl.col("entity_id").is_in(um.filter(pl.Series(rg.random(um.height) >= r))["entity_id"]
                                                         .to_list()))
            key = f"r={r}{'_dropunm' if drop_unm else ''}{'_clean' if a.clean and r < 1 else ''}"
            res[key] = {}
            # clean: also remove candidates that are GT copies of the dropped S1. Without this they stay as
            # ownerless look-alikes ('orphans'), which cannot exist on test (every true copy's owner is present).
            keep_rows = ~pl.col("cand_id").is_in(dm.to_list()) if a.clean else pl.lit(True)
            for rid, (model, feats, t, v) in models.items():
                F = cf.add_features(Xn[v], cf.SplitContext.from_frames(s1k, pk, v), workers=a.threads)
                have = [f for f in feats if f in base.columns]
                X = base.filter(keep_rows).select(["s1_id", "cand_id"] + have).join(
                    F.select(["s1_id", "cand_id"] + [f for f in feats if f not in have]), on=["s1_id", "cand_id"])
                Xp = X.select(["s1_id", "cand_id"] + feats).to_pandas()
                p = Xp[["s1_id", "cand_id"]].assign(prob=model.predict(Xp[feats], num_threads=a.threads))
                rep = metric.report(threshold_matches(assign_best_s1(p), t), truth, cty)
                res[key][rid] = {"f05": round(rep["f05_macro"], 5), "P": round(rep["pair_precision"], 4),
                                 "R": round(rep["pair_recall"], 4), "by_country": rep["f05_by_country"]}
                print(f"{key:14s} {rid[-40:]:40s} F0.5 {rep['f05_macro']:.5f} P {rep['pair_precision']:.4f} "
                      f"R {rep['pair_recall']:.4f} {time.time() - t0:.0f}s", flush=True)
        for rid in models:
            full = res["r=1.0"][rid]["f05"]
            for k in res:
                res[k][rid]["loss_vs_full"] = round(res[k][rid]["f05"] - full, 5)
        run.log(density=res, n_s1=a.n, timing_min=round((time.time() - t0) / 60, 1))
        run.note("; ".join(f"{k}: " + ", ".join(f"{rid[-12:]} {v['f05']:.4f} ({v['loss_vs_full']:+.4f})"
                                              for rid, v in d.items()) for k, d in res.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ctx-ver", type=int, default=1)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--compare-ctx-ver", type=int, default=1)
    ap.add_argument("--cache", default="v1_n1")
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--clean", action="store_true", help="drop GT copies of dropped S1 from the candidates too")
    main(ap.parse_args())
