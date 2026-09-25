"""Confirm runs on a held-out eval tag with FROZEN models and thresholds (no tuning), then paired bootstrap.

Each run must have artifacts/<run>/{model.lgb, features.json, decision.json(threshold)}. Features missing
from the cache tag are taken from the ctx1_<tag> cache (built here with ber.ctx_features if needed).

    python pipelines/confirm_v1.py --cache v1_n1 --tag fold0x --runs <parent_run>,<candidate_run> --threads 4
"""
import _bootstrap  # noqa: F401

import argparse
import gc
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

import features_v1 as fv
from ber import ctx_features as cf
from ber import harness, metric, paths
from ber.decision import assign_best_s1, threshold_matches, tune_threshold
from ber.tracking import Run

SLICE = 4_000_000


def main(a):
    t0 = time.time()
    os.environ.setdefault("POLARS_MAX_THREADS", str(a.threads))
    runs = a.runs.split(",")
    ctx = harness.EvalContext.load(a.tag)
    df = fv.load_cache(a.cache, a.tag)
    specs = {}
    for r in runs:
        art = paths.ART_DIR / r
        specs[r] = (lgb.Booster(model_file=str(art / "model.lgb")), json.loads((art / "features.json").read_text()),
                    json.loads((art / "decision.json").read_text())["threshold"])
    need = sorted({f for _, feats, _ in specs.values() for f in feats if f not in df.columns})
    if need:
        sctx = None if fv.ctx_path(a.cache, a.tag).exists() else cf.SplitContext.build("train")
        F = fv.featurize_tag(a.cache, a.tag, df, sctx, a.chunks, a.threads)
        df = df.join(F.select(["s1_id", "cand_id"] + need), on=["s1_id", "cand_id"], how="left")
        del F, sctx
        gc.collect()
    print(f"[{a.tag}] {df.height:,} pairs, {len(ctx.ids):,} eval S1; features ready {time.time() - t0:.0f}s")
    with Run(f"confirm-{a.tag}-{a.cache}", hypothesis=f"frozen models/thresholds of {runs} on {a.tag}",
             params=vars(a), tags=["confirm"], parent=runs[-1]) as run:
        res, ent = {}, {}
        for r, (model, feats, t) in specs.items():
            preds = []
            for o in range(0, df.height, SLICE):
                X = df.slice(o, SLICE).select(["s1_id", "cand_id"] + feats).to_pandas()
                preds.append(X[["s1_id", "cand_id"]].assign(prob=model.predict(X[feats], num_threads=a.threads)
                                                             .astype(np.float32)))
            p = pd.concat(preds, ignore_index=True)
            A = assign_best_s1(p[p.prob >= 0.01])
            m = threshold_matches(A, t)
            rep = metric.report(m, ctx.truth, ctx.country)
            t_best, f_best, _ = tune_threshold(A, ctx.truth)      # diagnostic only (never used to decide)
            res[r] = {"f05_frozen_t": rep["f05_macro"], "t": t, "by_country": rep["f05_by_country"],
                      "P": rep["pair_precision"], "R": rep["pair_recall"], "oracle_t": t_best, "f05_at_oracle_t": f_best}
            ent[r] = metric.per_entity_scores(m, ctx.truth)
            ent[r].to_parquet(run.art_dir / f"entity_scores_{r}.parquet")
            print(f"  {r}: F0.5 {rep['f05_macro']:.5f} @t={t} {rep['f05_by_country']} (best t on {a.tag}: {t_best} "
                  f"-> {f_best:.5f})", flush=True)
            del p, A, preds
            gc.collect()
        boot = metric.paired_bootstrap(ent[runs[0]], ent[runs[-1]]) if len(runs) > 1 else None
        run.log(confirm=res, bootstrap=boot, n_eval=len(ctx.ids), timing_min=round((time.time() - t0) / 60, 1))
        ent[runs[-1]].to_parquet(run.art_dir / "val_entity_scores.parquet")
        run.note(f"{a.tag} frozen: " + "; ".join(f"{r}: {v['f05_frozen_t']:.4f}" for r, v in res.items())
                 + (f". Bootstrap last vs first: {boot}" if boot else ""))
        print("bootstrap", boot)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="v1_n1")
    ap.add_argument("--tag", default="fold0x")
    ap.add_argument("--runs", required=True, help="comma list; bootstrap = last vs first")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=16)
    main(ap.parse_args())
