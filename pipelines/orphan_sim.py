"""EXP-A orphan-simulated validation (HANDOFF §5): does test's suspected S1 pruning explain the val->LB gap?

For each scenario (clean, uniform 19%, biased 19%, ...): remove S1 from the WHOLE train S1 set (eval queries and
competitors alike; ber.orphan.removal_mask), keep every S2/S3 record, recompute the S1-side ctx statistics
(G1 name / G2 address counts, G3 token df) on the survivors, featurize the surviving eval S1's cached candidates,
score each frozen model at its frozen threshold, assign records to their best surviving eval S1 and report
macro F0.5 / P / R / per country, the loss split (FP singleton / FP non-singleton / FN), FP sources (orphan vs
owned by another S1 vs unmatched) and the simulated records-per-S1 ratio (test 5.76).

Competition: the model has no record-side competition features; records compete only in assign_best_s1, over
the eval S1 in the scenario (the rest of the train S1 never compete, clean or orphan; measured cost +0.0009, see
docs/ideas_backlog.md EXP-017).

Paired bootstraps on identical surviving S1: orphan vs clean (same model) and each --runs model vs the first.

    python pipelines/orphan_sim.py --runs <baseline run>[,<run2>...] --ctx-vers 3[,3] --tag mini
    python pipelines/orphan_sim.py --runs ... --tag fold0x --scen clean,u19     # confirmation
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

from ber import ctx_features as cf
from ber import harness, io, metric, orphan, paths
from ber.decision import assign_best_s1, threshold_matches, tune_threshold
from ber.tracking import Run

# t19 = test-like: first thin each country to test's pre-pruning S1 density (test S1 count / 0.81, S1 removed WITH
# their records; unmatched records thinned at the same rate), then the u19 orphan pruning (records kept).
THIN = {"t19": True, "tclean": True}
SCEN = {"tw50": (0.5, "twin"), "tw25": (0.25, "twin"), "t19": (0.19, "uniform"), "tclean": (0.0, "uniform"), "clean": (0.0, "uniform"), "u19": (0.19, "uniform"), "b19": (0.19, "biased"), "u30": (0.30, "uniform"),
        "b30": (0.30, "biased")}
TEST_RPS = 5.76


def load_model(run: str):
    art = paths.ART_DIR / run
    return (lgb.Booster(model_file=str(art / "model.lgb")), json.loads((art / "features.json").read_text()),
            json.loads((art / "decision.json").read_text())["threshold"])


def featurize(base: pl.DataFrame, sctx: cf.SplitContext, nv: int, need: list[str], n_chunks: int, workers: int,
              log) -> pl.DataFrame:
    """Recompute ctx features for whole S1 groups (chunked) and join the cached base columns in `need`."""
    keys = base.select("s1_id", "cand_id", "name_tset").with_columns((pl.col("s1_id").hash(seed=11) % n_chunks)
                                                                     .alias("_ch"))
    have = [f for f in need if f in base.columns]
    parts = []
    for ch in range(n_chunks):
        P = keys.filter(pl.col("_ch") == ch).drop("_ch")
        X = cf.add_features(cf.attach_norm(P, "train", nv), sctx, workers=workers)
        add = [f for f in need if f not in have]
        parts.append(X.select(["s1_id", "cand_id"] + add))
        del X, P
        gc.collect()
        log(f"    chunk {ch + 1}/{n_chunks}")
    F = pl.concat(parts)
    return base.select(["s1_id", "cand_id"] + have).join(F, on=["s1_id", "cand_id"], how="inner")


def main(a):
    t0 = time.time()
    os.environ.setdefault("POLARS_MAX_THREADS", str(a.threads))
    log = lambda m: print(f"[{(time.time() - t0) / 60:5.1f}m] {m}", flush=True)  # noqa: E731
    runs = a.runs.split(",")
    vers = [int(v) for v in a.ctx_vers.split(",")] if a.ctx_vers else [3] * len(runs)
    assert len(vers) == len(runs)
    models = {r: load_model(r) for r in runs}
    feats_all = sorted({f for _, fs, _ in models.values() for f in fs})
    feats_raw = cf.expand_derived(feats_all)
    ctx = harness.EvalContext.load(a.subset)
    ids = sorted(ctx.ids)
    if a.n:
        ids = sorted(np.random.default_rng(7).choice(np.array(ids), a.n, replace=False).tolist())
    base_all = pl.read_parquet(paths.DATA_DIR / "cands" / a.cache / f"{a.tag}.parquet")
    base_all = base_all.filter(pl.col("s1_id").is_in(ids))
    keep = [c for c in dict.fromkeys(feats_raw + ["name_tset"]) if c in base_all.columns]
    base_all = base_all.select(["s1_id", "cand_id"] + keep)
    log(f"eval {a.tag}: {len(ids):,} S1, {base_all.height:,} pairs")

    nvs = sorted({cf.norm_of(v) for v in vers})
    s1n = {nv: pl.scan_parquet(cf._norm_file("train", 1, nv)).select("entity_id", "country", "n_core", "a_full", "a_street",
                                                                    "a_house").collect() for nv in nvs}
    pooln = {nv: pl.concat([pl.scan_parquet(cf._norm_file("train", s, nv)).select("entity_id", "country", "n_core", "a_full")
                            .collect() for s in (2, 3)]) for nv in nvs}
    gt = io.load_gt_pairs()[["s1_id", "match_id"]]
    thin_recs, thin_s1 = set(), set()
    if any(THIN.get(sc) for sc in a.scen.split(",")):
        s1_0 = s1n[nvs[0]]
        n_test = pl.scan_parquet(cf._norm_file("test", 1, nvs[0])).group_by("country").len("n_test").collect()
        kc = (s1_0.group_by("country").len("n_tr").join(n_test, on="country", how="left")
              .with_columns((pl.col("n_test") / 0.81 / pl.col("n_tr")).fill_null(1.0).clip(0, 1).alias("keep")))
        keep = dict(zip(kc["country"], kc["keep"]))
        log(f"test-like thinning keep fractions {keep}")
        u = orphan.md5_unit(s1_0["entity_id"].to_list(), "|thin")
        kv = s1_0["country"].replace_strict(keep, default=1.0).to_numpy()
        thin_s1 = set(s1_0.filter(pl.Series(u >= kv))["entity_id"].to_list())
        matched = set(gt.match_id)
        thin_recs = set(gt.match_id[gt.s1_id.isin(thin_s1)])
        um = pooln[nvs[0]].filter(~pl.col("entity_id").is_in(list(matched))).select("entity_id", "country")
        uu = orphan.md5_unit(um["entity_id"].to_list(), "|thin")
        thin_recs |= set(um.filter(pl.Series(uu >= um["country"].replace_strict(keep, default=1.0).to_numpy()))["entity_id"].to_list())
        log(f"thinned {len(thin_s1):,} S1 and {len(thin_recs):,} pool records")
    gt = gt
    owner = pd.Series(gt.s1_id.to_numpy(), index=gt.match_id.to_numpy())
    del gt
    truth_all = {k: ctx.truth[k] for k in ids}
    cty_all = {k: ctx.country[k] for k in ids}
    s1_ref = s1n[nvs[0]]
    out_dir = None
    res = {}
    with Run(f"orphan-sim-{a.tag}", hypothesis=a.hypothesis or f"orphan-sim eval of {runs}", params=vars(a),
             tags=["orphan-sim", "robustness"], parent=runs[0]) as run:
        out_dir = run.art_dir
        clean_scores = {}
        for sc in a.scen.split(","):
            rate, variant = SCEN[sc]
            mask = orphan.removal_mask(s1_ref, rate, variant)
            removed = set(mask.filter(pl.col("removed"))["entity_id"].to_list())
            thin = THIN.get(sc, False)
            gone_s1 = removed | thin_s1 if thin else removed          # S1 absent from the scenario's world
            gone_rec = thin_recs if thin else set()                   # records absent (thinned with their owner)
            removed = removed - thin_s1 if thin else removed          # orphan owners (their records stay)
            surv = [k for k in ids if k not in gone_s1]
            truth = {k: truth_all[k] for k in surv}
            cty = {k: cty_all[k] for k in surv}
            base = base_all.filter(pl.col("s1_id").is_in(surv))
            if gone_rec:
                base = base.filter(~pl.col("cand_id").is_in(list(gone_rec)))
            pool_sc = {nv: (P.filter(~pl.col("entity_id").is_in(list(gone_rec))) if gone_rec else P) for nv, P in pooln.items()}
            pool_by_c = pool_sc[nvs[0]].group_by("country").len("n_pool")
            s1c = s1_ref.filter(~pl.col("entity_id").is_in(list(gone_s1))).group_by("country").len("n_s1")
            rps = s1c.join(pool_by_c, on="country").with_columns((pl.col("n_pool") / pl.col("n_s1")).alias("rps"))
            info = {"rate": rate, "variant": variant, "removed_frac_all": round(len(gone_s1) / s1_ref.height, 4),
                    "orphan_owners": len(removed), "thinned_records": len(gone_rec),
                    "eval_survivors": len(surv), "records_per_s1": {r["country"]: round(r["rps"], 3) for r in rps.to_dicts()},
                    "records_per_s1_all": round(pool_by_c["n_pool"].sum() / s1c["n_s1"].sum(), 3)}
            log(f"[{sc}] removed {len(gone_s1):,} S1 ({info['removed_frac_all']:.3f}), eval survivors {len(surv):,}, "
                f"rec/S1 {info['records_per_s1_all']} (test {TEST_RPS})")
            res[sc] = {"info": info}
            sctx = {v: cf.SplitContext.from_frames(s1n[cf.norm_of(v)].filter(~pl.col("entity_id").is_in(list(gone_s1))),
                                                   pool_sc[cf.norm_of(v)], v) for v in set(vers)}
            Xcache = {}
            for r, v in zip(runs, vers):
                model, feats, t = models[r]
                if v not in Xcache:
                    log(f"  featurize ctx v{v}")
                    Xcache[v] = cf.add_derived(featurize(base, sctx[v], cf.norm_of(v), feats_raw, a.chunks, a.threads, log),
                                               feats_all).to_pandas()
                X = Xcache[v]
                pred = X[["s1_id", "cand_id"]].assign(prob=model.predict(X[feats], num_threads=a.threads).astype(np.float32))
                if a.save_preds:
                    pred.to_parquet(out_dir / f"pred_{sc}_{runs.index(r)}.parquet")
                asg = assign_best_s1(pred)
                m = threshold_matches(asg, t)
                rep = metric.report(m, truth, cty)
                t_opt, f_opt, _ = tune_threshold(asg, truth)
                pe = metric.per_entity_scores(m, truth)
                pe.to_parquet(out_dir / f"ent_{sc}_{runs.index(r)}.parquet")
                d = {"f05": round(rep["f05_macro"], 5), "P": round(rep["pair_precision"], 5),
                     "R": round(rep["pair_recall"], 5), "by_country": {k: round(x, 5) for k, x in rep["f05_by_country"].items()},
                     "singleton_acc": rep["singleton_acc"], "t": t, "t_opt": t_opt, "f05_at_t_opt": round(f_opt, 5),
                     **{k: round(x, 5) for k, x in orphan.loss_decomposition(m, truth).items()},
                     **orphan.fp_sources(m, truth, owner, removed)}
                if sc == "clean":
                    clean_scores[r] = pe
                elif r in clean_scores:
                    d["vs_clean_same_S1"] = metric.paired_bootstrap(clean_scores[r], pe)
                if r != runs[0]:
                    d["vs_base"] = metric.paired_bootstrap(pd.read_parquet(out_dir / f"ent_{sc}_0.parquet"), pe)
                res[sc][r] = d
                log(f"  {sc} {r[-45:]}: F0.5 {d['f05']:.5f} P {d['P']:.4f} R {d['R']:.4f} {d['by_country']} "
                    f"FP orphan {d['fp_orphan']}/{d['fp_total']} loss fpS {d['loss_fp_singleton']:.4f} "
                    f"fpNS {d['loss_fp_nonsingleton']:.4f} fn {d['loss_fn']:.4f}"
                    + (f" | vs clean {d['vs_clean_same_S1']['delta']:+.5f} {d['vs_clean_same_S1']['ci95']}"
                       if "vs_clean_same_S1" in d else "")
                    + (f" | vs base {d['vs_base']['delta']:+.5f} {d['vs_base']['ci95']}" if "vs_base" in d else ""))
                del pred, asg
            del Xcache, sctx, base, pool_sc
            gc.collect()
            run.log(orphan=res)
        run.log(orphan=res, timing_min=round((time.time() - t0) / 60, 1))
        run.note("; ".join(f"{sc}: " + ", ".join(f"{r[-14:]} {d['f05']:.5f}" +
                                                  (f" (vs clean {d['vs_clean_same_S1']['delta']:+.5f})" if "vs_clean_same_S1" in d else "")
                                                  for r, d in v.items() if r != "info") for sc, v in res.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list of features_v1 runs; the first is the baseline")
    ap.add_argument("--ctx-vers", default="", help="comma list of ctx versions per run (default 3)")
    ap.add_argument("--cache", default="v1_n2")
    ap.add_argument("--tag", default="mini", help="cache tag (mini | fold0x)")
    ap.add_argument("--subset", default=None, help="eval subset (default = tag)")
    ap.add_argument("--scen", default="clean,u19,b19")
    ap.add_argument("--n", type=int, default=0, help="subsample eval S1 (0 = all)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--save-preds", action="store_true")
    ap.add_argument("--hypothesis", default="")
    args = ap.parse_args()
    args.subset = args.subset or args.tag
    main(args)
