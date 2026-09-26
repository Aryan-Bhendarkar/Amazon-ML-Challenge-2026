"""EXP-B (claude/HANDOFF.md): retrain on ORPHAN-PRUNED training views so the model learns the test shift.

Test looks like train with ~19% of S1 removed and their S2/S3 records kept. Each train-tag S1 goes to exactly one
view by md5(s1_id + "|viewsel") mod 6:
    0, 1 -> clean (reuse the cached ctx<v>_train features)      2 -> uniform 19%   3 -> biased 19%
    4 -> uniform 30%                                              5 -> biased 30%
For a pruned view, the removal mask (ber.orphan.removal_mask, salt f"|view{k}", never the eval salt) is applied to
ALL train S1. View S1 removed by their own mask are dropped. The S1-side split statistics (ctx G1/G2/G3) are
recomputed on the surviving S1 with the FULL pool (all S2/S3 kept). Records keep their labels: a record is positive
only for its own (surviving) S1, so orphaned records are already negatives (asserted).
Same features, LightGBM params and ES as the parent features_v1 run. The threshold is tuned on CLEAN mini (cached
ctx features) and written in the features_v1 format, so pipelines/orphan_sim.py can score the run directly.

    python pipelines/orphan_train.py --cache v1_n2 --ctx-ver 3 --threads 6 --loco \\
        --parent 20260926-0710_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3
Smoke: --n-train-s1 5000 --threads 2
"""
import _bootstrap  # noqa: F401

import argparse
import gc
import hashlib
import json
import os
import time

import numpy as np
import polars as pl

import features_v1 as fv
from ber import ctx_features as cf
from ber import harness, io, paths
from ber.orphan import removal_mask
from ber.tracking import Run

VIEW_SETS = {
    "handoff": None,                                            # = VIEWS below (HANDOFF mixture)
    # DIAG-TWIN (26 Sep): the test shift is twin-targeted (same-name S1 at another house missing), not uniform pruning
    "twin": {0: ("clean", 0.0), 1: ("clean", 0.0), 2: ("uniform", 0.19), 3: ("twin", 0.5), 4: ("twin", 0.5),
             5: ("biased", 0.19)},
}
VIEWS = {0: ("clean", 0.0), 1: ("clean", 0.0), 2: ("uniform", 0.19), 3: ("biased", 0.19),
         4: ("uniform", 0.30), 5: ("biased", 0.30)}


def view_of(ids) -> np.ndarray:
    return np.fromiter((int(hashlib.md5((s + "|viewsel").encode()).hexdigest(), 16) % 6 for s in ids),
                       dtype=np.int8, count=len(ids))


def main(a):
    global VIEWS
    t0 = time.time()
    if VIEW_SETS.get(a.views):
        VIEWS = VIEW_SETS[a.views]
    os.environ.setdefault("POLARS_MAX_THREADS", str(a.threads))
    fv.CTX_VER = a.ctx_ver
    nv = cf.norm_of(a.ctx_ver)
    parent_feats = json.loads((paths.ART_DIR / a.parent / "features.json").read_text())
    tr = fv.load_cache(a.cache, "train")
    if a.n_train_s1:
        ids = np.sort(tr["s1_id"].unique().to_numpy())
        keep = np.random.default_rng(fv.SEED).choice(ids, size=min(a.n_train_s1, len(ids)), replace=False)
        tr = tr.filter(pl.col("s1_id").is_in(keep.tolist()))
    s1_ids = tr["s1_id"].unique().sort()
    vmap = pl.DataFrame({"s1_id": s1_ids, "view": view_of(s1_ids.to_list())})
    tr = tr.join(vmap, on="s1_id")
    ctx_cols = [c for c in cf.expand_derived(parent_feats) if c not in tr.columns]   # derived (q05) from raw
    print(f"[load] {tr.height:,} train pairs, {len(s1_ids):,} S1; {len(ctx_cols)} ctx features; "
          f"views {vmap.group_by('view').len().sort('view').to_dicts()}", flush=True)

    # clean views: cached ctx features
    clean = tr.filter(pl.col("view") <= 1)
    cached = pl.read_parquet(fv.ctx_path(a.cache, "train")).select(["s1_id", "cand_id"] + ctx_cols)
    parts = [clean.join(cached, on=["s1_id", "cand_id"], how="left")]
    del cached
    # pruned views: recompute ctx features on the surviving S1 (+ full pool)
    s1_all = pl.scan_parquet(cf._norm_file("train", 1, nv)).select(
        "entity_id", "country", "n_core", "a_full", "a_street", "a_house").collect()
    pool_all = pl.concat([pl.scan_parquet(cf._norm_file("train", s, nv)).select("country", "n_core", "a_full").collect()
                          for s in (2, 3)])
    gt = io.load_gt_pairs()
    owner = pl.from_pandas(gt[["s1_id", "match_id"]]).rename({"s1_id": "owner", "match_id": "cand_id"})
    stats = {}
    for k in range(2, 6):
        variant, rate = VIEWS[k]
        rem = removal_mask(s1_all, rate, variant, salt=f"|view{k}")
        removed = rem.filter(pl.col("removed"))["entity_id"]
        V = tr.filter(pl.col("view") == k).filter(~pl.col("s1_id").is_in(removed.to_list()))
        # label sanity: positives only for the pair's own S1 (at-most-one-owner) -> orphans are already negatives
        chk = V.filter(pl.col("label")).join(owner, on="cand_id", how="left")
        bad = chk.filter(pl.col("owner") != pl.col("s1_id")).height
        assert bad == 0, f"view {k}: {bad} positive pairs not owned by their S1"
        n_orphan = V.join(owner.filter(pl.col("owner").is_in(removed.to_list())), on="cand_id").height
        sctx = cf.SplitContext.from_frames(s1_all.filter(~pl.col("entity_id").is_in(removed.to_list())), pool_all,
                                           a.ctx_ver)
        feats_k = []
        V = V.with_columns((pl.col("s1_id").hash(seed=11) % a.chunks).alias("_ch"))
        for ch in range(a.chunks):
            P = V.filter(pl.col("_ch") == ch).select("s1_id", "cand_id", "name_tset")
            if P.height == 0:
                continue
            X = cf.add_features(cf.attach_norm(P, "train", nv), sctx, workers=a.threads)
            feats_k.append(X.select(["s1_id", "cand_id"] + ctx_cols))
            del X
            gc.collect()
        Fk = pl.concat(feats_k)
        parts.append(V.drop("_ch").join(Fk, on=["s1_id", "cand_id"], how="left"))
        stats[f"view{k}_{variant}{int(rate * 100)}"] = {"s1_removed_total": int(rem["removed"].sum()),
                                                      "view_s1_kept": int(V["s1_id"].n_unique()),
                                                      "view_pairs": V.height, "orphan_cand_pairs": n_orphan}
        print(f"  view {k} {variant} {rate}: {stats[list(stats)[-1]]} {time.time() - t0:.0f}s", flush=True)
        del sctx, Fk, V
        gc.collect()
    del s1_all, pool_all
    trv = cf.add_derived(pl.concat(parts, how="diagonal_relaxed"), parent_feats)
    del parts, tr
    gc.collect()
    feats = parent_feats
    cats = fv.CAT_BASE + [c for c in fv.CAT_NEW if c in feats]
    trp = trv.select(["s1_id", "cand_id", "label", "is_es", "view"] + feats).to_pandas()
    del trv
    gc.collect()
    ctx = harness.EvalContext.load(a.subset)
    ev = fv.load_cache(a.cache, a.eval_tag).filter(pl.col("s1_id").is_in(sorted(ctx.ids)))
    ev = ev.join(pl.read_parquet(fv.ctx_path(a.cache, a.eval_tag)).select(["s1_id", "cand_id"] + ctx_cols),
                 on=["s1_id", "cand_id"], how="left")
    ev = cf.add_derived(ev, feats).select(["s1_id", "cand_id"] + feats).to_pandas()
    with Run(f"orphan-train-{a.cache}-ctx{a.ctx_ver}-{a.views}", hypothesis=a.hypothesis or
             "EXP-B: training on orphan-pruned views (0/19/30%, uniform+biased) makes the model robust to test's S1 "
             "removal: orphan-sim mini +0.003, clean >= -0.001, LOCO >= -0.0005",
             params={**{k: v for k, v in vars(a).items()}, "views": {k: list(v) for k, v in VIEWS.items()},
                     "ctx_ver": a.ctx_ver, "n_feats": len(feats)},
             tags=["features", "orphan"], parent=a.parent) as run:
        run.log(view_stats=stats, train_pairs=int((~trp.is_es).sum()), es_pairs=int(trp.is_es.sum()),
                pairs_by_view={int(k): int(v) for k, v in trp.view.value_counts().sort_index().items()})
        model = fv.train_lgb(trp, feats, cats, a.threads)
        model.save_model(str(run.art_dir / "model.lgb"))
        (run.art_dir / "features.json").write_text(json.dumps(feats))
        imp = dict(sorted(zip(feats, model.feature_importance("gain").round(1).tolist()), key=lambda x: -x[1]))
        run.log(best_iter=model.best_iteration, feature_gain=imp)
        pred = ev[["s1_id", "cand_id"]].assign(prob=model.predict(ev[feats], num_threads=a.threads).astype(np.float32))
        out = harness.log_predictions(run, pred, ctx)
        (run.art_dir / "decision.json").write_text(json.dumps({"threshold": out["val"]["threshold"]}))
        del model
        gc.collect()
        if a.loco:
            try:
                folds = io.load_folds()
                cmap = dict(zip(folds.s1_id, folds.country))
                del folds
                run.log(loco=fv.loco(trp, ev, feats, cats, ctx.truth, cmap, a.threads, a.loco_n))
            except Exception as e:  # noqa: BLE001
                run.log(loco_error=repr(e))
                print("[loco] FAILED", repr(e))
        run.log(timing_min=round((time.time() - t0) / 60, 1))
        run.note(f"clean {ctx.subset} F0.5={out['val']['f05_macro']:.4f} ({out['val']['f05_by_country']}); "
                 f"views {stats}. Orphan-sim eval: pipelines/orphan_sim.py (amlc-49).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="v1_n2")
    ap.add_argument("--ctx-ver", type=int, default=3)
    ap.add_argument("--eval-tag", default="mini")
    ap.add_argument("--subset", default="mini", choices=["micro", "mini", "fold0"])
    ap.add_argument("--parent", default="20260926-0710_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3")
    ap.add_argument("--n-train-s1", type=int, default=0, help="subsample train S1 (smoke test)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--loco", action="store_true")
    ap.add_argument("--loco-n", type=int, default=60_000)
    ap.add_argument("--views", default="handoff", choices=list(VIEW_SETS))
    ap.add_argument("--hypothesis", default="")
    main(ap.parse_args())
