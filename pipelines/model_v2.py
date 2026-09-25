"""EXP model_v2, phase B on the cands_v2 union:
  1. pruner   : LightGBM on cheap blocking features (retriever bits, tf-idf cosines, their ranks in the S1)
                trained on role 'train' (ES on 'es') -> keep top-N per S1 by pruner prob (replaces the cos_na cap)
  2. features : baseline_v0 pair features + blocking features + ctx_features G1..G5 on the kept pairs (cached)
  3. matcher  : baseline LightGBM recipe -> eval subset via the standard harness (+ post-rules for reference)

    python pipelines/model_v2.py --keep 40 [--name ...]
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

import baseline_v0 as bv0
from ber import blocking as B
from ber import ctx_features as cf
from ber import harness, metric, paths, postrules
from ber.decision import assign_best_s1
from ber.tracking import Run

SEED = 42
NORM_V = 1
CANDS = paths.DATA_DIR / "cands" / "v2"
DROP_RET = {"rev", "tf_name"}     # tf_name: 6 of 298k mini positives found only by it; ~40% of TF-IDF time
RETS = [n for n in B.RBITS if n not in DROP_RET]
DROP_MASK = sum(B.RBITS[n] for n in DROP_RET)
PRUNE_FEATS = ["cos_name", "cos_addr", "cos_na", "r_name", "r_addr", "r_na", "n_union", "top_na", "gap_na"] + \
              [f"b_{n}" for n in RETS]
CATS = ["house_rel", "legal_rel", "state_rel", "cand_kind", "edit_type"]
LGB = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
           verbose=-1, seed=SEED, num_threads=0, deterministic=True, force_row_wise=True)

bv0.NORM_V = NORM_V
cf.NORM_V = NORM_V
postrules_norm_v = NORM_V


def load_union() -> pl.DataFrame:
    U = pl.concat([pl.read_parquet(p) for p in sorted(CANDS.glob("blk_*.parquet"))])
    return add_prune_feats(U)


def add_prune_feats(U: pl.DataFrame) -> pl.DataFrame:
    """U: s1_id, rbits, cos_name, cos_addr, cos_na (complete per S1). Drops pairs only found by DROP_RET."""
    U = U.with_columns((pl.col("rbits") & ~pl.lit(DROP_MASK, dtype=pl.Int32)).alias("rbits")).filter(pl.col("rbits") != 0)
    U = U.with_columns([((pl.col("rbits") & B.RBITS[n]) > 0).cast(pl.Int8).alias(f"b_{n}") for n in RETS])
    g = "s1_id"
    return U.with_columns(
        pl.col("cos_name").rank("min", descending=True).over(g).cast(pl.Float32).alias("r_name"),
        pl.col("cos_addr").rank("min", descending=True).over(g).cast(pl.Float32).alias("r_addr"),
        pl.col("cos_na").rank("min", descending=True).over(g).cast(pl.Float32).alias("r_na"),
        pl.len().over(g).cast(pl.Float32).alias("n_union"),
        pl.col("cos_na").max().over(g).alias("top_na"),
    ).with_columns((pl.col("top_na") - pl.col("cos_na")).alias("gap_na"))


def train_lgb(X: pd.DataFrame, y: np.ndarray, es: np.ndarray, feats, cats, params=None, rounds=3000):
    p = {**LGB, **(params or {})}
    dtr = lgb.Dataset(X.loc[~es, feats], y[~es], categorical_feature=cats, free_raw_data=True)
    des = lgb.Dataset(X.loc[es, feats], y[es], reference=dtr)
    return lgb.train(p, dtr, rounds, valid_sets=[des],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])


def prune(U: pl.DataFrame, keep: int, log) -> pl.DataFrame:
    p = CANDS / f"pruned_k{keep}_nt.parquet"
    if p.exists():
        return pl.read_parquet(p)
    tr = U.filter(pl.col("role") != "eval").select(PRUNE_FEATS + ["label", "role"]).to_pandas()
    m = train_lgb(tr, tr.label.to_numpy().astype(int), (tr.role == "es").to_numpy(), PRUNE_FEATS, [],
                  dict(num_leaves=63, learning_rate=0.1), 1000)
    del tr
    m.save_model(str(CANDS / "pruner_nt.lgb"))
    pp = np.concatenate([m.predict(U[o:o + 2_000_000].select(PRUNE_FEATS).to_pandas(), num_threads=0)
                         for o in range(0, U.height, 2_000_000)]).astype(np.float32)
    U = U.with_columns(pl.Series("p_prune", pp))
    U = U.with_columns(pl.col("p_prune").rank("ordinal", descending=True).over("s1_id").alias("r_prune"))
    for role in ("train", "eval"):
        R = U.filter(pl.col("role") == role)
        pos = R["label"].sum()
        log(f"[pruner] {role}: union recall 1.0 (pos {pos:,}); " + ", ".join(
            f"top{k}: {R.filter((pl.col('r_prune') <= k) & pl.col('label')).height / pos:.4f}"
            for k in (10, 20, 30, 40, 60, 80, 120)))
    K = U.filter(pl.col("r_prune") <= keep)
    K.write_parquet(p)
    return K


def featurize(K: pl.DataFrame, log) -> pl.DataFrame:
    """baseline_v0 features (per country, chunked by S1) + ctx G1..G5. Cached."""
    p = CANDS / f"feats_nt_{K.height}.parquet"
    if p.exists():
        return pl.read_parquet(p)
    t0 = time.time()
    out = []
    keepcols = ["s1_id", "cand_id", "role", "label", "p_prune", "r_prune"] + PRUNE_FEATS
    for ctry in K["country"].unique().sort().to_list():
        Kc = K.filter(pl.col("country") == ctry)
        ptbl = B.pool_table("train", ctry, NORM_V)
        s1 = (pl.scan_parquet(B.norm_file("train", 1, NORM_V)).select(bv0.COLS)
                .join(Kc.select(pl.col("s1_id").unique().alias("entity_id")).lazy(), on="entity_id").collect()
                .with_row_index("idx"))
        imap = dict(zip(s1["entity_id"].to_list(), range(s1.height)))
        ids = Kc["s1_id"].unique().sort().to_list()
        CH = 20_000
        for o in range(0, len(ids), CH):
            part = Kc.filter(pl.col("s1_id").is_in(ids[o:o + CH]))
            pairs = pl.DataFrame({"i": np.array([imap[s] for s in part["s1_id"].to_list()], dtype=np.uint32),
                                  "j": part["j"].to_numpy().astype(np.uint32),
                                  "kmask": part["rbits"].to_numpy().astype(np.int32)})
            pool_part = bv0.fetch_pool(ptbl, pairs["j"].to_numpy())
            f = bv0.featurize(pairs, s1, pool_part)
            f = pl.from_pandas(f).join(part.select(keepcols), on=["s1_id", "cand_id"], how="left")
            out.append(f)
            log(f"  [feat {ctry}] {o + len(ids[o:o + CH]):,}/{len(ids):,} S1 {time.time() - t0:.0f}s")
            del pairs, pool_part, f
            gc.collect()
        del ptbl, s1
    F = pl.concat(out)
    del out
    gc.collect()
    log("[ctx features]")
    sctx = cf.SplitContext.build("train")
    parts = []
    F = F.with_columns((pl.col("s1_id").hash(seed=11) % 8).alias("_ch"))
    for ch in range(8):
        Pch = F.filter(pl.col("_ch") == ch).drop("_ch")
        X = cf.add_features(cf.attach_norm(Pch.select("s1_id", "cand_id", "name_tset"), "train"), sctx)
        new = cf.new_feature_cols(X, ["name_tset"])
        parts.append(Pch.join(X.select(["s1_id", "cand_id"] + new), on=["s1_id", "cand_id"], how="left"))
        log(f"  [ctx] chunk {ch + 1}/8 {time.time() - t0:.0f}s")
        del X, Pch
        gc.collect()
    F = pl.concat(parts)
    F.write_parquet(p)
    return F


def main(a):
    t0 = time.time()
    log = lambda s: print(s, flush=True)  # noqa: E731
    U = load_union()
    log(f"[union] {U.height:,} pairs, {U['s1_id'].n_unique():,} S1 ({time.time() - t0:.0f}s)")
    K = prune(U, a.keep, log)
    del U
    gc.collect()
    F = featurize(K, log)
    del K
    feats = [c for c in F.columns if c not in ("s1_id", "cand_id", "role", "label", "country", "i", "j", "rbits")]
    if a.drop:
        feats = [c for c in feats if c not in a.drop.split(",")]
    cats = [c for c in CATS if c in feats]
    ctx = harness.EvalContext.load(a.subset)
    tr = F.filter(pl.col("role") != "eval").select(["s1_id", "cand_id", "role", "label"] + feats).to_pandas()
    ev = F.filter(pl.col("role") == "eval").select(["s1_id", "cand_id"] + feats).to_pandas()
    del F
    gc.collect()
    with Run(a.name, hypothesis=a.hypothesis, parent=a.parent, tags=["blocking", "features", "model"],
             params={"keep": a.keep, "n_feats": len(feats), "norm_v": NORM_V, "lgb": LGB, "drop": a.drop}) as run:
        log(f"[data] train {len(tr):,} pairs ({tr.s1_id.nunique():,} S1), eval {len(ev):,} pairs")
        model = train_lgb(tr, tr.label.to_numpy().astype(int), (tr.role == "es").to_numpy(), feats, cats)
        model.save_model(str(run.art_dir / "model.lgb"))
        (run.art_dir / "features.json").write_text(json.dumps(feats))
        imp = dict(sorted(zip(feats, model.feature_importance("gain").round(1).tolist()), key=lambda x: -x[1]))
        run.log(train_pairs=len(tr), best_iter=model.best_iteration, feature_gain=imp)
        pred = ev[["s1_id", "cand_id"]].assign(prob=model.predict(ev[feats], num_threads=0).astype(np.float32))
        harness.log_blocking(run, ev[["s1_id", "cand_id"]], ctx)
        out = harness.log_predictions(run, pred, ctx)
        t = out["val"]["threshold"]
        (run.art_dir / "decision.json").write_text(json.dumps({"threshold": t}))
        # reference: team post-rules on top
        fl = postrules.context_flags(assign_best_s1(pred), "train", NORM_V)
        best = max(((metric.macro_f05(postrules.to_matches(fl, tt), ctx.truth), tt)
                    for tt in np.round(np.arange(0.4, 0.91, 0.025), 3)))
        run.log(postrules_f05=best[0], postrules_t=best[1])
        log(f"[postrules] F0.5 {best[0]:.4f} @t={best[1]}")
        pa_ = paths.ART_DIR / a.parent / "val_entity_scores.parquet" if a.parent else None
        if pa_ is not None and pa_.exists():
            bs = metric.paired_bootstrap(pd.read_parquet(pa_), pd.read_parquet(run.art_dir / "val_entity_scores.parquet"))
            run.log(vs_parent=bs)
            log(f"[bootstrap vs parent] {bs}")
        run.log(timing_min=round((time.time() - t0) / 60, 1))
        v = out["val"]
        run.note(f"{a.subset} F0.5={v['f05_macro']:.4f} (post-rules {best[0]:.4f}) by_country={v['f05_by_country']} "
                 f"P={v['pair_precision']:.4f} R={v['pair_recall']:.4f} t={t}. Top feats: {list(imp)[:10]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=40)
    ap.add_argument("--subset", default="mini")
    ap.add_argument("--name", default="model-v2-pruned-union-ctx-notfname")
    ap.add_argument("--hypothesis", default="blocking v1 union + learned pruner + base+ctx features -> beats 0.9107")
    ap.add_argument("--parent", default="20260925-1531_aryan-bhendarkar_postrules-v0")
    ap.add_argument("--drop", default="")
    main(ap.parse_args())
