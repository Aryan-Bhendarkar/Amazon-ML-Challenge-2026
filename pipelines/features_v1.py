"""EXP-B1 features v1: S1-context (P3/P4) + difference (P5) features on a candidate cache.

Reads the shared candidate cache (data/cands/<ver>/<tag>*.parquet: s1_id, cand_id, label, base feats),
adds ber.ctx_features groups (featurized once per cache/tag and cached as ctx1_<tag>.parquet),
retrains the baseline LightGBM recipe, evaluates on the eval tag with the standard harness and
optionally runs the leave-one-country-out (LOCO) France proxy for base-only vs base+groups.

    python pipelines/features_v1.py --cache keys_v0 --groups none            # control (base feats)
    python pipelines/features_v1.py --cache keys_v0 --groups G1,G2,G3,G4,G5 --loco
Threads: --threads (default 2 while another session runs heavy jobs on the box).
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
from ber import harness, io, paths
from ber.decision import assign_best_s1, threshold_matches, tune_threshold
from ber.metric import macro_f05
from ber.tracking import Run

SEED = 42
BASE_RUN = "20260925-1236_aryan_baseline-v0-keys-lgbm"
CAT_BASE = ["house_rel", "legal_rel", "state_rel", "cand_kind"]
CAT_NEW = ["edit_type"]
EXTRA_BASE = ["rbits"]                  # default = the v1_n1 gate run; --extra-base adds cos_name,cos_na
CANDS = paths.DATA_DIR / "cands"


def base_feats() -> list[str]:
    return json.loads((paths.ART_DIR / BASE_RUN / "features.json").read_text())


def load_cache(ver: str, tag: str) -> pl.DataFrame:
    files = sorted((CANDS / ver).glob(f"{tag}*.parquet"))
    files = [f for f in files if not f.name.startswith("ctx")]
    if not files:
        raise FileNotFoundError(f"no cache files data/cands/{ver}/{tag}*.parquet")
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    if "kmask" not in df.columns and "rbits" in df.columns:     # keys_v0: rbits == baseline kmask
        df = df.with_columns((pl.col("rbits") & 15).cast(pl.Int8).alias("kmask"))
    if "role" in df.columns:
        df = df.with_columns((pl.col("role") == "es").alias("is_es"))
    elif "fold" in df.columns:
        df = df.with_columns((pl.col("fold") == 1).alias("is_es"))
    else:
        df = df.with_columns(pl.lit(False).alias("is_es"))
    return df


CTX_VER = 1                              # ber.ctx_features version (set from --ctx-ver); cache = ctx<v>_<tag>


def ctx_path(ver: str, tag: str):
    return CANDS / ver / f"ctx{CTX_VER}_{tag}.parquet"


def featurize_tag(ver: str, tag: str, df: pl.DataFrame, sctx: cf.SplitContext, n_chunks: int, workers: int,
                  log=print, split: str = "train") -> pl.DataFrame:
    """ctx features for every (s1_id, cand_id) of a tag, in S1-hash chunks (complete S1 groups)."""
    p = ctx_path(ver, tag)
    if p.exists():
        return pl.read_parquet(p)
    t0 = time.time()
    keys = df.select("s1_id", "cand_id", "name_tset").with_columns(
        (pl.col("s1_id").hash(seed=11) % n_chunks).alias("_ch"))
    parts = []
    for ch in range(n_chunks):
        P = keys.filter(pl.col("_ch") == ch).drop("_ch")
        X = cf.add_features(cf.attach_norm(P, split, cf.norm_of(CTX_VER)), sctx, workers=workers)
        new = cf.new_feature_cols(X, ["name_tset"])
        parts.append(X.select(["s1_id", "cand_id"] + new))
        log(f"  [{tag}] chunk {ch + 1}/{n_chunks} {P.height:,} pairs {time.time() - t0:.0f}s")
        del X, P
        gc.collect()
    out = pl.concat(parts)
    tmp = p.with_suffix(".tmp")
    out.write_parquet(tmp, compression="zstd")
    os.replace(tmp, p)
    return out


def to_pandas(df: pl.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.select(cols).to_pandas()


# pure similarity scores (higher = more alike): P(match) must not decrease in them (France robustness guardrail)
MONO_UP = ["name_tset", "name_tsort", "name_ratio", "name_partial", "comp_jw", "comp_partial", "alias_tset",
           "addr_tset", "addr_ratio", "street_tset"]
MONOTONE = False                           # set from --monotone


def train_lgb(tr: pd.DataFrame, feats: list[str], cats: list[str], threads: int, lr=0.05, rounds=3000):
    es = tr["is_es"].to_numpy()
    dtr = lgb.Dataset(tr.loc[~es, feats], tr.loc[~es, "label"].astype(int), categorical_feature=cats,
                      free_raw_data=True)
    des = lgb.Dataset(tr.loc[es, feats], tr.loc[es, "label"].astype(int), reference=dtr)
    params = dict(objective="binary", learning_rate=lr, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=SEED, num_threads=threads, deterministic=True, force_row_wise=True)
    if MONOTONE:
        params.update(monotone_constraints=[1 if f in MONO_UP else 0 for f in feats],
                      monotone_constraints_method="advanced")
    return lgb.train(params, dtr, rounds, valid_sets=[des],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])


def f05_at(pred: pd.DataFrame, truth: dict, t: float | None = None) -> tuple[float, float]:
    a = assign_best_s1(pred)
    if t is None:
        t, f, _ = tune_threshold(a, truth)
        return f, t
    return macro_f05(threshold_matches(a, t), truth), t


def loco(tr: pd.DataFrame, ev: pd.DataFrame, feats: list[str], cats: list[str], truth: dict, country: dict,
         threads: int, n_s1: int) -> dict:
    """Train on one country (subsample n_s1 S1 incl. its ES slice), evaluate on the other country's eval S1.
    Reports F0.5 at the target-tuned threshold and at the source-tuned threshold (threshold drift)."""
    out = {}
    ctry_tr = tr["s1_id"].map(country)
    ctry_ev = ev["s1_id"].map(country)
    cs = sorted(set(ctry_ev.dropna()))
    rng = np.random.default_rng(SEED)
    for src in cs:
        ids = tr.loc[ctry_tr == src, "s1_id"].unique()
        ids = set(rng.choice(np.sort(ids), size=min(n_s1, len(ids)), replace=False))
        m = train_lgb(tr[tr.s1_id.isin(ids)], feats, cats, threads, lr=0.1, rounds=1500)
        # source threshold: tuned on the source country's eval S1
        res = {}
        for tgt in cs:
            e = ev[ctry_ev == tgt]
            pr = e[["s1_id", "cand_id"]].assign(prob=m.predict(e[feats], num_threads=threads))
            tt = {k: v for k, v in truth.items() if country.get(k) == tgt}
            res[tgt] = (pr, tt)
        f_src, t_src = f05_at(*res[src])
        for tgt in cs:
            if tgt == src:
                continue
            f_tgt, t_tgt = f05_at(*res[tgt])
            f_srct, _ = f05_at(*res[tgt], t=t_src)
            out[f"{src}->{tgt}"] = {"f05_tuned": round(f_tgt, 5), "t_tuned": t_tgt,
                                    "f05_at_src_t": round(f_srct, 5), "t_src": t_src,
                                    "in_country_f05": round(f_src, 5)}
        del m
        gc.collect()
    return out


def main(a):
    global CTX_VER, MONOTONE
    t0 = time.time()
    MONOTONE = a.monotone
    os.environ.setdefault("POLARS_MAX_THREADS", str(a.threads))
    CTX_VER = a.ctx_ver
    groups = [] if a.groups == "none" else a.groups.split(",")
    base = base_feats()
    print("[load cache]")
    tr_pl = load_cache(a.cache, "train")
    ev_pl = load_cache(a.cache, a.eval_tag)
    if a.n_train_s1:
        ids = np.sort(tr_pl["s1_id"].unique().to_numpy())
        keep = np.random.default_rng(SEED).choice(ids, size=min(a.n_train_s1, len(ids)), replace=False)
        tr_pl = tr_pl.filter(pl.col("s1_id").is_in(keep.tolist()))
    base += [c for c in a.extra_base.split(",") if c and c in tr_pl.columns]
    if a.featurize_only:
        sctx = cf.SplitContext.build("train", CTX_VER)
        for tag in ("train", a.eval_tag):
            featurize_tag(a.cache, tag, load_cache(a.cache, tag), sctx, a.chunks, a.threads)
        print(f"featurized in {(time.time() - t0) / 60:.1f} min")
        return
    ctx = harness.EvalContext.load(a.subset)
    ev_pl = ev_pl.filter(pl.col("s1_id").is_in(sorted(ctx.ids)))
    new = []
    if groups:
        print("[ctx features]")
        sctx = cf.SplitContext.build("train", CTX_VER)
        f_tr = featurize_tag(a.cache, "train", load_cache(a.cache, "train"), sctx, a.chunks, a.threads)
        f_ev = featurize_tag(a.cache, a.eval_tag, load_cache(a.cache, a.eval_tag), sctx, max(1, a.chunks // 4),
                             a.threads)
        del sctx
        gc.collect()
        new = [c for c in f_tr.columns if c not in ("s1_id", "cand_id") and _group_of(c) in groups]
        tr_pl = tr_pl.join(f_tr.select(["s1_id", "cand_id"] + new), on=["s1_id", "cand_id"], how="left")
        ev_pl = ev_pl.join(f_ev.select(["s1_id", "cand_id"] + new), on=["s1_id", "cand_id"], how="left")
        del f_tr, f_ev
    feats = base + new
    cats = CAT_BASE + [c for c in CAT_NEW if c in new]
    tr = to_pandas(tr_pl, ["s1_id", "cand_id", "label", "is_es"] + feats)
    ev = to_pandas(ev_pl, ["s1_id", "cand_id"] + feats)
    del tr_pl, ev_pl
    gc.collect()
    name = f"feat-v1-{a.cache}-{'-'.join(groups) or 'base'}" + (f"-ctx{CTX_VER}" if CTX_VER != 1 else "") \
        + ("-mono" if MONOTONE else "")
    with Run(name, hypothesis=a.hypothesis or f"ctx feature groups {groups or 'none'} on {a.cache} cache",
             params={"cache": a.cache, "ctx_ver": CTX_VER, "monotone": MONOTONE, "groups": groups, "subset": a.subset, "n_train_s1": a.n_train_s1,
                     "threads": a.threads, "n_feats": len(feats), "new_feats": new},
             tags=["features"], parent=a.parent) as run:
        run.log(train_pairs=int((~tr.is_es).sum()), es_pairs=int(tr.is_es.sum()), eval_pairs=len(ev))
        out = None
        if not a.loco_only:
            model = train_lgb(tr, feats, cats, a.threads)
            model.save_model(str(run.art_dir / "model.lgb"))
            (run.art_dir / "features.json").write_text(json.dumps(feats))
            imp = dict(sorted(zip(feats, model.feature_importance("gain").round(1).tolist()), key=lambda x: -x[1]))
            run.log(best_iter=model.best_iteration, feature_gain=imp)
            pred = ev[["s1_id", "cand_id"]].assign(prob=model.predict(ev[feats], num_threads=a.threads).astype(np.float32))
            out = harness.log_predictions(run, pred, ctx)
            (run.art_dir / "decision.json").write_text(json.dumps({"threshold": out["val"]["threshold"]}))
            del model
            gc.collect()
        if a.loco or a.loco_only:
            print("[loco]")
            try:                                       # never lose the main result to a LOCO failure
                folds = io.load_folds()
                cmap = dict(zip(folds.s1_id, folds.country))   # train S1 are not in ctx.country
                del folds
                run.log(loco=loco(tr, ev, feats, cats, ctx.truth, cmap, a.threads, a.loco_n))
            except Exception as e:  # noqa: BLE001
                run.log(loco_error=repr(e))
                print("[loco] FAILED", repr(e))
        run.log(timing_min=round((time.time() - t0) / 60, 1))
        if out is not None:
            run.note(f"{a.subset} F0.5={out['val']['f05_macro']:.4f} "
                     f"({out['val']['f05_by_country']}), groups={groups}. Top gain: {list(imp)[:10]}")
        else:
            run.note(f"LOCO-only reference for groups={groups} (no main model).")


_PREFIX = {"G1": ("s1_name", "pool_name", "n_key_equal"),
           "G2": ("s1_addr", "s1_hs", "pool_addr", "a_key_equal", "hs_key_equal"),
           "G3": ("ex_", "mi_", "sub_jw", "edit_type", "s1_prefix", "c_prefix", "first_tok"),
           "G4": ("house_", "sec_num", "postcode"),
           "G5": ("sib_",),
           "G6": ("twin_",)}


def _group_of(col: str) -> str:
    for g, pre in _PREFIX.items():
        if col.startswith(pre):
            return g
    raise KeyError(col)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="keys_v0")
    ap.add_argument("--eval-tag", default="mini")
    ap.add_argument("--subset", default="mini", choices=["micro", "mini", "fold0"])
    ap.add_argument("--groups", default="G1,G2,G3,G4,G5", help="comma list or 'none'")
    ap.add_argument("--n-train-s1", type=int, default=0, help="subsample training S1 (0 = all in cache)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--loco", action="store_true")
    ap.add_argument("--extra-base", default=",".join(EXTRA_BASE), help="cache columns added to the base features")
    ap.add_argument("--ctx-ver", type=int, default=1, help="ber.ctx_features version (2 = density-invariant G3, 3 = v2 on norm_v1)")
    ap.add_argument("--monotone", action="store_true", help="monotone(+1) constraints on pure similarity scores")
    ap.add_argument("--loco-only", action="store_true", help="skip the main model; LOCO reference only")
    ap.add_argument("--featurize-only", action="store_true", help="only build ctx1_<tag>.parquet caches")
    ap.add_argument("--loco-n", type=int, default=60_000)
    ap.add_argument("--parent", default=BASE_RUN)
    ap.add_argument("--hypothesis", default="")
    main(ap.parse_args())
