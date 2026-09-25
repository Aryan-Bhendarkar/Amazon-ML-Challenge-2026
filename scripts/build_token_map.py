"""Learn the native-script token map from TRAIN GT pairs of folds >= 1 (never fold 0) and save it to
data/features/token_map_v1.json. Also reports a holdout check (learn on folds 2-4, test on fold 1):
token accuracy on covered tokens, coverage, and the name token_set gain on fold-1 native pairs.

Usage: python scripts/build_token_map.py [--min-support 3] [--min-precision 0.5]
"""
import _bootstrap  # noqa: F401

import argparse
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from ber import paths, tokenmap
from ber.normalize import normalize_name

OUT = paths.FEATURE_DIR / "token_map_v1.json"
C = ["entity_id", "n_full", "n_script"]


def native_pairs(min_fold: int) -> pl.DataFrame:
    fo = pl.scan_parquet(paths.folds_path()).select("s1_id", "fold").filter(pl.col("fold") >= min_fold)
    gt = pl.scan_parquet(paths.gt_pairs_path()).join(fo, on="s1_id")
    s1 = pl.scan_parquet(paths.FEATURE_DIR / "norm_v0_train_s1.parquet").select(C)
    pool = pl.concat([pl.scan_parquet(paths.FEATURE_DIR / f"norm_v0_train_s{s}.parquet").select(C) for s in (2, 3)])
    pool = pool.filter(pl.col("n_script") != "latin")
    return (gt.join(pool, left_on="match_id", right_on="entity_id")
              .join(s1.filter(pl.col("n_script") == "latin"), left_on="s1_id", right_on="entity_id", suffix="_s1")
              .collect())


def holdout(p: pl.DataFrame, a) -> dict:
    tr, te = p.filter(pl.col("fold") >= 2), p.filter(pl.col("fold") == 1)
    tmap, _ = tokenmap.learn(tokenmap.aligned_pairs(tr["n_full"], tr["n_full_s1"]), a.min_support, a.min_precision)
    al = tokenmap.aligned_pairs(te["n_full"], te["n_full_s1"])
    mapped = al.with_columns(pl.col("src").replace(tmap).alias("pred"))
    cov = mapped.filter(pl.col("src").is_in(list(tmap.keys())))
    before = process.cpdist(te["n_full"].to_list(), te["n_full_s1"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    after_names = [" ".join(tokenmap.apply(x.split(), tmap)) for x in te["n_full"].to_list()]
    after = process.cpdist(after_names, te["n_full_s1"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    return {"fold1_token_alignments": al.height,
            "acc_identity_before": float((al["src"] == al["dst"]).mean()),
            "acc_after_map": float((mapped["pred"] == mapped["dst"]).mean()),
            "coverage_of_changed_tokens": float(cov.height / max(1, (al["src"] != al["dst"]).sum())),
            "acc_on_mapped": float((cov["pred"] == cov["dst"]).mean()) if cov.height else None,
            "name_tset_mean_before": float(np.mean(before)), "name_tset_mean_after": float(np.mean(after)),
            "name_tset_ge90_before": float(np.mean(before >= 90)), "name_tset_ge90_after": float(np.mean(after >= 90))}


def main(a):
    t0 = time.time()
    p = native_pairs(min_fold=1)
    print(f"native-script train pairs (folds 1-4): {p.height:,}", flush=True)
    ho = holdout(p, a)
    print("[holdout learn 2-4 / test fold 1]", {k: round(v, 4) if isinstance(v, float) else v for k, v in ho.items()})
    tmap, stats = tokenmap.learn(tokenmap.aligned_pairs(p["n_full"], p["n_full_s1"]), a.min_support, a.min_precision)
    tokenmap.save(tmap, OUT, meta={**stats, "holdout": ho, "folds": "1-4", "norm_input": "v0"})
    print("[final map]", stats, f"-> {OUT} ({time.time() - t0:.0f}s)")
    ex = ["gret phaumdesn", "siv prodyusr praibhet limitet", "rayl kncltnci llp"]
    for x in ex:
        print(f"  {x!r} -> {' '.join(tokenmap.apply(x.split(), tmap))!r}")
    # sanity: the normalizer applies it only to non-Latin names
    print("  normalize_name latin untouched:", normalize_name("Siv Prodyusr", token_map=tmap).core)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-support", type=int, default=3)
    ap.add_argument("--min-precision", type=float, default=0.5)
    main(ap.parse_args())
