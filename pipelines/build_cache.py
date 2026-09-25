"""Candidate + feature cache (Phase 0.3) so model/decision experiments skip blocking.

cand_ver=keys_v0: baseline_v0 blocking + its 28 features, for
  train.parquet  : baseline training sample (seed 42: 150k fold>=2 'train' + 30k fold-1 'es'), label from GT
  <subset>.parquet: eval subset S1 (role 'eval'), candidates from the FULL train pool of the country
Columns: s1_id, cand_id, label, role, + baseline feature columns (kmask included).
Also writes data/cands/train_ids.parquet (s1_id, role).

    python pipelines/build_cache.py --cand-ver keys_v0 [--subset mini] [--n-train 150000]
"""
import _bootstrap  # noqa: F401

import argparse
import time

import numpy as np
import pandas as pd

import baseline_v0 as bv0
from ber import harness, io, paths

SEED = 42


def train_ids(n_train: int) -> pd.DataFrame:
    folds = io.load_folds()
    rng = np.random.default_rng(SEED)
    tr_pool = np.sort(folds[folds.fold >= 2].s1_id.to_numpy())
    es_pool = np.sort(folds[folds.fold == 1].s1_id.to_numpy())
    tr = rng.choice(tr_pool, size=min(n_train, len(tr_pool)), replace=False)
    es = rng.choice(es_pool, size=min(n_train // 5, len(es_pool)), replace=False)
    return pd.concat([pd.DataFrame({"s1_id": tr, "role": "train"}), pd.DataFrame({"s1_id": es, "role": "es"})],
                     ignore_index=True)


def label(f: pd.DataFrame, ids) -> pd.Series:
    g = io.load_gt_pairs(ids)
    pos = pd.MultiIndex.from_arrays([g.s1_id, g.match_id])
    return pd.Series(pd.MultiIndex.from_arrays([f.s1_id, f.cand_id]).isin(pos), index=f.index)


def main(a):
    assert a.cand_ver == "keys_v0", "v1 cache is built by pipelines/blocking_v1.py + features"
    out = paths.ROOT / "data" / "cands" / a.cand_ver
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not a.skip_train:
        ids = train_ids(a.n_train)
        ids.to_parquet(out.parent / "train_ids.parquet", index=False)
        f, _ = bv0.collect("train", set(ids.s1_id), a.chunk)
        f["label"] = label(f, set(ids.s1_id)).to_numpy()
        f["role"] = f.s1_id.map(dict(zip(ids.s1_id, ids.role)))
        f.to_parquet(out / "train.parquet", index=False)
        print(f"train: {len(f):,} pairs, pos rate {f.label.mean():.4f}, {(time.time() - t0) / 60:.1f} min", flush=True)
        del f
    ctx = harness.EvalContext.load(a.subset)
    f, _ = bv0.collect("train", ctx.ids, a.chunk)
    f["label"] = label(f, ctx.ids).to_numpy()
    f["role"] = "eval"
    f.to_parquet(out / f"{a.subset}.parquet", index=False)
    print(f"{a.subset}: {len(f):,} pairs, pos rate {f.label.mean():.4f}, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-ver", default="keys_v0")
    ap.add_argument("--subset", default="mini", choices=["micro", "mini", "fold0"])
    ap.add_argument("--n-train", type=int, default=150_000)
    ap.add_argument("--chunk", type=int, default=40_000)
    ap.add_argument("--skip-train", action="store_true")
    main(ap.parse_args())
