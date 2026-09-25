"""Stage 5: export pair text for the Kaggle cross-encoder (kaggle/xenc_train_score.py).

Band selector = the v1_n1 GATE model (base feats + rbits), fixed across tags so the exported sets never depend on
the stage-1 model that is still evolving. decision_v1 --xenc later masks the feature to the uncertain band of the
*current* stage-1 prob, so the exported bands are generous supersets of 0.02 < p < 0.98.

Cross-fitting (no in-sample logits reach stage 2): train-role S1 (folds 2-4) are split into halves cf=0/1 by S1
hash; model k trains on half k and scores half 1-k. es (fold 1) / eval / test rows have cf=-1 -> both models.

Upload dir data/kaggle/xenc/ (pair_id, text_a, text_b[, label], cf) - NO entity ids leave the box.
Map dir   data/kaggle/xenc_map/<name>.parquet (pair_id, s1_id, cand_id, gate_prob) - stays local.

Parts (--parts), fast ones first so the GPU can start while the CPU does the slow full-gate passes:
  trainset     train.parquet  train-role S1: all positives + NEG_HARD hardest + NEG_RAND random negatives
               valid.parquet  es S1 (fold 1): same recipe (kernel metrics only)
               hardness = raw score of the gate's first RANK_TREES trees (1562 trees in full = too slow on 17M)
  mini         band_mini.parquet  exact gate probs from the gate run's val_pred.parquet, LO < p < HI
  score_train  score_train.parquet  train+es roles, full gate (in-sample) TR_LO < p < TR_HI    [slow]
  fold0x,test  band_<tag>.parquet  full gate LO < p < HI, streamed per row group              [slow]

    python pipelines/xenc_export.py --parts trainset,mini
    python pipelines/xenc_export.py --parts score_train,test --threads 4
"""
import _bootstrap  # noqa: F401

import argparse
import gc
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from ber import blocking as B
from ber import paths

SEED = 42
GATE_RUN = "20260925-2015_aryan-bhendarkar_gate-blocking-v1-n1"
CACHE = paths.DATA_DIR / "cands" / "v1_n1"
OUT = paths.DATA_DIR / "kaggle" / "xenc"
MAP = paths.DATA_DIR / "kaggle" / "xenc_map"
LO, HI = 0.005, 0.995            # eval/test superset of the 0.02<p<0.98 band
TR_LO, TR_HI = 0.001, 0.999      # train tag: gate probs are in-sample there -> wider
NEG_HARD, NEG_RAND = 6, 2
RANK_TREES = 200
# pair_id offsets keep ids unique across files (joins are per file anyway)
OFFSET = {"train": 0, "valid": 100_000_000, "score_train": 200_000_000, "band_mini": 300_000_000,
          "band_fold0x": 400_000_000, "band_test": 500_000_000}


def gate():
    art = paths.ART_DIR / GATE_RUN
    return lgb.Booster(model_file=str(art / "model.lgb")), json.loads((art / "features.json").read_text())


def stream_predict(path, model, feats, threads, keep_cols, lo, hi, n_iter=None, raw=False, log=print):
    """Stream a cache by row group, predict the gate (optionally first n_iter trees, raw), keep lo < score < hi."""
    pf = pq.ParquetFile(path)
    cols = list(dict.fromkeys(["s1_id", "cand_id"] + keep_cols + feats))
    out, t0, n = [], time.time(), 0
    for rg in range(pf.num_row_groups):
        df = pl.from_arrow(pf.read_row_group(rg, columns=cols))
        n += df.height
        p = model.predict(df.select(feats).to_pandas(), num_threads=threads, num_iteration=n_iter,
                          raw_score=raw).astype(np.float32)
        out.append(df.select(["s1_id", "cand_id"] + keep_cols).with_columns(pl.Series("gate_prob", p))
                   .filter((pl.col("gate_prob") > lo) & (pl.col("gate_prob") < hi)))
        log(f"  {path.name} rg {rg + 1}/{pf.num_row_groups} {n:,} rows {time.time() - t0:.0f}s", flush=True)
        del df
        gc.collect()
    return pl.concat(out)


def texts(split: str) -> pl.DataFrame:
    """entity_id -> 'name | address' (norm_v1 n_full / a_full) for S1+S2+S3 of a split."""
    parts = []
    for s in (1, 2, 3):
        t = pl.read_parquet(B.norm_file(split, s, 1), columns=["entity_id", "n_full", "a_full"])
        parts.append(t.select("entity_id", (pl.col("n_full").fill_null("") + " | " + pl.col("a_full").fill_null(""))
                              .str.strip_chars().alias("text")))
    return pl.concat(parts)


def write(name: str, df: pl.DataFrame, T: pl.DataFrame, extra: list[str]):
    df = df.sort(["s1_id", "cand_id"]).with_row_index("pair_id", offset=OFFSET[name]) \
           .with_columns(pl.col("pair_id").cast(pl.Int64))
    df = df.join(T.rename({"entity_id": "s1_id", "text": "text_a"}), on="s1_id", how="left") \
           .join(T.rename({"entity_id": "cand_id", "text": "text_b"}), on="cand_id", how="left")
    miss = df.filter(pl.col("text_a").is_null() | pl.col("text_b").is_null()).height
    assert miss == 0, f"{name}: {miss} pairs without text"
    OUT.mkdir(parents=True, exist_ok=True)
    MAP.mkdir(parents=True, exist_ok=True)
    df.select(["pair_id", "text_a", "text_b"] + extra).write_parquet(OUT / f"{name}.parquet", compression="zstd",
                                                                      compression_level=9)
    df.select("pair_id", "s1_id", "cand_id", "gate_prob").write_parquet(MAP / f"{name}.parquet", compression="zstd")
    lab = f", pos {int(df['label'].sum()):,}" if "label" in df.columns else ""
    print(f"[write] {name}: {df.height:,} pairs, {df['s1_id'].n_unique():,} S1{lab}, "
          f"{(OUT / f'{name}.parquet').stat().st_size / 1e6:.1f} MB", flush=True)


def pick(df: pl.DataFrame) -> pl.DataFrame:
    """All positives + per S1 the NEG_HARD highest-scored negatives + NEG_RAND random other negatives."""
    df = df.with_columns(pl.Series("_r", np.random.default_rng(SEED).random(df.height).astype(np.float32)))
    neg = df.filter(pl.col("label") == 0)
    hard = neg.filter(pl.col("gate_prob").rank("ordinal", descending=True).over("s1_id") <= NEG_HARD)
    rest = neg.join(hard.select("s1_id", "cand_id"), on=["s1_id", "cand_id"], how="anti")
    easy = rest.filter(pl.col("_r").rank("ordinal").over("s1_id") <= NEG_RAND)
    return pl.concat([df.filter(pl.col("label") == 1), hard, easy]).drop("_r")


def train_roles(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("label").cast(pl.Int8),
                           pl.when(pl.col("role") == "train")
                           .then((pl.col("s1_id").hash(seed=SEED) % 2).cast(pl.Int8))
                           .otherwise(pl.lit(-1, pl.Int8)).alias("cf"))


def part_trainset(model, feats, threads):
    df = train_roles(stream_predict(CACHE / "train.parquet", model, feats, threads, ["label", "role"], -1e9, 1e9,
                                    n_iter=RANK_TREES, raw=True))
    T = texts("train")
    cols = ["s1_id", "cand_id", "gate_prob", "label", "cf"]      # gate_prob = partial raw score here
    write("train", pick(df.filter(pl.col("role") == "train")).select(cols), T, ["label", "cf"])
    write("valid", pick(df.filter(pl.col("role") == "es")).select(cols), T, ["label", "cf"])


def part_score_train(model, feats, threads):
    df = train_roles(stream_predict(CACHE / "train.parquet", model, feats, threads, ["label", "role"], TR_LO, TR_HI))
    for lo, hi in ((0.02, 0.98), (LO, HI)):
        b = df.filter((pl.col("gate_prob") > lo) & (pl.col("gate_prob") < hi))
        print(f"  train tag in-sample band ({lo},{hi}): {b.height:,} pairs, pos {int(b['label'].sum()):,}")
    write("score_train", df.select("s1_id", "cand_id", "gate_prob", "cf"), texts("train"), ["cf"])


def part_mini():
    v = pl.read_parquet(paths.ART_DIR / GATE_RUN / "val_pred.parquet").rename({"prob": "gate_prob"})
    b = v.filter((pl.col("gate_prob") > LO) & (pl.col("gate_prob") < HI))
    print(f"  mini: {v.height:,} pairs -> superset {b.height:,}")
    write("band_mini", b.with_columns(pl.lit(-1, pl.Int8).alias("cf")), texts("train"), ["cf"])


def part_band(tag, model, feats, threads):
    p = CACHE / f"{tag}.parquet"
    try:
        n = pq.ParquetFile(p).metadata.num_rows
    except Exception as e:  # noqa: BLE001 - missing or still being written
        print(f"[skip] {tag}: cache not readable ({type(e).__name__}) - rerun when it exists")
        return
    b = stream_predict(p, model, feats, threads, [], LO, HI)
    n98 = b.filter((pl.col("gate_prob") > 0.02) & (pl.col("gate_prob") < 0.98)).height
    print(f"  {tag}: {n:,} pairs -> superset {b.height:,} (0.02-0.98: {n98:,})")
    write(f"band_{tag}", b.with_columns(pl.lit(-1, pl.Int8).alias("cf")),
          texts("test" if tag == "test" else "train"), ["cf"])


def main(a):
    t0 = time.time()
    os.environ.setdefault("POLARS_MAX_THREADS", str(a.threads))
    model, feats = gate()
    for part in a.parts.split(","):
        print(f"[{part}]", flush=True)
        if part == "trainset":
            part_trainset(model, feats, a.threads)
        elif part == "score_train":
            part_score_train(model, feats, a.threads)
        elif part == "mini":
            part_mini()
        else:
            part_band(part, model, feats, a.threads)
        gc.collect()
    size = sum(f.stat().st_size for f in OUT.glob("*.parquet")) / 1e9
    print(f"[done] {(time.time() - t0) / 60:.1f} min, upload dir {size:.2f} GB")
    assert size < 2.0, "upload dir over 2 GB"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="trainset,mini")
    ap.add_argument("--threads", type=int, default=2)
    main(ap.parse_args())
