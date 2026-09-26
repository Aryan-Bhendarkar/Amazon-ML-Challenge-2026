"""Lane G / G1: export cross-encoder training pairs for the Kaggle G2 kernels (kaggle/xenc_g2.py).

From the v1_n2 TRAIN cache (train role = folds 2-4 S1; es role = fold 1):
  train.parquet  pair_id, text_a, text_b, label, cf     cf = md5(s1_id) % 2 (cross-fit half; model k trains on cf == k)
                 all positives + hard negatives (drop negatives with name_tset < 40 AND addr_tset < 40), capped:
                 the N_HARD hardest (name_tset + addr_tset) + N_RAND random of the remaining hard negatives
  valid.parquet  same recipe on fold 1 (es role), random VALID_N rows, cf = -1
Text (Ditto-style serialization, lead 20:10): "[COL] name [VAL] <n_full> [COL] address [VAL] <a_full> [NUM] <house>"
on norm_v2 (== norm_v1 on train: France rules are no-ops there). Legal forms and digits kept.
Upload dir: data/kaggle/g1/ (NO entity ids). Local map: data/kaggle/g1_map/{train,valid}.parquet (pair_id, s1_id, cand_id).

    python pipelines/xenc_g1_export.py
"""
import _bootstrap  # noqa: F401

import hashlib
import time

import numpy as np
import polars as pl

from ber import paths

N_HARD, N_RAND, VALID_N, SEED = 5_500_000, 2_000_000, 300_000, 42
OUT = paths.DATA_DIR / "kaggle" / "g1"
MAP = paths.DATA_DIR / "kaggle" / "g1_map"


def texts(split: str = "train", v: int = 2) -> tuple[pl.DataFrame, pl.DataFrame]:
    def load(sources):
        d = pl.concat([pl.read_parquet(paths.FEATURE_DIR / f"norm_v{v}_{split}_s{s}.parquet",
                                       columns=["entity_id", "n_full", "a_full", "a_house"]) for s in sources])
        return d.select("entity_id", pl.concat_str([
            pl.lit("[COL] name [VAL] "), pl.col("n_full").fill_null(""),
            pl.lit(" [COL] address [VAL] "), pl.col("a_full").fill_null(""),
            pl.when(pl.col("a_house").fill_null("") != "").then(pl.lit(" [NUM] ") + pl.col("a_house"))
              .otherwise(pl.lit(""))]).alias("text"))
    return load((1,)), load((2, 3))


def half(ids: pl.Series) -> np.ndarray:
    return np.array([int(hashlib.md5(s.encode()).hexdigest(), 16) % 2 for s in ids.to_list()], dtype=np.int8)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    MAP.mkdir(parents=True, exist_ok=True)
    c = (pl.scan_parquet(paths.DATA_DIR / "cands" / "v1_n2" / "train.parquet")
           .select("s1_id", "cand_id", "role", "label", "name_tset", "addr_tset")
           .filter(pl.col("label") | (pl.col("name_tset") >= 40) | (pl.col("addr_tset") >= 40)).collect())
    s1t, pt = texts()
    rng = np.random.default_rng(SEED)
    for role, name in (("train", "train"), ("es", "valid")):
        d = c.filter(pl.col("role") == role).with_columns((pl.col("name_tset") + pl.col("addr_tset")).alias("_h"))
        pos, neg = d.filter(pl.col("label")), d.filter(~pl.col("label")).sort("_h", descending=True)
        if name == "train":
            rest = neg.slice(N_HARD)
            neg = pl.concat([neg.head(N_HARD), rest.sample(min(N_RAND, rest.height), seed=SEED)])
            d = pl.concat([pos, neg])
            d = d.with_columns(pl.Series("cf", half(d["s1_id"])))
        else:
            d = pl.concat([pos, neg]).sample(VALID_N, seed=SEED).with_columns(pl.lit(-1, dtype=pl.Int8).alias("cf"))
        d = d.sample(fraction=1.0, shuffle=True, seed=SEED).with_row_index("pair_id")
        d = d.with_columns((pl.lit(f"{name[0]}") + pl.col("pair_id").cast(pl.String)).alias("pair_id"))
        d = (d.join(s1t.rename({"entity_id": "s1_id", "text": "text_a"}), on="s1_id", how="left")
              .join(pt.rename({"entity_id": "cand_id", "text": "text_b"}), on="cand_id", how="left"))
        assert d["text_a"].null_count() == 0 and d["text_b"].null_count() == 0
        d.select("pair_id", "text_a", "text_b", pl.col("label").cast(pl.Int8), "cf") \
         .write_parquet(OUT / f"{name}.parquet", compression="zstd")
        d.select("pair_id", "s1_id", "cand_id").write_parquet(MAP / f"{name}.parquet")
        by = d.group_by("cf").agg(pl.len(), pl.col("label").mean().round(4)).sort("cf").rows()
        print(f"[{name}] {d.height:,} rows, pos {d['label'].mean():.4f}, by cf {by}  {time.time() - t0:.0f}s", flush=True)
    del rng


if __name__ == "__main__":
    main()
