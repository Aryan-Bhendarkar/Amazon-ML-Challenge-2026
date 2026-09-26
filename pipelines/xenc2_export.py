"""xenc revival (lead 19:05/19:30/20:10): pair sets for scoring the EXISTING cross-fitted MiniLM ckpts (amlc-xenc
ckpt_m0/m1) and for Lane G's new cross-encoders.

Pair sets = stage-1 p >= PMIN (0.01: the floor of the saved test predictions, so every split uses the same cut):
  train  : v1_n2 train cache, decision_v1 4-fold OOF probs (v1_n1 pairs; nkey_num-only pairs have no OOF -> not scored)
           cf = s1_id.hash(42) % 2 for role 'train' (the halves the ckpts were cross-fitted on; verified 0 mismatches
           vs data/kaggle/xenc_map), -1 for role 'es' (fold 1, never trained on) -> OOF logits only
  mini / fold0x / test : psemb (20260926-1737) predictions, cf = -1 (scored by both models, averaged)
Outputs
  data/kaggle/xenc2/score_<tag>.parquet   pair_id, text_a, text_b, cf    ("n_full | a_full", the ckpts' format)
  data/kaggle/xenc2_map/<tag>.parquet     pair_id, s1_id, cand_id        (stays on the box)
  data/kaggle/g3/{pairs,records}.parquet  Lane G: pair_id, tag, cf, k1, k2 + key, name, addr, house (raw fields,
                                          norm_v1; Lane G builds its own "[COL] name [VAL] ..." serialization)
"""
import _bootstrap  # noqa: F401

import os

import polars as pl

from ber import ctx_features as cf
from ber import paths

PMIN = 0.01
PSEMB = paths.ART_DIR / "20260926-1737_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3-psemb"
SRC = {"train": paths.ART_DIR / "20260925-2235_aryan-bhendarkar_decision-v1-v1-n1" / "oof_train.parquet",
       "mini": PSEMB / "val_pred.parquet",
       "fold0x": paths.ART_DIR / "20260926-1803_aryan-bhendarkar_dm-val-fold0x" / "fold0x_pred.parquet",
       "test": PSEMB / "test_pred.parquet"}
OFFSET = {"train": 0, "mini": 100_000_000, "fold0x": 200_000_000, "test": 300_000_000}
OUT, MAP, G3 = (paths.DATA_DIR / "kaggle" / d for d in ("xenc2", "xenc2_map", "g3"))


def records(split: str) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(cf._norm_file(split, s, 1), columns=["entity_id", "n_full", "a_full", "a_house"])
                      for s in (1, 2, 3)]).with_columns(pl.col("n_full", "a_full", "a_house").fill_null(""))


def main():
    os.environ.setdefault("POLARS_MAX_THREADS", "4")
    for d in (OUT, MAP, G3):
        d.mkdir(parents=True, exist_ok=True)
    R = {"train": records("train"), "test": records("test")}
    roles = pl.read_parquet(paths.DATA_DIR / "cands" / "v1_n2" / "train.parquet", columns=["s1_id", "cand_id", "role"])
    g3_pairs, keys = [], []
    for tag, src in SRC.items():
        split = "test" if tag == "test" else "train"
        d = pl.scan_parquet(src).filter(pl.col("prob") >= PMIN).select("s1_id", "cand_id").collect()
        if tag == "train":
            d = d.join(roles, on=["s1_id", "cand_id"], how="inner").with_columns(
                pl.when(pl.col("role") == "train").then((pl.col("s1_id").hash(seed=42) % 2).cast(pl.Int8))
                .otherwise(pl.lit(-1, pl.Int8)).alias("cf")).drop("role")
        else:
            d = d.with_columns(pl.lit(-1, pl.Int8).alias("cf"))
        d = d.sort(["s1_id", "cand_id"]).with_row_index("pair_id", offset=OFFSET[tag])
        T = R[split].select("entity_id", (pl.col("n_full") + " | " + pl.col("a_full")).alias("t"))
        x = (d.join(T.rename({"entity_id": "s1_id", "t": "text_a"}), on="s1_id", how="left")
              .join(T.rename({"entity_id": "cand_id", "t": "text_b"}), on="cand_id", how="left"))
        assert x.filter(pl.col("text_a").is_null() | pl.col("text_b").is_null()).height == 0
        x.select("pair_id", "text_a", "text_b", "cf").write_parquet(OUT / f"score_{tag}.parquet", compression="zstd")
        d.select("pair_id", "s1_id", "cand_id").write_parquet(MAP / f"{tag}.parquet", compression="zstd")
        g3_pairs.append(d.select("pair_id", pl.lit(tag).alias("tag"), "cf", (pl.lit(split + "|") + pl.col("s1_id")).alias("k1"),
                                 (pl.lit(split + "|") + pl.col("cand_id")).alias("k2")))
        print(f"{tag}: {d.height:,} pairs (cf counts {d.group_by('cf').len().sort('cf').rows()})", flush=True)
    P = pl.concat(g3_pairs)
    P.write_parquet(G3 / "pairs.parquet", compression="zstd")
    need = pl.concat([P.select(pl.col("k1").alias("key")), P.select(pl.col("k2").alias("key"))]).unique()
    rec = pl.concat([R[s].select((pl.lit(s + "|") + pl.col("entity_id")).alias("key"), pl.col("n_full").alias("name"),
                                 pl.col("a_full").alias("addr"), pl.col("a_house").alias("house")) for s in ("train", "test")])
    rec.join(need, on="key", how="semi").write_parquet(G3 / "records.parquet", compression="zstd")
    print("sizes MB", {f"{d.name}/{f.name}": round(f.stat().st_size / 1e6, 1) for d in (OUT, G3) for f in d.glob("*.parquet")})


if __name__ == "__main__":
    main()
