"""AUDIT (27 Sep): error analysis + prediction-level rules for run D on mini / fold0x, scored with DM-val.

build : python pipelines/audit_d.py build --tag fold0x     -> artifacts/audit_d/<tag>_{asg,s1,miss}.parquet
DM scoring reuses ber.dmval semantics (band [lo, hi] negatives replicated w times, 5 seeded draws) but takes an
arbitrary boolean selection per assigned row (rules), not only prob >= t.
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import polars as pl

from ber import dmval, harness, io, paths

RUN_D = "20260927-0508_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3-d"
PRED = {"mini": paths.ART_DIR / RUN_D / "val_pred.parquet",
        "fold0x": paths.ART_DIR / "20260927-0519_aryan-bhendarkar_dm-val-fold0x" / "fold0x_pred.parquet"}
DM = {"lo": 0.2, "hi": 0.975, "w": 2.7503, "t": 0.825, "draws": 5}
OUT = paths.ART_DIR / "audit_d"
BASE_COLS = ["name_tset", "addr_tset", "street_tset", "house_rel", "extra_tok", "missing_tok", "cand_addr_empty",
             "cand_src", "cand_native", "n_cand_s1", "name_rank_in_s1", "addr_rank_in_s1", "kmask", "rbits"]


def build(tag: str):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = harness.EvalContext.load(tag)
    ids = sorted(ctx.ids)
    pred = pl.read_parquet(PRED[tag])
    # assignment = ber.decision.assign_best_s1 semantics (best S1 per record; margin to the 2nd S1)
    srt = pred.sort(["cand_id", "prob"], descending=[False, True]).with_columns(
        pl.int_range(pl.len()).over("cand_id").alias("_r"))
    best = srt.filter(pl.col("_r") == 0).drop("_r")
    sec = srt.filter(pl.col("_r") == 1).select("cand_id", pl.col("prob").alias("p2"), pl.col("s1_id").alias("s1_2"))
    asg = best.join(sec, on="cand_id", how="left").with_columns(pl.col("p2").fill_null(0.0),
                                                                 (pl.col("prob") - pl.col("p2").fill_null(0.0)).alias("margin"))
    gt = io.load_gt_pairs(ids)
    own = pl.from_pandas(gt[["s1_id", "match_id"]]).rename({"s1_id": "true_s1", "match_id": "cand_id"})
    # true owner of any record (all S1, not only eval S1): records never matched have no owner
    allgt = pl.read_parquet(paths.PARQUET_DIR / "train_gt_pairs.parquet")
    oc = [c for c in allgt.columns if c != "s1_id"][0] if "match_id" not in allgt.columns else "match_id"
    owner_any = allgt.select(pl.col("s1_id").alias("owner_any"), pl.col(oc).alias("cand_id")).unique("cand_id")
    asg = (asg.join(own, on="cand_id", how="left").join(owner_any, on="cand_id", how="left")
              .with_columns((pl.col("true_s1") == pl.col("s1_id")).fill_null(False).alias("lab")))
    cache = pl.scan_parquet(paths.DATA_DIR / "cands" / "v1_n2" / f"{tag}.parquet").select(["s1_id", "cand_id"] + BASE_COLS)
    asg = asg.join(cache.collect(), on=["s1_id", "cand_id"], how="left")
    cty = pl.DataFrame({"s1_id": list(ctx.country.keys()), "country": list(ctx.country.values())})
    asg = asg.join(cty, on="s1_id", how="left")
    asg.write_parquet(OUT / f"{tag}_asg.parquet")
    # per-S1 truth + fate of every true record: in candidates? assigned to its S1? p?
    n_true = pl.DataFrame({"s1_id": ids, "n_true": [len(ctx.truth.get(k, ())) for k in ids]}).join(cty, on="s1_id", how="left")
    n_true.write_parquet(OUT / f"{tag}_s1.parquet")
    tr = pl.from_pandas(gt[["s1_id", "match_id"]]).rename({"match_id": "cand_id"})
    incand = pred.select("s1_id", "cand_id", pl.col("prob").alias("p_own"))
    miss = (tr.join(incand, on=["s1_id", "cand_id"], how="left")
              .join(asg.select("cand_id", pl.col("s1_id").alias("asg_s1"), pl.col("prob").alias("p_asg")), on="cand_id", how="left")
              .with_columns(pl.when(pl.col("p_own").is_null()).then(pl.lit("blocking_miss"))
                            .when(pl.col("asg_s1") != pl.col("s1_id")).then(pl.lit("lost_to_other_s1"))
                            .otherwise(pl.lit("in_cand_assigned")).alias("fate")))
    miss.write_parquet(OUT / f"{tag}_miss.parquet")
    print(f"[{tag}] asg {asg.height:,}, S1 {len(ids):,}, true pairs {tr.height:,}; fates "
          f"{miss.group_by('fate').len().rows()} {time.time() - t0:.0f}s", flush=True)


def load(tag: str):
    asg = pd.read_parquet(OUT / f"{tag}_asg.parquet")
    s1 = pd.read_parquet(OUT / f"{tag}_s1.parquet").set_index("s1_id")
    return asg, s1


def entity_scores(asg: pd.DataFrame, n_true: pd.Series, sel: np.ndarray, dm: bool, draws: int = DM["draws"]) -> pd.Series:
    """Per-S1 F0.5 for an arbitrary selection of assigned rows (clean, or DM-weighted averaged over draws)."""
    def one(c):
        lab = asg["lab"].to_numpy()
        g = pd.DataFrame({"s1_id": asg["s1_id"].to_numpy(), "tp": (sel & lab).astype(float),
                          "fp": (sel & ~lab) * c}).groupby("s1_id")[["tp", "fp"]].sum().reindex(n_true.index, fill_value=0.0)
        tp, fp, nt = g.tp.to_numpy(), g.fp.to_numpy(), n_true.to_numpy().astype(float)
        f = np.where(tp > 0, 1.25 * tp / np.maximum(1.25 * tp + 0.25 * (nt - tp) + fp, 1e-12), 0.0)
        return pd.Series(np.where(nt == 0, (fp == 0).astype(float), f), index=n_true.index)
    if not dm:
        return one(np.ones(len(asg)))
    return sum(one(dmval.copies(asg, DM["w"], DM["lo"], DM["hi"], seed=42 + d)) for d in range(draws)) / draws


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--tag", default="fold0x")
    a = ap.parse_args()
    os.environ.setdefault("POLARS_MAX_THREADS", "6")
    build(a.tag)
