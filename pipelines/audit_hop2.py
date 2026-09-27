"""AUDIT C5: record<->record (2-hop) similarity to the S1's ACCEPTED records, as prediction-level features.

For every assigned row with p >= P_MIN whose S1 has >= 1 accepted record (p >= t, a different record):
  hop_full = max token_set_ratio(rec "name | address", accepted sibling text) over the S1's accepted records
  hop_name = same on names only, hop_n = number of accepted siblings
Adds columns to artifacts/audit_d/<tag>_asg.parquet -> <tag>_asg_hop.parquet. Text = norm_v1 n_full / a_full (train split).
    python pipelines/audit_hop2.py --tag mini ; python pipelines/audit_hop2.py --tag fold0x
"""
import _bootstrap  # noqa: F401

import argparse
import time

import numpy as np
import pandas as pd
import polars as pl
from rapidfuzz import fuzz, process

import audit_d as A
from ber import ctx_features as cf

P_MIN = 0.05


def main(a):
    t0 = time.time()
    asg = pl.read_parquet(A.OUT / f"{a.tag}_asg.parquet")
    t = A.DM["t"]
    acc = asg.filter(pl.col("prob") >= t).select("s1_id", pl.col("cand_id").alias("sib_id"))
    q = asg.filter(pl.col("prob") >= P_MIN).select("s1_id", "cand_id")
    pairs = q.join(acc, on="s1_id").filter(pl.col("cand_id") != pl.col("sib_id"))
    ids = pl.concat([pairs["cand_id"], pairs["sib_id"]]).unique()
    rec = pl.concat([pl.read_parquet(cf._norm_file("train", s, 1), columns=["entity_id", "n_full", "a_full"]) for s in (2, 3)]) \
            .filter(pl.col("entity_id").is_in(ids.implode())).with_columns(pl.col("n_full", "a_full").fill_null(""))
    rec = rec.with_columns((pl.col("n_full") + " | " + pl.col("a_full")).alias("full"))
    txt = dict(zip(rec["entity_id"].to_list(), zip(rec["n_full"].to_list(), rec["full"].to_list())))
    c, s = pairs["cand_id"].to_list(), pairs["sib_id"].to_list()
    W = dict(workers=a.workers, dtype=np.float32)
    full = process.cpdist([txt[x][1] for x in c], [txt[x][1] for x in s], scorer=fuzz.token_set_ratio, **W)
    name = process.cpdist([txt[x][0] for x in c], [txt[x][0] for x in s], scorer=fuzz.token_set_ratio, **W)
    hop = (pairs.with_columns(pl.Series("hf", full), pl.Series("hn", name))
                .group_by(["s1_id", "cand_id"]).agg(pl.col("hf").max().alias("hop_full"),
                                                     pl.col("hn").max().alias("hop_name"), pl.len().alias("hop_n")))
    out = asg.join(hop, on=["s1_id", "cand_id"], how="left")
    out.write_parquet(A.OUT / f"{a.tag}_asg_hop.parquet")
    cov = out.filter(pl.col("prob") >= P_MIN)["hop_full"].is_not_null().mean()
    print(f"[{a.tag}] sibling pairs {pairs.height:,}; rows with hop feats {hop.height:,} (coverage of p>={P_MIN}: {cov:.3f}) "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mini")
    ap.add_argument("--workers", type=int, default=6)
    main(ap.parse_args())
