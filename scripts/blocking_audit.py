"""Blocking audit: bucket EVERY missed true pair of an eval subset (rule-based, one bucket each)
and print a stratified sample for manual reading.

Usage: python scripts/blocking_audit.py --cands artifacts/<run>/val_pred.parquet [--subset mini]
          [--norm-v 0] [--sample 60] [--out experiments/runs/<run>/blocking_audit.md]
cands parquet needs columns s1_id, cand_id (extra columns ignored).
"""
import _bootstrap  # noqa: F401

import argparse
import time

import polars as pl
from rapidfuzz import fuzz, process

from ber import paths

C = ["entity_id", "n_core", "n_compact", "n_alias", "n_kind", "n_script", "a_house", "a_street",
     "a_empty", "a_full"]

# first matching rule wins -> exclusive buckets
BUCKETS = [
    ("native_script", pl.col("n_script_c") != "latin"),
    ("empty_name", pl.col("n_kind_c") == "empty"),
    ("domain_handle", pl.col("n_kind_c").is_in(["domain", "handle"])),
    ("alias", pl.col("n_alias_c") != ""),
    ("name_near_identical", pl.col("tset") >= 90),
    ("name_extra_or_reorder", (pl.col("tset") >= 70) & (pl.col("ratio") < 80)),
    ("name_typo_moderate", pl.col("ratio") >= 60),
    ("name_heavy_diff", pl.lit(True)),
]


def load_norm(split, sources, ids, norm_v):
    lf = pl.concat([pl.scan_parquet(paths.FEATURE_DIR / f"norm_v{norm_v}_{split}_s{s}.parquet").select(C)
                    for s in sources])
    return lf.join(ids.lazy(), on="entity_id").collect()


def main(a):
    t0 = time.time()
    fo = pl.read_parquet(paths.folds_path())
    fo = fo.filter(pl.col(a.subset)) if a.subset != "fold0" else fo.filter(pl.col("fold") == 0)
    fo = fo.select("s1_id", "country")
    gt = pl.scan_parquet(paths.gt_pairs_path()).join(fo.lazy(), on="s1_id").collect()
    cand = pl.read_parquet(a.cands, columns=["s1_id", "cand_id"]).rename({"cand_id": "match_id"})
    miss = gt.join(cand, on=["s1_id", "match_id"], how="anti")
    s1 = load_norm("train", [1], miss.select(pl.col("s1_id").alias("entity_id")).unique(), a.norm_v)
    pc = load_norm("train", [2, 3], miss.select(pl.col("match_id").alias("entity_id")).unique(), a.norm_v)
    m = miss.join(s1, left_on="s1_id", right_on="entity_id").join(pc, left_on="match_id", right_on="entity_id", suffix="_c")
    W = dict(workers=-1)
    m = m.with_columns(
        pl.Series("tset", process.cpdist(m["n_core"].to_list(), m["n_core_c"].to_list(), scorer=fuzz.token_set_ratio, **W)),
        pl.Series("ratio", process.cpdist(m["n_core"].to_list(), m["n_core_c"].to_list(), scorer=fuzz.ratio, **W)),
        pl.Series("addr_tset", process.cpdist(m["a_full"].to_list(), m["a_full_c"].to_list(), scorer=fuzz.token_set_ratio, **W)),
        pl.col("match_id").str.slice(0, 2).alias("src"),
    )
    expr = pl.when(BUCKETS[0][1]).then(pl.lit(BUCKETS[0][0]))
    for name, cond in BUCKETS[1:]:
        expr = expr.when(cond).then(pl.lit(name))
    m = m.with_columns(expr.alias("bucket"),
                       pl.when(pl.col("a_empty_c")).then(pl.lit("cand_addr_empty"))
                         .when(pl.col("a_house_c") == "").then(pl.lit("cand_no_house"))
                         .when(pl.col("a_house").str.strip_chars_start("0") == pl.col("a_house_c").str.strip_chars_start("0"))
                         .then(pl.when(pl.col("a_house") == pl.col("a_house_c")).then(pl.lit("house_equal"))
                               .otherwise(pl.lit("house_zero_pad")))
                         .otherwise(pl.lit("house_diff")).alias("addr_state"))
    n_true = gt.height
    lines = [f"# Blocking audit ({a.subset}, cands={a.cands})", "",
             f"true pairs {n_true:,} · missed {m.height:,} ({m.height / n_true:.4f}) · recall {1 - m.height / n_true:.4f}", ""]
    tab = (m.group_by("country", "bucket").len().pivot(on="country", index="bucket", values="len").fill_null(0)
             .with_columns(pl.sum_horizontal(pl.exclude("bucket")).alias("total")).sort("total", descending=True)
             .with_columns((pl.col("total") / n_true).round(4).alias("recall_pts")))
    lines += ["## Name bucket (exclusive) × country", "```", str(tab), "```", ""]
    tab2 = (m.group_by("bucket", "addr_state").len().pivot(on="addr_state", index="bucket", values="len").fill_null(0))
    lines += ["## Name bucket × address state of the candidate", "```", str(tab2), "```", ""]
    pl.Config.set_fmt_str_lengths(70)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_tbl_rows(100)
    samp = (m.with_columns(pl.int_range(pl.len()).shuffle(seed=42).over("country", "src").alias("r"))
              .filter(pl.col("r") < max(1, a.sample // 4)).sort("country", "src", "bucket"))
    lines += ["## Stratified sample", "```",
              str(samp.select("country", "src", "bucket", "addr_state", "tset", "n_core", "n_core_c", "a_full", "a_full_c")),
              "```"]
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    if a.save_misses:
        m.write_parquet(a.save_misses)
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--subset", default="mini", choices=["micro", "mini", "fold0"])
    ap.add_argument("--norm-v", type=int, default=0)
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-misses", default=None)
    main(ap.parse_args())
