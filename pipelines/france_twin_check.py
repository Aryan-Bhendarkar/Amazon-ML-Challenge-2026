"""Label-free France check for templated same-name / different-street matches (audit I1 of the v1_n2 run).

Val has no France and no cross-S1 competition, so this compares TEST predictions of two runs on France:
among matched pairs whose only retriever is nkey_num (rbits == 512), how many look like a DIFFERENT street
(significant street tokens, i.e. tokens in < 0.3% of the country's S1 streets, with token_set_ratio < 50)?
Recipe from the auditor's chk6.py. Lower is better, if the number of matches doesn't collapse.

    python pipelines/france_twin_check.py --runs <run_a>:test_matches.parquet,<run_b>:test_matches_slice.parquet
"""
import _bootstrap  # noqa: F401

import argparse
import json

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

from ber import paths

FD = paths.FEATURE_DIR


def main(a):
    s1 = pl.read_parquet(FD / "norm_v1_test_s1.parquet", columns=["entity_id", "country", "a_street"]) \
        .rename({"entity_id": "s1_id", "a_street": "st"})
    fr = s1.filter(pl.col("country") == a.country)
    c = pl.concat([pl.read_parquet(FD / f"norm_v1_test_s{k}.parquet", columns=["entity_id", "a_street"])
                   for k in (2, 3)]).rename({"entity_id": "cand_id", "a_street": "st_c"})
    toks = fr.select(pl.col("st").fill_null("").str.split(" ").list.unique().alias("t")).explode("t", empty_as_null=True) \
        .filter(pl.col("t") != "")
    common = set(toks.group_by("t").len("df").filter(pl.col("df") > 0.003 * fr.height)["t"].to_list())
    pf = pq.ParquetFile(paths.DATA_DIR / "cands" / a.cache / "test.parquet")
    frids = set(fr["s1_id"].to_list())
    n512 = []
    for rg in range(pf.num_row_groups):
        t = pl.from_arrow(pf.read_row_group(rg, columns=["s1_id", "cand_id", "rbits"]))
        t = t.filter((pl.col("rbits") == 512) & pl.col("s1_id").is_in(list(frids)))
        if t.height:
            n512.append(t.select("s1_id", "cand_id"))
    n512 = pl.concat(n512)

    def sig(st):
        return " ".join(sorted(x for x in (st or "").split() if x not in common and not x.isdigit()))

    out = {}
    for spec in a.runs.split(","):
        run, fname = spec.split(":")
        m = pl.read_parquet(paths.ART_DIR / run / fname).rename({"match_id": "cand_id"}) \
            .filter(pl.col("s1_id").is_in(list(frids)))
        m512 = m.join(n512, on=["s1_id", "cand_id"]).join(fr.select("s1_id", "st"), on="s1_id").join(c, on="cand_id")
        sa, sb = [sig(x) for x in m512["st"].to_list()], [sig(x) for x in m512["st_c"].to_list()]
        r = process.cpdist(sa, sb, scorer=fuzz.token_set_ratio, workers=a.threads)
        both = np.array([bool(x) and bool(y) for x, y in zip(sa, sb)])
        diff = int(((r < 50) & both).sum())
        out[run] = {"file": fname, "fr_matches": m.height, "fr_s1_with_match": m["s1_id"].n_unique(),
                    "nkey_only_matches": m512.height, "both_have_sig_street": int(both.sum()),
                    "diff_street": diff, "diff_street_rate": round(diff / max(both.sum(), 1), 4)}
        print(run[-45:], json.dumps(out[run]))
    if a.out:
        (paths.ROOT / a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="run_id:matches_file,... (same country subset is compared)")
    ap.add_argument("--cache", default="v1_n2")
    ap.add_argument("--country", default="France")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="")
    main(ap.parse_args())
