"""Data-driven hedge (replaces pipelines/hedge_fr.py, which hard-coded "France").

The cross-encoders were fine-tuned only on countries present in the training labels. For S1 whose country is NOT seen
in the training data, the pipeline falls back to the GBDT WITHOUT cross-encoder features (--base-run); every other S1
takes the matches of the model with cross-encoder features (--run). The set of seen countries is computed from the train
S1 table (open set: nothing is hard-coded, a new country in test is simply "unseen"). Country selects rows only; it is
never a model feature.

    python pipelines/hedge_unseen.py --run <xenc run> [--base-run <psemb run>] [--check <reference matches parquet>]
Inputs : artifacts/<run>/<--name>, artifacts/<base-run>/<--name> (default test_matches_c1.parquet: C1 path, France
         rows featurized on norm v2)
Output : artifacts/<run>/test_matches_hedge.parquet (s1_id, match_id) -> scripts/make_submission.py --matches ...
"""
import _bootstrap  # noqa: F401

import argparse
import json

import polars as pl

from ber import io, paths

PSEMB = "20260926-1737_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3-psemb"


def seen_countries() -> set:
    return set(io.load_source("train", 1, columns=["country"])["country"].dropna().unique().tolist())


def hedge(xenc_m: pl.DataFrame, base_m: pl.DataFrame, test_s1: pl.DataFrame, seen: set) -> tuple[pl.DataFrame, dict]:
    unseen_ids = test_s1.filter(~pl.col("country").is_in(sorted(seen)))["entity_id"]
    u = pl.col("s1_id").is_in(unseen_ids.implode())
    out = pl.concat([xenc_m.filter(~u).select("s1_id", "match_id"), base_m.filter(u).select("s1_id", "match_id")])
    assert out["match_id"].is_unique().all(), "a record is matched to more than one S1"
    by = test_s1.group_by("country").len("n_s1").with_columns(pl.col("country").is_in(sorted(seen)).alias("seen"))
    info = {"seen_countries": sorted(seen), "unseen_countries": sorted(set(test_s1["country"].unique()) - seen),
            "test_s1_by_country": {r["country"]: {"n_s1": r["n_s1"], "model": "xenc" if r["seen"] else "base (no xenc)"}
                                   for r in by.to_dicts()},
            "matches": out.height, "matches_from_base": int(base_m.filter(u).height)}
    return out, info


def main(a):
    seen = seen_countries()
    test_s1 = pl.from_pandas(io.load_source("test", 1, columns=["entity_id", "country"]))
    xm = pl.read_parquet(paths.ART_DIR / a.run / a.name)
    bm = pl.read_parquet(paths.ART_DIR / a.base_run / a.name)
    out, info = hedge(xm, bm, test_s1, seen)
    dst = paths.ART_DIR / a.run / "test_matches_hedge.parquet"
    out.write_parquet(dst)
    if a.check:
        ref = pl.read_parquet(a.check).select("s1_id", "match_id")
        same = out.sort(["s1_id", "match_id"]).equals(ref.sort(["s1_id", "match_id"]))
        info["check"] = {"reference": a.check, "identical": bool(same), "rows_ref": ref.height}
        assert same, f"hedge differs from the reference {a.check}"
    print(json.dumps(info, indent=1))
    print(f"-> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="model WITH cross-encoder features (seen countries)")
    ap.add_argument("--base-run", default=PSEMB, help="model WITHOUT cross-encoder features (unseen countries)")
    ap.add_argument("--name", default="test_matches_c1.parquet")
    ap.add_argument("--check", default="", help="assert identical to this matches parquet")
    main(ap.parse_args())
