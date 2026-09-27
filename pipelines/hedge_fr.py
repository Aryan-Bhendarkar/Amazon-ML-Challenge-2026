"""HEDGE file (lead): a run's C1-path matches for US/IN + the France rows of a reference C1 matches file (default: psemb-C1,
the LB 0.968 file). Writes artifacts/<run>/test_matches_c1_frref.parquet; build the submission with make_submission.

    python pipelines/hedge_fr.py --run <run> [--ref-run 20260926-1737_…-psemb]"""
import _bootstrap  # noqa: F401

import argparse

import polars as pl

from ber import ctx_features as cf
from ber import paths

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--ref-run", default="20260926-1737_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3-psemb")
a = ap.parse_args()
fr = pl.read_parquet(cf._norm_file("test", 1, 1), columns=["entity_id", "country"]).filter(pl.col("country") == "France")["entity_id"]
b = pl.read_parquet(paths.ART_DIR / a.run / "test_matches_c1.parquet")
p = pl.read_parquet(paths.ART_DIR / a.ref_run / "test_matches_c1.parquet")
h = pl.concat([b.filter(~pl.col("s1_id").is_in(fr.implode())), p.filter(pl.col("s1_id").is_in(fr.implode())).select(b.columns)])
assert h["match_id"].is_unique().all()
out = paths.ART_DIR / a.run / "test_matches_c1_frref.parquet"
h.write_parquet(out)
print(out, h.height)
