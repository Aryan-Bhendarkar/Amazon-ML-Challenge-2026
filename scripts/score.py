"""Score validation predictions against train GT with the official macro F0.5.

Input: a parquet/tsv with columns (s1_id, match_id) -- one row per predicted pair --, OR a
matching_results-style TSV (source1_entity_id, matched_entity_ids), OR a probability parquet
(s1_id, cand_id, prob) + --threshold (assign_best_s1 is applied first).
Usage:  python scripts/score.py preds.parquet --subset mini [--threshold 0.6]   (micro|mini|fold0)
"""
import _bootstrap  # noqa: F401

import argparse
import json

import pandas as pd

from ber import io, metric, split


def load_pred(path: str, threshold=None) -> dict:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        if "prob" in df.columns:
            assert threshold is not None, "probability file: pass --threshold"
            from ber.decision import assign_best_s1, threshold_matches
            return {k: set(v) for k, v in threshold_matches(assign_best_s1(df), threshold).items()}
        return metric.pairs_to_sets(df)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "matched_entity_ids" in df.columns:
        return {r.source1_entity_id: set(filter(None, r.matched_entity_ids.split(",")))
                for r in df.itertuples()}
    return metric.pairs_to_sets(df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("--subset", default="mini", choices=["micro", "mini", "fold0"])
    ap.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args()
    folds = io.load_folds()
    ids = split.eval_ids(folds, a.subset)
    truth = io.load_gt_sets(ids)
    country = dict(zip(folds.s1_id, folds.country))
    print(json.dumps(metric.report(load_pred(a.pred, a.threshold), truth, country), indent=2))
