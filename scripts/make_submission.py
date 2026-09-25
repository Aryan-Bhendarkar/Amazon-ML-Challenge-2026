"""Build a leaderboard submission from TEST predictions of a run, validate, diagnose, record.

Inputs (parquet, produced by your pipeline on the TEST split):
  --matches     columns s1_id, match_id   (final predicted pairs)
  --candidates  columns s1_id, cand_id    (exact set the model scored; required for final zip)
Writes submissions/files/<sub_id>/{matching_results.tsv,candidate_pairs.tsv} (git-ignored),
runs the official validator with --check-ids, prints per-country diagnostics, and saves
submissions/records/<sub_id>.json with status "prepared".
After uploading in the portal:  python scripts/record_lb.py <sub_id> --score 0.XXXX

Usage: python scripts/make_submission.py --run <run_id> --matches m.parquet --candidates c.parquet --note "..."
"""
import _bootstrap  # noqa: F401

import argparse
import json
from datetime import datetime

import pandas as pd

from ber import io, paths, submission
from ber.tracking import IST, author


def main(a) -> int:
    s1 = io.load_source("test", 1, columns=["entity_id", "country"])
    order = s1.entity_id.tolist()
    m = pd.read_parquet(a.matches)
    c = pd.read_parquet(a.candidates) if a.candidates else None
    if c is not None:
        cs = set(zip(c.s1_id, c.cand_id))
        missing = sum((x, y) not in cs for x, y in zip(m.s1_id, m.match_id))
        if missing:
            print(f"WARNING: {missing} matched pairs are not in candidates (pipeline bug?)")
    dup = m.duplicated("match_id").sum()
    if dup:
        print(f"WARNING: {dup} records matched to >1 S1 — violates the at-most-one structure; "
              "use ber.decision.assign_best_s1 before thresholding")
    sub_id = f"{datetime.now(IST):%Y%m%d-%H%M}_{author()}"
    out = paths.SUB_DIR / "files" / sub_id
    ml = m.groupby("s1_id").match_id.agg(list).to_dict()
    st_m = submission.write_id_lists(out / "matching_results.tsv", ml, order, submission.MATCH_HEADER)
    st_c = None
    if c is not None:
        cl = c.groupby("s1_id").cand_id.agg(list).to_dict()
        st_c = submission.write_id_lists(out / "candidate_pairs.tsv", cl, order, submission.CAND_HEADER)
    ok, log = submission.run_validator(out / "matching_results.tsv",
                                       out / "candidate_pairs.tsv" if c is not None else None,
                                       check_ids=not a.no_check_ids)
    print(log)
    diag = submission.diagnostics(ml, s1)
    print(json.dumps(diag, indent=2))
    rec = {"sub_id": sub_id, "run_id": a.run, "author": author(), "note": a.note,
           "created_ist": datetime.now(IST).isoformat(timespec="seconds"),
           "status": "prepared" if ok else "invalid", "validator_pass": ok,
           "val_f05": a.val_f05, "stats_matching": st_m, "stats_candidates": st_c,
           "diagnostics": diag, "files": str(out.relative_to(paths.ROOT))}
    p = submission.save_record(rec)
    print(f"\n{'PASS' if ok else 'FAIL'} -> {out / 'matching_results.tsv'}\nrecord: {p}")
    if ok:
        print("Upload matching_results.tsv in the portal, then: "
              f"python scripts/record_lb.py {sub_id} --score <public score>")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--matches", required=True)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--val-f05", type=float, default=None)
    ap.add_argument("--no-check-ids", action="store_true")
    raise SystemExit(main(ap.parse_args()))
