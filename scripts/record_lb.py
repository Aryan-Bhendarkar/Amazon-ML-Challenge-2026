"""Mark a prepared submission as uploaded and store its public leaderboard score.
Usage: python scripts/record_lb.py <sub_id> --score 0.8123 [--note "..."]
"""
import _bootstrap  # noqa: F401

import argparse
import json
from datetime import datetime

from ber import paths
from ber.tracking import IST

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sub_id")
    ap.add_argument("--score", type=float, required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    p = paths.SUB_DIR / "records" / f"{a.sub_id}.json"
    r = json.loads(p.read_text(encoding="utf-8"))
    r.update(status="uploaded", public_score=a.score,
             uploaded_ist=datetime.now(IST).isoformat(timespec="seconds"))
    if a.note:
        r["lb_note"] = a.note
    p.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    recs = [json.loads(q.read_text(encoding="utf-8")) for q in sorted((paths.SUB_DIR / "records").glob("*.json"))]
    lines = ["# Leaderboard submissions (max 5/day, 15 total)", "",
             "| sub_id | run | val F0.5 | public LB | status | note |", "|---|---|---|---|---|---|"]
    for x in recs:
        lines.append(f"| {x['sub_id']} | {x.get('run_id','')} | {x.get('val_f05') or ''} | "
                     f"{x.get('public_score') or ''} | {x['status']} | {x.get('note','')} |")
    (paths.SUB_DIR / "LOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"recorded {a.sub_id}: public {a.score}")
