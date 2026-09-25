"""One-screen competition status: time left, submission budget, best runs, latest runs.
Stdlib only (also used by the SessionStart hook).   Usage:  python scripts/status.py [--short]
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IST = timezone(timedelta(hours=5, minutes=30))
START = datetime(2026, 9, 25, 0, 0, tzinfo=IST)
DEADLINE = datetime(2026, 9, 27, 23, 59, tzinfo=IST)
PER_DAY, TOTAL = 5, 15
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _load(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def headline(m):
    v = (m or {}).get("val")
    if isinstance(v, dict) and "f05_macro" in v:
        return v["f05_macro"]
    return (m or {}).get("val_f05")


def gather():
    now = datetime.now(IST)
    left = DEADLINE - now
    recs = [r for r in (_load(p) for p in (ROOT / "submissions" / "records").glob("*.json")) if r]
    up = [r for r in recs if r.get("status") == "uploaded"]
    today = now.date().isoformat()
    used_today = sum(1 for r in up if str(r.get("uploaded_ist", ""))[:10] == today)
    runs = []
    for d in (ROOT / "experiments" / "runs").glob("*/"):
        meta, met = _load(d / "meta.json"), _load(d / "metrics.json")
        if meta:
            runs.append((meta, headline(met)))
    best = sorted([r for r in runs if r[1] is not None], key=lambda r: -r[1])[:3]
    latest = sorted(runs, key=lambda r: r[0].get("started_ist", ""))[-3:]
    best_lb = max((r.get("public_score") or 0 for r in up), default=None)
    return now, left, used_today, len(up), best, latest, best_lb, recs


def main(short=False):
    now, left, used_today, used, best, latest, best_lb, recs = gather()
    h = max(left.total_seconds(), 0) / 3600
    if short:
        b = f"{best[0][1]:.4f}" if best else "-"
        print(f"AMLC | {h:.1f}h left | subs today {used_today}/{PER_DAY}, total {used}/{TOTAL} | best val {b}"
              + (f" | best LB {best_lb:.4f}" if best_lb else ""))
        return
    print("=== Amazon ML Challenge 2026 — status ===")
    print(f"Now {now:%a %d %b %H:%M} IST | deadline 27 Sep 23:59 IST | {h:.1f} h left")
    print(f"Leaderboard submissions: today {used_today}/{PER_DAY}, total {used}/{TOTAL}"
          + (f" | best public LB {best_lb:.4f}" if best_lb else ""))
    prepared = [r for r in recs if r.get("status") == "prepared"]
    if prepared:
        print(f"Prepared but not uploaded: {', '.join(r['sub_id'] for r in prepared[-3:])}")
    print("Best val runs:" if best else "Best val runs: none yet")
    for m, s in best:
        print(f"  {s:.4f}  {m['run_id']}  — {m.get('hypothesis', '')[:70]}")
    if latest:
        print("Latest runs:")
        for m, s in latest:
            print(f"  [{m.get('status')}] {m['run_id']}" + (f"  {s:.4f}" if s is not None else ""))
    if h < 6:
        print("!! < 6h left: freeze features, only threshold/ensemble tweaks, prepare final zip (/package-final)")


if __name__ == "__main__":
    main(short="--short" in sys.argv)
