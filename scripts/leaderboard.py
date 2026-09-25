"""Rank all experiment runs by val macro F0.5 and write experiments/LEADERBOARD.md.
Usage:  python scripts/leaderboard.py [--top 25] [--tag blocking]
"""
import _bootstrap  # noqa: F401

import argparse

from ber import paths
from ber.tracking import headline, load_runs


def fmt(x, nd=4):
    return "" if x is None else f"{x:.{nd}f}"


def main(top: int, tag: str | None) -> str:
    runs = [r for r in load_runs() if not tag or tag in r.get("tags", [])]
    for r in runs:
        r["_score"] = headline(r["metrics"])
    scored = sorted([r for r in runs if r["_score"] is not None], key=lambda r: -r["_score"])
    lines = ["# Experiment leaderboard (val macro F0.5)", "",
             "| # | run_id | F0.5 | US | India | pair P | pair R | blk recall | cands | subset | status |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(scored[:top], 1):
        v = r["metrics"].get("val", {})
        byc = v.get("f05_by_country", {})
        blk = r["metrics"].get("blocking", {})
        lines.append(f"| {i} | {r['run_id']} | **{fmt(r['_score'])}** | {fmt(byc.get('US'))} | "
                     f"{fmt(byc.get('India'))} | {fmt(v.get('pair_precision'))} | {fmt(v.get('pair_recall'))} | "
                     f"{fmt(blk.get('pair_recall'))} | {fmt(blk.get('cands_mean'), 1)} | "
                     f"{r['params'].get('subset', '')} | {r['status']} |")
    other = [r for r in runs if r["_score"] is None]
    if other:
        lines += ["", "## Runs without a val score", ""]
        lines += [f"- {r['run_id']} ({r['status']}) — {r.get('hypothesis', '')[:90]}" for r in other[-15:]]
    md = "\n".join(lines) + "\n"
    (paths.ROOT / "experiments" / "LEADERBOARD.md").write_text(md, encoding="utf-8")
    return md


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    print(main(a.top, a.tag))
