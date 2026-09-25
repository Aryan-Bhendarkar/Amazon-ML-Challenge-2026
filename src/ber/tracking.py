"""Minimal, git-friendly experiment tracking (no server, merge-conflict free).

Each run = its own folder  experiments/runs/<run_id>/  (committed, small):
    meta.json     who/when/git commit/params/hypothesis/status
    metrics.json  everything logged via run.log(...)
    notes.md      free text: what changed, what we learned, next step
Large outputs go to artifacts/<run_id>/ (git-ignored) via run.art_dir.

Usage:
    from ber.tracking import Run
    with Run("tfidf-blocking", hypothesis="char-3gram tfidf top50 reaches 99% recall",
             params={"k": 50, "subset": "mini"}, tags=["blocking"]) as run:
        ...
        run.log(blocking=blocking_report(...), val=metric.report(...))
        run.note("recall 0.987; misses are native-script names -> try transliteration map")
Then: python scripts/leaderboard.py
"""
from __future__ import annotations

import getpass
import json
import os
import re
import subprocess
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

from . import paths

IST = timezone(timedelta(hours=5, minutes=30))


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=paths.ROOT,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def author() -> str:
    return (os.environ.get("AMLC_AUTHOR") or _git("config", "user.name") or getpass.getuser()).split()[0].lower()


class Run:
    def __init__(self, name: str, hypothesis: str = "", params: dict | None = None,
                 tags: list[str] | None = None, parent: str | None = None):
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]
        self.run_id = f"{datetime.now(IST):%Y%m%d-%H%M}_{author()}_{slug}"
        self.dir = paths.EXP_DIR / self.run_id
        self.art_dir = paths.ART_DIR / self.run_id
        self.meta = {
            "run_id": self.run_id, "name": name, "author": author(), "hypothesis": hypothesis,
            "params": params or {}, "tags": tags or [], "parent": parent,
            "git_commit": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "started_ist": datetime.now(IST).isoformat(timespec="seconds"), "status": "running",
        }
        self.metrics: dict = {}

    def __enter__(self) -> "Run":
        self.dir.mkdir(parents=True, exist_ok=True)
        self.art_dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()
        self._dump()
        notes = self.dir / "notes.md"
        if not notes.exists():
            notes.write_text(f"# {self.run_id}\n\n**Hypothesis:** {self.meta['hypothesis']}\n\n", encoding="utf-8")
        print(f"[run] {self.run_id}")
        return self

    def log(self, **kv) -> None:
        self.metrics.update(kv)
        self._dump()

    def note(self, text: str) -> None:
        with open(self.dir / "notes.md", "a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n\n")

    def __exit__(self, et, ev, tb) -> bool:
        self.meta["status"] = "failed" if et else "done"
        self.meta["duration_min"] = round((time.time() - self._t0) / 60, 2)
        self.meta["finished_ist"] = datetime.now(IST).isoformat(timespec="seconds")
        if et:
            self.meta["error"] = "".join(traceback.format_exception_only(et, ev)).strip()
        self._dump()
        return False

    def _dump(self) -> None:
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str), encoding="utf-8")
        (self.dir / "metrics.json").write_text(json.dumps(self.metrics, indent=2, default=str), encoding="utf-8")


def load_runs() -> list[dict]:
    runs = []
    for d in sorted(paths.EXP_DIR.glob("*/")):
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            met = json.loads((d / "metrics.json").read_text(encoding="utf-8")) if (d / "metrics.json").exists() else {}
        except Exception:
            continue
        runs.append({**meta, "metrics": met})
    return runs


def headline(metrics: dict) -> float | None:
    """The single number runs are ranked by: val macro F0.5 (metrics['val']['f05_macro'])."""
    v = metrics.get("val")
    if isinstance(v, dict) and "f05_macro" in v:
        return float(v["f05_macro"])
    if "val_f05" in metrics:
        return float(metrics["val_f05"])
    return None
