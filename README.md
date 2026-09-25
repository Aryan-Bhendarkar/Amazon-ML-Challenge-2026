# Amazon ML Challenge 2026: Business Entity Resolution

This is the team repo. Start with **`CLAUDE.md`**, which covers the task, the hard rules, the workflow and the repo map. The details are in `docs/`.

## Quick start (every teammate, every machine)
```bash
# 0) put the competition folder at ./student_resource  (or set AMLC_RAW_DIR)
python -m venv .venv --system-site-packages        # reuses a system torch+CUDA if present
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/Mac: .venv/bin/python
.venv/Scripts/python scripts/check_env.py
.venv/Scripts/python scripts/prepare_data.py       # TSV -> parquet, GT pairs, folds, samples (~3 min)
.venv/Scripts/python scripts/build_norm_cache.py   # normalized name/address cache (~5 min)
.venv/Scripts/python -m pytest -q                   # library tests
.venv/Scripts/python pipelines/_template.py --subset micro   # plumbing test -> a tracked run
claude                                              # then: /status
```

## Daily loop
- `/status`: time left, submission budget, best runs
- `/experiment <idea>`: a tracked run (`experiments/runs/…`) → `python scripts/leaderboard.py`
- `/error-analysis <run_id>` and `/blocking-audit <run_id>`: what to fix next
- `/submit <run_id>`: prepare + validate LB files (a human uploads, then runs `scripts/record_lb.py`)
- `/package-final <sub_id> <team>`: the final zip on Day 3

## Layout
See the repo map in `CLAUDE.md`.
