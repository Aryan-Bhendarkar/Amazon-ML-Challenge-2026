---
name: experiment
description: Run ONE tracked, hypothesis-driven experiment end-to-end for the entity-resolution pipeline (blocking, normalization, features, model, or decision rule), evaluate with the standard protocol, log via ber.tracking.Run, compare to the best run, and record learnings. Use whenever trying any idea that could change the score.
argument-hint: "<hypothesis or backlog id, e.g. 'EXP-002 tfidf name blocking'>"
---

# Experiment protocol: $ARGUMENTS

Follow every step. Skipping logging makes the result useless to the team.

## 0. Frame it (2 minutes, write it down before coding)
- **Hypothesis**: what changes, and why it should help (tie it to an error category or a number).
- **Parent run**: the run_id this builds on (usually the current best; see `python scripts/leaderboard.py`).
- **Success criterion**: e.g. "mini F0.5 +0.003 with no drop in LOCO" or "blocking recall ≥ 0.99 at ≤ 40 cands".
- Load the `er-playbook` skill if the idea touches blocking, features, models, France or decisions.

## 1. Build
- Put the code in `pipelines/<slug>.py` (new approach) or extend the parent pipeline. Reusable logic goes into `src/ber/` (with a test).
- Start from `pipelines/_template.py` (the standard skeleton: load → candidates → features → model → harness).
- Use `ber.io` loaders and the normalized cache (`data/features/norm_v*_*.parquet`). Never re-read raw TSVs.
- Smoke-test on `--subset micro` first (it must finish in minutes). Then run on `mini`.
- Memory/time budget on the laptop: stay under ~12 GB RAM. Log elapsed time. If it doesn't fit, ask the `perf-engineer` agent or move it to the EC2 box (`/cloud`).

## 2. Run with tracking
```python
from ber.tracking import Run
from ber import harness
ctx = harness.EvalContext.load(subset)
with Run("<slug>", hypothesis="...", params={...,"subset": subset}, tags=["blocking"|"features"|"model"|"decision"|"france"], parent="<run_id>") as run:
    cands = ...;  harness.log_blocking(run, cands, ctx, n_pool=len(pool))
    pred = ...;   harness.log_predictions(run, pred, ctx)      # (s1_id, cand_id, prob)
    run.log(timing_min=..., n_train_pairs=..., feature_importance=top20_dict)
    run.note("result + interpretation + next step")
```
- Run artifacts (models, candidates, predictions) go in `run.art_dir`. The run record in `experiments/runs/<run_id>/` is committed.

## 3. Compare
- `python scripts/leaderboard.py` shows the ranking.
- `python scripts/compare_runs.py <parent> <this>` runs a paired bootstrap on the same entities. **Keep only if p_not_better < 0.05** (or the gain is structurally obvious).
- Check the per-country and `f05_by_ntrue` breakdowns. A gain that comes only from one bucket may hide a regression elsewhere.
- For features/models: also check the leave-one-country-out proxy (train US → eval India). It must not drop.

## 4. Record
- `run.note(...)` must contain: the result vs parent (Δ F0.5, P, R, blocking recall), why you think it happened, and the next step.
- Update `docs/ideas_backlog.md` (status + run_id + Δ). If it changes how we work, append to `docs/decisions.md`.
- Commit: `git add -A && git commit -m "<run_id>: <what> (mini F0.5 x.xxxx)"`.

## Anti-patterns (don't)
- Tuning on the same entities you report without noting it. Several knobs tuned on `mini` → confirm on `fold0` minus `mini`.
- Using `country` as a feature, or filtering test by country lists.
- Training on fold-0 S1 entities, or using GT-derived statistics computed on fold 0 (e.g. IDF weights from positives) in features.
- Changing two things at once and crediting one.
- Reporting a number without the run record.
