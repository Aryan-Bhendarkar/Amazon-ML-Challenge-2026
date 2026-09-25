# Experiments

- One folder per run in `runs/<YYYYMMDD-HHMM>_<author>_<slug>/`, created by `ber.tracking.Run`:
  - `meta.json`
  - `metrics.json`
  - `notes.md`
  - optionally `error_analysis.md`

  Commit these folders.
- Large outputs live in `artifacts/<run_id>/` (git-ignored): predictions, candidates, models, `val_entity_scores.parquet`.
- `python scripts/leaderboard.py` regenerates `LEADERBOARD.md`, ranked by val macro F0.5.
- `python scripts/compare_runs.py <a> <b>` runs a paired bootstrap on identical eval entities.
