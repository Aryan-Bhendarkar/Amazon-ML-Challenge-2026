---
name: evaluate
description: Evaluate validation predictions or a run with the official macro F0.5 metric — per-country and per-match-count breakdowns, pair precision/recall, singleton accuracy, threshold curve, paired bootstrap vs the best run, and the leave-one-country-out France proxy. Use after any run or when asked "is this better?".
argument-hint: "<run_id | path to preds parquet/tsv> [subset=mini]"
---

# Evaluate: $ARGUMENTS

1. Score it:
   - If given a run_id: read `experiments/runs/<run_id>/metrics.json`.
   - If given a file: `python scripts/score.py <file> --subset mini`.
2. Compare to the best run: `python scripts/leaderboard.py --top 5`, then `python scripts/compare_runs.py <best> <this>`. Report delta ± CI.
3. Read the breakdowns and say what they imply:
   - `f05_by_country` (US vs India gap)
   - `f05_by_ntrue` (singletons "0" and large clusters)
   - `singleton_acc`
   - `empty_pred_on_nonsingleton`: each one scores 0, so this is usually the cheapest gain
   - `entities_with_fp`
   - pair P/R, and `val_expected_f05_rule` vs the threshold rule
4. Threshold sanity: open `artifacts/<run_id>/threshold_curve.csv`. A flat optimum is robust. A sharp peak means calibration is fragile, so prefer a slightly higher (precision-side) threshold.
5. France proxy: if the run trains a model, is there a LOCO number (train on one country, eval the other)? If not, flag it for model/feature changes.
6. Verdict: **keep / kill / needs-confirmation (fold0)**, with a one-line reason. Append the verdict to the run's `notes.md`.

Metric reminders:
- Missing S1 = empty prediction.
- Singleton + any prediction = 0.
- Non-singleton + empty prediction = 0.
- Macro over ALL entities.
