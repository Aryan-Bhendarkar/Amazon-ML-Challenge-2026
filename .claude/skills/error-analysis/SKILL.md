---
name: error-analysis
description: Deep-dive the false positives and false negatives of a validation run, categorize them into a taxonomy with counts and example pairs, estimate the score each bucket costs, and propose ranked fixes. Use after every meaningful run and before tuning anything.
argument-hint: "<run_id>"
context: fork
agent: error-analyst
---

Analyse the errors of run **$ARGUMENTS**.

Inputs:
- `artifacts/$ARGUMENTS/val_pred.parquet`: (s1_id, cand_id, prob)
- `artifacts/$ARGUMENTS/val_entity_scores.parquet`: per-entity tp/fp/fn/f05
- `experiments/runs/$ARGUMENTS/metrics.json`: includes `val.threshold`
- Raw and normalized records: `ber.io.load_source`, `data/features/norm_v*_train_s*.parquet`
- GT: `ber.io.load_gt_pairs()`

Procedure:
1. Rebuild the final matches: `assign_best_s1` + threshold from metrics. Label each prediction TP/FP. Collect FNs, split into "in candidates but scored low" and "not in candidates" (a blocking miss).
2. Score impact: for each error type, compute the F0.5 gained if fixed. Recompute per-entity F0.5 with the error removed. Rank by total macro-F0.5 impact, not by count.
3. Sample about 40 FPs and about 40 FNs (stratified by country and prob band). Print both records raw and normalized, with the prob and the rank within the S1.
4. Build the taxonomy with counts. Common buckets:
   - FP: extra business word, house number ±k, same address different business, generic name, native-script collision, record owned by another S1 (competition)
   - FN: native script, domain/handle, missing address, heavy typo, number noise (407→07), alias
   - Singleton FPs are costly (each one costs 1.0)
5. For the top 3 buckets, propose a concrete fix: a feature, a normalization rule, a retriever, or a decision rule, with its expected gain and cost.
6. Write `experiments/runs/$ARGUMENTS/error_analysis.md`. Append the top fixes to `docs/ideas_backlog.md`.

Return a 10-line summary with the top 3 buckets, their estimated F0.5 cost, and the recommended next experiment.
