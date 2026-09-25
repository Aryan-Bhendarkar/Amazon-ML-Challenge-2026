# Validation audit: submission 20260925-1533_aryan-bhendarkar (LB #2)

Run 20260925-1531_aryan-bhendarkar_postrules-v0. It takes the probs of parent 20260925-1236_aryan_baseline-v0-keys-lgbm and applies the S1-context post-rules (ruleA add, dropA remove, ruleB add).
Auditor: validation-auditor agent, 2026-09-25. No code or data was modified. The checks ran from scratchpad scripts, limited to 2 cores.

## Verdict: **GO**. Two WARNs should be fixed in the next iteration; neither blocks this submission.
The expected LB gain over #1 is about +0.003 to +0.005. That is lower than the +0.0050 measured on val, for the two reasons in the WARNs below.

## Checklist

| # | item | status | evidence |
|---|---|---|---|
| 1 | Train only on folds != 0 | PASS | pipelines/baseline_v0.py:320-321. Train is folds >= 2 and early stopping is fold 1. GT is loaded only for tr/es ids (:327). The rules themselves use no labels. The S1 key counts are unlabeled and transductive (src/ber/postrules.py:43-47). |
| 2 | Eval candidates come from the full country pool | PASS | baseline_v0.py:270 `pool_table(split, ctry)` returns the whole S2/S3 pool. The queries are mini S1 only. |
| 3 | Knobs tuned on val and confirmed on a disjoint set | WARN | About 4 discrete choices were picked on mini (ruleA exact vs tset, B name<50, dropA >= 2, dropB rejected); t=0.675 is inherited. A fold0\mini confirmation is **not possible**: the parent val_pred.parquet covers only mini (86,780 S1 with preds). Proxy: 20 random mini split-halves all fall in +0.0045 to +0.0055, and the per-country deltas are India +0.0050 / US +0.0050. Selection noise is negligible (paired SE ~0.00017). The real risk is protocol mismatch, covered in the WARNs. |
| 4 | Early stopping not on fold 0 | PASS | baseline_v0.py:321,345 (fold 1). |
| 5 | Metric over all eval S1 | PASS | metric.per_entity_scores iterates over the truth of all 87,952 mini S1. Re-computed at floor 0.05: base 0.90572, rules 0.91072. |
| 6 | Test uses the same pipeline as val | PASS | The same model/probs, floor 0.05 (the test_pred minimum is 0.0500), assign_best_s1, t=0.675 and rules code. Key counts use all S1 of the split, as on val. |
| 7 | At most one S1 per record | PASS | 0 duplicate match_id in test_matches.parquet. 5,078,526 IDs, all unique in the TSV. |
| 8 | matches ⊆ candidates, candidates = scored set | PASS | candidate_pairs.tsv is byte-identical to submission #1 (cmp). The record shows matched_pairs_in_candidates = total = 5,078,526. The rules only re-decide pairs that are already in test_pred. |
| 9 | Model licenses | PASS | No pretrained weights. The libraries are LightGBM (MIT), rapidfuzz (MIT) and polars (MIT). |
| 10 | No external data | PASS | No requests/urllib/geocoding in postrules, baseline_v0, normalize or decision. |
| 11 | Country open set | PASS | Country is used only as a group_by/join key taken from the data (postrules.py:45-46, 53-54). countries_of() reads it from the data (baseline_v0.py:90). There are no hard-coded {US, India} and no country feature. France receives rule firings. |
| 12 | Writer and validator | PASS | Checked by re-reading the TSV: 0 CR, 0 spaces, exact header, 1,732,544 rows in test_source1 order, TSV = parquet pairs. The official validator was re-run with `--check-ids`: "PASS — no blocking issues found" (163,499 empty rows, 1,569,045 non-empty). |
| 13 | Test diagnostics | PASS | Empty rate goes from 0.0954 to 0.0944, mean matches from 2.902 to 2.931. Per country (FR/IN/US), empty rate 0.061/0.121/0.077 becomes 0.060/0.120/0.076. The empty rate stays above the 5.6% prior (blocking misses, as in #1). |
| 14 | Gain is plausible | PASS | +0.0050 against +0.0044 in the error analysis without the floor. The added pairs are 97% (A) and 98% (B) true on val. The mechanism is clear (name-only records) and no labels are used. |

## WARN 1: dropA gain is overstated on val and could be about 0 or slightly negative on test
- On val, assign_best_s1 sees only mini S1s, so name-only records owned by non-mini same-name siblings land on the mini S1. dropA then removes them: 1,608 drops, only 44.7% true.
- On test, all S1 compete. The test drop rate per S1 is **0.91%, against 1.83% on val**, which fits that mechanism. Test drops are also concentrated in s1_same_name == 2 (FR 72%, US 58%, IN 50% of drops, against 34% on val). That is the riskiest group: 60% true on val even without competition.
- The val break-even true share is 66% (per false drop +0.238, per true drop −0.121). Applying the val per-group true shares to the test mix gives about 53–56% true, and full competition pushes the real share higher.
- Estimated test effect of dropA: −0.0005 to +0.0009, against +0.0014 on val (no-dropA variant: 0.90931).

## WARN 2: France ruleB firings look distribution-shifted
- The per-S1 rate of ruleB adds is FR 2.0% (5,212), IN 1.4%, US 0.95%, and 1.16% on val. France fires most even though France address uniqueness is lower (0.81 against 0.93–0.95).
- Sample of France ruleB adds: the candidate names are real, generic French names at the same address ("amicale robert", "bordeaux club", "college pierre", "comice patrimoine", "latelier services"), mostly with prob 0.06–0.2.
- The US adds instead show the synthetic random-name positive pattern the rule was built for ("jaxavi", "lumavi", "quoarc"), with prob 0.4–0.6. The val B adds have a median prob of 0.515.
- 444 France S1s receive 2 or more records through ruleB. Precision on France is unknown.
- Bounded impact: 5,212 adds / 1.73M S1. If only 20–50% are true, that is about −0.0002 to −0.0005 on the full LB.
- Cheap hedge, measured on mini: require prob >= 0.2 for ruleB. Val drops only 0.91072 to 0.91064 (0.3 gives 0.91054), and it would remove most of the low-prob France adds. It was not applied to this submission.

## Rule counts vs val (per S1)
| | adds (A+B) | ruleA | ruleB (excl. A) | dropA |
|---|---:|---:|---:|---:|
| val mini (88k) | 3.60% | 2.44% | 1.16% | 1.83% |
| test France (259k) | 4.31% | 2.30% | 2.01% | 0.76% |
| test India (810k) | 3.74% | 2.34% | 1.39% | 0.86% |
| test US (663k) | 3.74% | 2.79% | 0.95% | 1.02% |

S1 name-key uniqueness is FR 0.475 / IN 0.463 / US 0.599 on test, against IN 0.456 / US 0.520 on train. Candidate empty-address share is FR 3.0%, in line with the others.

## Minimal fix list
- None blocking for this submission.
- Hygiene: commit submissions/records/20260925-1533_aryan-bhendarkar.json. The run meta says git_dirty=true at 28b7bb7, but the rules code is committed in bee306c and is unchanged since.
- Next iteration (EXP-016 / EXP-017):
  - Move the rules into model features.
  - Measure dropA under competition-aware val (EXP-017).
  - Gate ruleB with a prob floor (>= 0.2) or a France-safe condition.
  - Produce val_pred for fold0\mini so future decision rules can be confirmed on a disjoint set.
