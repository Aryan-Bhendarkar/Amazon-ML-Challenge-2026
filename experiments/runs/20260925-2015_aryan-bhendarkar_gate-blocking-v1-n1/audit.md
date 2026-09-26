# Audit: gate-blocking-v1-n1 (+0.0554 mini F0.5 vs control-keys-v0)

Auditor: validation-auditor, 2026-09-25. Checks were re-run from the caches and artifacts with 2 threads (scripts in the session scratchpad, not committed).

## Verdict: TRUSTWORTHY for the val claim. Two conditions before any LB submission (see the end).

The gain is not leakage. It is fully explained by blocking coverage, it is 90% of the oracle-ceiling gain, and it is stable across halves and thresholds.

## (a) Leakage in blocking / TF-IDF / token map / cap / features: PASS
- Token map (`scripts/build_token_map.py:24-32`, `native_pairs(min_fold=1)`):
  - I recomputed the alignment input: 441,365 pairs, folds {1: 110,317, 2: 110,171, 3: 110,139, 4: 110,738}.
  - Fold 0: 0 pairs. Mini: 0 pairs.
  - The recomputed map equals the saved `data/features/token_map_v1.json` (512 entries).
  - Counterfactual: learning the map WITH fold 0 gives the identical 512 entries. Even hypothetical leakage would have had zero effect. The vocabulary is small (671 source types), and the map holds generic words (aditia→aditya, praibhet→private).
- TF-IDF (`src/ber/blocking.py:189-205`, `TfViews`): the IDF is fit on the text of all S1 in the country plus the pool. No labels.
- Keys:
  - The keys_v0 token frequency and block caps use the pool only (`blocking.py:113-143`).
  - `key_join` caps pool blocks by size (`blocking.py:176-182`).
  - No labels.
- Cap score (`pipelines/blocking_v1.py:94-98`): cos_na, or cos_name when an address is missing. No labels.
- Reverse retrieval is NOT in v1_n1:
  - `build_cache.build_v1` calls `run_country(...)` with the default `do_rev=False`.
  - 0 pairs carry rbit 128 in either cache.
- Featurizer (`pipelines/baseline_v0.py:213-254`): label-free. Context features are grouped by S1 only, and every S1 falls within one chunk (`build_cache.py:72` filters on the query row).
- `norm_v1_train_s1` is byte-for-byte equal to v0 (the map touches only non-Latin names, and S1 are Latin).

## (b) Train/eval disjointness and feature list: PASS
- Train S1 are fold 2–4 (150,000, role 'train') plus fold 1 (30,000, role 'es'). Mini S1 overlap: 0. Train S1 with fold 0: 0. Early stopping uses fold 1.
- The eval cache holds exactly the 87,952 mini S1, all fold 0, all with at least one candidate.
- `label` equals GT for every row in both caches.
- `features.json` = 28 baseline features + `rbits`. `label`, `role`, `cos_name` and `cos_na` are not among them.

## (c) Full pool and identical cap: PASS
- Of the 3.18M distinct mini candidate records:
  - 67.5% are owned by NON-mini S1 (folds 1–4 each about 447k),
  - 23.1% are unowned,
  - 9.5% are owned by mini S1.
  So candidates come from the full country pool.
- Candidates per S1 are identical for train and mini: max 100, mean 94.2, p50 100, 64.8% vs 64.7% at the cap. `n_cand_s1` equals the actual count.

## (d) rbits / context features encoding the label: PASS
- P(label | rbits) matches between train and mini for every major pattern:
  - 32: 0.00024 vs 0.00024
  - 4: 0.0151 vs 0.0153
  - 5: 0.159 vs 0.161
  - 93: 0.9967 vs 0.9967
  - 21: 0.405 vs 0.412
- The pattern shares also match (48.9% / 23.4% / 13.2%).
- No retriever reads GT. Every retriever and the cap are per query and independent of the query-set size. That is why the train (180k queries) and mini distributions are identical.

## (e) Gain decomposition (mini, 87,952 S1; contribution = Σ delta / N)
Coverage buckets are control → v1:

| coverage, control → v1 | n | F0.5 control → v1 | contribution |
|---|---|---|---|
| none → full | 2,505 | 0.000 → 0.916 | +0.0261 |
| partial → full | 19,606 | 0.849 → 0.961 | +0.0251 |
| none → partial | 377 | 0 → 0.796 | +0.0034 |
| partial → partial | 3,322 | 0.825 → 0.878 | +0.0020 |
| full → full | 56,811 | 0.9738 → 0.9725 | -0.0009 (FN 9,686 → 10,553; FP 2,493 → 2,425) |
| coverage losses (full/partial → none/partial) | 264 | | -0.0004 |
| singletons | 4,957 | 0.9514 → 0.9514 | 0 (241 wrong in both runs; 153 are the same S1) |
| **total** | | | **+0.0554** |

- Coverage:
  - full coverage of non-singletons: 0.688 → 0.951
  - zero-coverage non-singletons: 3.61% → 0.16%
- Oracle ceiling (perfect decisions within the candidates): 0.9332 → 0.9945, a gain of +0.0613. The realized +0.0554 is 90% of the ceiling gain, which is consistent.
- Precision:
  - pair P: 0.9856 → 0.9868
  - entities with an FP: 3.72% → 3.87%
  - FP: 3,633 → 3,722 (+89), against TP +30,087
- Country: India 0.877 → 0.953, US 0.925 → 0.967. India gains more, consistent with India pair recall 0.827 → 0.975.
- Split-half delta: +0.0540 / +0.0567.

## (f) Threshold: PASS
- There is 1 knob and the curve is flat. v1 F0.5 at t:

  | t | F0.5 |
  |---|---|
  | 0.60 | 0.9602 |
  | 0.65 | 0.9606 |
  | 0.675 | 0.9608 |
  | 0.70 | 0.9611 |
  | 0.725 | 0.9611 |
  | 0.75 | 0.9607 |
  | 0.80 | 0.9600 |

- At the control's t=0.675 the gain is +0.0551 (99.5% survives). The control at 0.725 scores 0.9052.
- fold0x confirmation is not possible: there is no v1_n1 fold0x cache. With a flat 1-knob curve the risk is negligible.

## (g) Test-time parity: WARN (not verifiable yet)
- **No test path exists for v1.** `blocking_v1.run_country` and `build_cache.build_v1` are only called with split "train". Parity is structurally plausible:
  - the retrievers and the cap are per query;
  - the IDF is fit on the test S1+pool of each country (unlabeled, allowed);
  - key caps depend on the pool only;
  - countries are discovered from the data (`B.countries_of`);
  - reverse retrieval is not used.

  But the test run must call the same `run_country` + `cap_per_query(score, 100)` + `featurize` on `norm_v1_test_*`.
- **France**:
  - 0.0% non-Latin names in the test S2/S3 for France, so the token map is inert there. India non-Latin is 23.6% / 13.3% (S2/S3), about the same as train.
  - Only the blocking gain can transfer to France. `skel` strips non-[a-z] characters, so accented French tokens lose letters. This affects recall only.
  - LOCO was NOT run for this model (`loco: false`). `rbits` is a new feature whose pattern mix may shift for France. This includes tf_empty, which is 49% of pairs and fills the cap.
- Val lacks cross-S1 competition. `assign_best_s1` on mini sees only mini S1, while 67.5% of the candidate records belong to other S1. This is the same for control and v1, so it does not bias the delta, but the absolute test level will differ.
- Minor: `cap_per_query` sorts without `maintain_order`. Ties at the cap boundary (duplicate names with equal cos_name) can make the kept candidates nondeterministic run-to-run. This affects reproducibility only.

## Conditions before submitting a v1 model
1. Write the v1 test path by reusing `run_country` / `cap_per_query` / `featurize` verbatim. Diff the test candidate stats against mini: cands/S1 distribution, rbits pattern shares per country (France included), and empty-candidate rate.
2. Run `train_eval.py --loco` for v1_n1 (US→India and India→US at the source threshold). Confirm the France proxy is not worse than the control's India→US 0.9098.
