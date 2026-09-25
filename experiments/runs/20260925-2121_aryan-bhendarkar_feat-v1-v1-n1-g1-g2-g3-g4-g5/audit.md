# Validation audit: 20260925-2121_aryan-bhendarkar_feat-v1-v1-n1-g1-g2-g3-g4-g5

Auditor: validation-auditor, 2026-09-26 (IST). Started fresh; earlier partial audits are superseded.
Claim: mini 0.9611 → 0.9800 (+0.0189) over the gate run 20260925-2015_aryan-bhendarkar_gate-blocking-v1-n1.
Reproduction scripts for every number below are in `audit_scripts/`. All ran on 15k mini S1 (sample seed 7), 2 threads, with a frozen model.

## Verdict: GO-WITH-CONDITIONS

There is no label leakage and no val/test code mismatch. The gain is confirmed on the disjoint fold0x (+0.0189).
However, the gain on **test** will be smaller than on val:
- Two features groups depend on density: the idf features and the S1-count features. Test has 0.5× (US) and 0.2× (France) the train S1 density.
- Val has no cross-S1 competition.

Realistic expected test gain over the parent is about **+0.010 to +0.015**. That is still clearly positive in every stress test.

### Conditions before upload
1. sub3 finishes. `make_submission.py` must print validator **PASS** (with `--check-ids`) and **no `WARNING` lines**: no duplicate match_id, and no matched pairs outside the candidates. Both checks only warn; they do not fail.
2. Full-test per-country diagnostics must be within these bounds. The reference points are the parent's full test (France 5.48% / 3.29, India 6.17% / 3.18, US 5.97% / 3.24) and the new model's val (India 6.10% / 3.29, US 5.81% / 3.33).

   | Country | Empty rate | Mean matches |
   |---|---|---|
   | India, US | 5.5–6.6% | 3.2–3.45 |
   | France | ≤ 8% | ≥ 2.9 |

   If France falls outside its range, eyeball 12 France S1 where the new model dropped the parent's matches before uploading.
3. Correct notes.md. The density check there (−0.002 at 50%) understates the shift; see B2. Record the expected test gain as +0.010 to +0.015.

### Fix after this submission (highest-value follow-up)
- **Density-invariant token rarity (B1).** Replace `ex_/mi_idf_*` (= log(n_s1/df), where an unseen token gets log(n_s1)) with:
  - `log1p(df)` capped: raw S1 document frequency, which is invariant for rare tokens under subsampling
  - explicit `ex_unseen_n` / `mi_unseen_n` counts (tokens with df = 0 in S1)

  Retrain. Then accept the change only if the density simulation (`audit_scripts/dens.py`, r = 0.5 and 0.2) loses ≤ 0.002 while mini holds.
- Optionally make the S1-count features relative too, e.g. count / E[count] per country. A cheaper option is to keep only the `_eq1` flags and cap at 5.

## Checklist

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Train only on folds ≠ 0; no fold-0 labels | PASS | The train cache has roles train = folds 2,3,4 (150k) and es = fold 1 (30k). train∩mini = 0 and fold0x∩(mini ∪ train) = 0 (checked). `build_cache.py:29-37`. The labels are attached after featurization (`build_cache.py:90`). ctx features read only norm parquet files (`ctx_features.py:85-98`). The `label` column is never passed to `add_features` (`features_v1.py:71`). |
| 2 | Eval candidates from the FULL country pool | PASS | `blocking_v1.run_country`: pool = the whole country pool. TfViews are fit on all S1 + pool. The key-join caps are pool-side. `do_rev=False`. No retriever depends on the query set, so candidate lists (and hence rank/G5 features) are identical for mini-only and all-test queries. |
| 3 | Threshold knobs / disjoint confirmation | PASS | One knob, t = 0.75. The curve is flat within 1e-4 on 0.70–0.80. **fold0x frozen t: 0.97989** (India 0.9748, US 0.9833); the oracle t on fold0x = 0.725 gives the same F. Paired delta vs parent +0.01887, CI [0.01857, 0.01917] (run 20260926-0109 confirm-fold0x-v1-n1). |
| 4 | Early stopping not on fold 0 | PASS | `is_es` = role 'es' = fold 1 (`features_v1.py:51-52`, 94-102). |
| 5 | Metric over ALL eval S1 | PASS | `harness.log_predictions` → `metric.macro_f05` over `ctx.truth`; missing = empty. The mini cache covers all 87,952 S1. |
| 6 | Test = validated pipeline | PASS / WARN | I recomputed the ctx features for 15k mini S1 with the current code and compared them with the cached `ctx1_mini.parquet`: **0 mismatched columns**. `ctx_features.py` has been unchanged since 10:04 UTC (commit 6f9769a); the model was trained at 16:17. Both val and test use norm_v0 for ctx and norm_v1 for base/test.parquet (v1_test default `--norm-v 1`, the same featurizer as build_cache). The model has `pandas_categorical: []` (integer codes). WARN: density-dependent features, see B1/B2. |
| 7 | assign_best_s1 before the decision | PASS | `predict_test_v1.py:85`. PMIN = 0.01 truncation is harmless: argmax per record is unchanged for any record that can reach t = 0.75, and `threshold_matches` is a pure threshold. confirm_v1 applies the same 0.01 filter. |
| 8 | Candidates = exact scored set; matches ⊆ candidates | PASS | Every row of every row group of `test.parquet` is scored; PMIN applies only after predict. sub3 passes `--candidates data/cands/v1_n1/test.parquet`. The row-group assertion (`predict_test_v1.py:55`) is correct: an S1 split across batches would fail. `write_candidates_streaming` checks the same thing independently. |
| 9 | Pretrained models | PASS | None. LightGBM (MIT), rapidfuzz (MIT), polars (MIT). |
| 10 | No external data | PASS | Grep of src/ber and the pipelines found no requests, urllib, geocoding, http or downloads. |
| 11 | Country open set | PASS / WARN | `country` is only a partition/join key; it is not among the 83 features (checked `features.json`). Countries come from the data (`B.countries_of`). The normalize state rules fall back to "". WARN: `BIZ_WORDS` includes `india france america` (see B6). |
| 12 | Output via write_id_lists, validator PASS | PENDING | The code path is correct (`make_submission.py:36-47`, LF). Validator output isn't available yet: sub3 was at 1.19M / 1.73M S1 at 10,231 s, with about 1.3 h left. |
| 13 | Test diagnostics plausible | WARN (pending) | See B3. The parent's full test matches its val per country, France included. |
| 14 | Gain plausibility (+0.0189) | PASS | Gain composition:<br>- FP entities 3.87% → 1.23%<br>- fired records owned by nobody (distractors): 1,923 → 562<br>- fired records owned by non-query S1: 1,768 → 587<br>It holds on fold0x (+0.0189) and in LOCO at the source t: India→US +0.011, US→India +0.020 (vs base ref run 20260925-2211). It is a precision effect from typed edits and uniqueness, not leakage. |

## Answers to the six questions

1. **Label leakage: none.** SplitContext statistics are unsupervised counts over all S1 and all S2+S3 of the split. The eval S1 and their own copies are included, exactly as at test (hygiene #3 allows this). The cache labels are attached after featurization and are never read by ctx_features. The only GT-dependent structure is which S1 are in the train/es/mini role sets, and that only selects rows.
2. **Val/test consistency: identical code.** Nothing depends on the query set:
   - Candidate lists are query-independent.
   - G5 and the rank features are per-S1 over complete lists; completeness is asserted and hash-chunked.
   - G1/G2 counts are over all S1 of the split, not over the query set.

   The inconsistencies are in the *data*, not the code: density (B1/B2) and competition (B4).
3. **Distribution shift: the team's density simulation is inadequate.** It recomputed only the S1 counts and left the idf and pool counts at train density. The measured shift (`audit_scripts/ctxshift.py`):

   | Country | Test S1 vs train | Test idf_max | Train idf_max |
   |---|---|---|---|
   | US | 0.50× | 13.40 | 14.10 |
   | France | 0.20× vs US train | 12.47 | 14.10 (US) |
   | India | 0.92× | about equal | about equal |

   - Pool per S1 is **5.8 in test vs 4.7 in train** for every country, i.e. about 2× the unmatched records per S1. The pool-count features proved insensitive to this (0 loss).
   - France: `addr_uniq` 0.81 vs 0.93, `addr>=5` 4.8% vs 1.7–2.7%, `ex_idf_min` 4.1 vs 5.4–5.8, `pool_addr_c` 2×. This is out of the training range.

   Full simulation in B2.
4. **Open set: PASS.** Country is only a partition key. BIZ_WORDS is a hand-written list of general words, so it is allowed under the rules. It must be listed in the methodology doc, and the country-name entries should be reconsidered (B6).
5. **Threshold: PASS.** The fold0x confirmation arrived: frozen t = 0.75 gives 0.97989, and the fold0x-optimal 0.725 gives the same F. Caveat B4.
6. **predict_test_v1: PASS** on the assertion, the PMIN truncation and the exact candidate set; see items 7 and 8.

## Issues, ranked

**B1 [HIGH, fix after this submission]: idf features are not density-invariant.**
- `idf = log(n_s1/df)` per country, and an unseen token gets `idf_max = log(n_s1)` (`ctx_features.py:96-104`, 160-162).
- The model keys on the exact value as an "unseen token" flag. At full density, a uniform ±0.2 nat offset alone costs **−0.0033** (`fix.py`, NREF run).
- At test, US unseen = 13.40, which in train corresponds to a df = 2 token; France unseen = 12.47.
- Simulated idf-only shift at US-like density (r = 0.5): **−0.0045**, all of it recall (R 0.9515 → 0.9401).
- Fix: `log1p(df)` + `*_unseen_n` features, retrain, then gate on the density sim.

**B2 [MEDIUM]: the reported density check understates the drift.**
The simulation drops (1−r) of the non-query train S1 and their GT copies from all statistics, then recomputes all ctx features (counts, pool counts, idf). The frozen model scores 15k mini S1 at t = 0.75. Numbers are F0.5, with the change vs the full-density baseline (0.9808) in brackets.

| Scenario | New model | Parent (same sample) | Gain vs parent |
|---|---|---|---|
| Full density | 0.9808 | 0.9611 | +0.0197 |
| r = 0.5, candidates of dropped S1 removed ("clean") | 0.9774 (−0.0035) | 0.9635 | +0.014 |
| r = 0.5, dropped S1's copies left as orphan candidates | 0.9735 (−0.0073) | 0.9611 | +0.012 |
| r = 0.2, clean | 0.9764 (−0.0044) | 0.9651 | +0.011 |
| r = 0.2, orphans | 0.9676 (−0.0132) | 0.9611 | +0.0065 |

- The orphan loss is precision, driven by the S1 counts.
- Ablation: pool counts cost 0; the S1 counts cost −0.002 (r = 0.5) and −0.005 (r = 0.2); the rest is idf.
- Test pool per S1 is 2× the train unmatched density, so the truth likely lies between the clean and orphan rows.
- Weighted by test mix (India 47% ≈ no shift, US 38% ≈ r = 0.5, France 15% ≈ r = 0.2): expected drift ≈ −0.002 to −0.005 vs val.
- Past LB runs were val − 0.005 (0.9057 → 0.901 and 0.9107 → 0.9046).

**B3 [MEDIUM]: France behaves differently in a test sample.**
Setup (`testsamp.py`): 6k S1 per country, with competition only within the sample.

| Country | Empty rate, new vs parent | Mean matches, new vs parent | Same direction as val? |
|---|---|---|---|
| India | 6.60% vs 6.63% | 3.25 vs 3.18 | yes |
| US | 5.95% vs 6.18% | 3.37 vs 3.23 | yes |
| France | **6.07% vs 4.65%** | **3.16 vs 3.46** | no (reversed) |

The new model is *more* conservative on France. That is probably good, since the parent's no-competition France sample looks over-matched against the train prior of 5.6% empty. However, it relies on out-of-range co-location and idf features. Gate on the full-test diagnostics (condition 2).

**B4 [LOW]: no cross-S1 competition in val (recurring).**
- If the owners always won at test (the oracle, removing fired records owned by non-query S1), the gain would be **+0.0159** at fixed t, or **+0.0129** when both models retune.
- The optimal t in that oracle drops to 0.475 (+0.001).
- fold0x has the same bias. Leave t = 0.75 for this submission. For the decision layer, simulate competition before tuning t.

**B5 [LOW]: make_submission only warns** on duplicate match_id and on matches outside the candidates (`make_submission.py:29-32`, 41-43). Make both hard failures. Until then, grep the sub3 log for `WARNING`.

**B6 [LOW]: BIZ_WORDS contains `india`, `france` and `america`** (`ctx_features.py:44`).
- This is legal as a hand-written generic list, but it's asymmetric for an unseen country (e.g. "germany").
- Document it in the methodology. Consider replacing it with a generic "token equals a country value seen in the data" flag, derived from the `country` column values, not hard-coded.

**B7 [INFO]:** the team's density-simulation code was not committed. `audit_scripts/` now holds a reproducible version.
