# Validation audit (DELTA): sub 20260926-0953_aryan-bhendarkar (replaces un-uploaded sub3)

Run: 20260926-0710_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3 (v1_n2 cache = v1_n1 + `nkey_num`; ctx v3).
Base design audited in `20260925-2121_aryan-bhendarkar_feat-v1-v1-n1-g1-g2-g3-g4-g5/audit.md` (GO-with-conditions).
Auditor: validation-auditor, 2026-09-26. The checks ran with 2 threads and no training. Scripts are in `audit_scripts/chk1..7.py`.

## Verdict: GO-WITH-CONDITIONS (replace sub3 with this submission)

I found no leakage and no val/test mismatch. The candidate file is exactly the scored set, and every match is in it.
Expected test gain over sub3 is about **+0.0015 to +0.003**:
- India: the nkey_num gain is measured and transfers.
- US and France: ctx v2/v3 is more robust to density.

One new France risk (I1 below) takes back part of the gain but does not reverse it.

Conditions:
1. Upload this INSTEAD of sub3, not in addition. The expected gain is below LB noise, so don't spend a slot on the A/B comparison.
2. After the LB score arrives, log that the France slice of this model has an estimated ~34% FP rate on nkey_num-only matches (I1). Fix it before the next submission, since the current val cannot see it.

## Answers to the specific questions

**(a) Is nkey_num leak-free and consistent across train, mini, fold0x and test? PASS**
- `augment_nkey_num.py` applies one code path to every file. Query sets are:
  - train_ids (180k: roles train = folds 2–4, es = fold 1)
  - `split.eval_ids` for mini and fold0x
  - all S1 for test
- The pool is always the full split pool of the country. The block cap (50) counts pool records only. So a query's pairs do not depend on which other S1 are in the query set, and nothing is query-set dependent.
- Labels are attached after featurization (`augment_nkey_num.py:113`). The retriever uses no labels.
- `rbits == 512` appears in every file, and P(label | 512) is the same in train, mini and fold0x (chk1):

  | File | 512 pairs | P(label) |
  |---|---|---|
  | train | 196k (train 163k / es 33k) | 0.0106 |
  | mini | 96k | 0.0097 |
  | fold0x | 385k | 0.0107 |

  512 is never OR'd with other bits: 512 means "nkey_num only". In all four files, cos_name is null and kmask = 0 on exactly those rows. cos_name is not a model feature.
- Rank features (`name_rank_in_s1`, `addr_rank_in_s1`, `name_gap_s1`, `n_cand_s1`) were recomputed from scratch per S1 on v1_n1 and v1_n2, for train, mini, fold0x and all 164.7M test rows. **0 mismatches** (chk1, chk2). The definitions match the original v1_n1 values, so recomputing only the touched row groups is safe.
- Test cache integrity (chk2): 45 row groups, 0 S1 spanning row groups, 0 duplicate pairs, all 1,732,544 S1 present.
- New pairs per S1 are consistent between train/val and test:

  | Country | Train/val | Test |
  |---|---|---|
  | India | 2.62 | 2.72 |
  | US | 0.07 | 0.04 |
  | France | n/a | 0.76 (no reference) |

**(b) Does ctx v2/v3 close B1, and is anything else density- or norm-version-dependent? PASS / WARN**
- B1 (idf) is closed:
  - `ex_/mi_idf_*` were replaced by `*_unseen_n`, `*_ldf_*` (capped raw df) and `*_lfrac_c*` (df/n_s1 for df ≥ 50 only) (`ctx_features.py:138-156`).
  - The clean density sim at r = 0.2 gives −0.0008 vs v1's −0.0046, and at r = 0.5 −0.0022 vs −0.0033.
- Norm version is consistent everywhere via `norm_of(3) = 1`:
  - train/mini: `features_v1.py:79, 166`
  - fold0x: `confirm_v1.py:33-45`, meta `ctx_ver: 3`
  - test: `predict_test_v1.py:59-86`, `SplitContext.build("test", 3)` + `attach_norm(..., norm_of(cv))`; the log prints "ctx_features version 3 (norm_v1)"
- The ctx caches are per version (`ctx3_*`). The v1_n2 ctx3 files (01:39–02:04 UTC) are newer than the v1_n2 caches (01:06–01:14) and the last ctx_features change (01:03). The norm_v1 train and test files were built from the same token_map_v1 (10:20).
- WARN, remaining density dependence:
  - The G1/G2 S1-count features are still raw counts, capped at 20. The orphan variant of the sim at r = 0.2 is still −0.0117 (v1: −0.0132), so B2 is only partly closed.
  - The `df ≥ 50` gate on lfrac moves with density.
  - v3 itself (on norm_v1) was not density-simmed. The formula is unchanged, so no new mechanism is expected.

**(c) Is candidate_pairs.tsv exactly the scored set? PASS**
- Streamed fingerprint (chk3): candidate_pairs.tsv has 1,732,544 rows and 164,745,367 ids.
  - Its order-free 64-bit hash-sum of `s1|cand` **equals** that of `data/cands/v1_n2/test.parquet` (164,745,367 rows, 0 duplicates).
  - That parquet is the file predict_test_v1 scored: its log says "164,745,367 pairs scored".
- matching_results.tsv:
  - 5,736,331 pairs, identical in both directions to `test_matches.parquet`
  - 0 duplicate match_id, 0 duplicate S1 rows, 0 CR bytes, 0 spaces
- All 5,736,331 matches are in the candidate parquet. The header is correct and there are no CRLF lines.

**(d) Is the missing LOCO on v1_n2 acceptable? WARN (acceptable)**
- A US↔India LOCO would not have shown the actual France risk. That risk is name *genericity* (see I1), which neither train country has.
- The LOCO that exists for ctx v3 on v1_n1, vs the sub3 model (ctx v1), is mixed and within noise: India→US 0.9587 → 0.9552 (−0.0035); US→India 0.9237 → 0.9279 (+0.0042).
- I replaced the missing LOCO with direct France test inspection (I1). It is more informative.

**(e) Any reason not to replace sub3? No blocker.**
- Test diagnostics are within the bounds set in the previous audit:

  | Country | Empty rate, sub3 → this | Mean matches, sub3 → this |
  |---|---|---|
  | France | 6.30% → 6.14% | 3.16 → 3.18 |
  | India | 6.18% → 6.03% | 3.26 → 3.27 |
  | US | 5.70% → 5.68% | 3.40 → 3.41 |

- The fold0x confirmation is on a disjoint set with a frozen t: +0.00128, CI [0.00114, 0.00142].
- The code is unchanged since the run. Only `train_eval.py` is dirty, and it is not used here.

## Issues, ranked

**I1 [MEDIUM, France; follow-up, not blocking]: nkey_num surfaces generic-name collisions in France, and the model accepts about a third of them.**
- France nkey_num-only pairs fire 8× more often than in India (chk4, chk5):

  | Country | nkey_num pairs | Matches | Firing rate |
  |---|---|---|---|
  | France | 197,578 | 14,696 | 7.4% |
  | India | 2.2M | 19,526 | 0.9% (val India: 0.85%, precision 94.8%) |

- Many of the France fires are the same generic name + the same house number + a **different street**, for example:
  - "Tourcoing Comite SAS, 8 Rue du Général Dampierre" ↔ "8 R DE VASSY"
  - "Mérignac Club SARL, 2 Rue du Cap Vert" ↔ "02 RUE DE LA FOUGERAIE", at p = 0.99
- The street heuristic in chk6 compares distinctive street tokens: rare tokens only, city and region excluded, token_set_ratio < 50.

  | Pair set | Different street |
  |---|---|
  | France nkey_num fires | **35.7%** |
  | Other France fires | 0.5% |
  | US GT positives | 1.0% |

  I eyeballed a random sample of 20 France nkey_num fires: 6 are clear different-street FPs (30%). All 10 low-street samples are FPs.
  The heuristic cannot judge India, because its native-script addresses make 76% of India val TPs look like "different street".
- Competition: for 1,187 of the ~4,985 suspected FPs, another S1 also scored the record ≥ t and lost it through assign_best_s1. For 583 of those, that S1's street matches the record (street sim ≥ 90). This is the recurring "val has no competition" blind spot.
- Estimated France effect of the nkey_num matches:
  - heuristic truth: −0.0004 on the France slice, plus about −0.0004 from stolen records
  - if all were correct: +0.0077 on France

  Overall that is ≈ −0.0001 to +0.001. So France is roughly neutral, and the gain comes from India plus the ctx v3 density fix.
- Why: in US/India a name + number match almost always means a true match. French names in test are templated ("<City> Club/Amis/Pharmacie <legal form>"), so same name + same small number collide across streets.
- Follow-up fixes, in order:
  1. A cross-S1 competition feature: for each candidate, the best street/address similarity of any *other* S1 sharing its name key (reverse lookup; unsupervised, same code for test).
  2. A block-size feature for the nkey_num key: pool and S1 records sharing (name key, number), as `_eq1`-style flags.
  3. A decision-level tie-break: when two S1 both score a record ≥ t, prefer the higher addr_tset. Validate it with the competition bracket (`20260925-2121_*/audit_scripts/comp.py`).

**I2 [LOW-MEDIUM]: B2 (S1-count density drift) is still partly open.** The orphan sim at r = 0.2 is −0.0117 for v2. Expect val→LB ≈ −0.005 as before (see LB history). The expected LB for this sub is roughly 0.975–0.977, depending on France.

**I3 [LOW]: the n_cand_s1 feature can now exceed 100.** The share of rows at > 100 is 7.2% in train. On test it is 17% in India, 6% in France and 0.3% in US. The range is covered by training, so no action is needed.

**I4 [LOW, carried over]:**
- `make_submission.py:29-32, 41-43` still only WARNs on duplicate match_id and on matches outside the candidates (B5). I verified both at 0 directly.
- BIZ_WORDS still contains `india/france/america` (B6).

## Checklist (delta items only; the rest are unchanged from the parent audit)

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Train only on folds ≠ 0 | PASS | train cache roles train/es only, 180k S1 (chk1). `features_v1` asserts nothing new; the disjointness was verified in the parent audit and the S1 set is unchanged |
| 2 | Eval candidates come from the full country pool | PASS | `augment_nkey_num.py:52-54` `pool_table(split, ctry)` |
| 3 | Knobs tuned on val and confirmed on a disjoint set | PASS | 1 knob (t = 0.775); fold0x frozen +0.00128 (optimal t there is 0.75, +0.00006) |
| 4 | Early stopping without fold 0 | PASS | es = fold 1 |
| 5 | Metric over all eval S1 | PASS | harness; n = 87,952 / 353,503 |
| 6 | Test = validated pipeline | PASS | same cache procedure, ctx_ver 3/norm_v1, model + t from the run; 0 rank mismatches |
| 7 | assign_best_s1 applied | PASS | `predict_test_v1.py:129`; 0 duplicate match_id in the TSV |
| 8 | candidates = scored set; matches ⊆ candidates | PASS | hash-sum equality; 5,736,331/5,736,331 |
| 9 | Models | PASS | LightGBM only |
| 10 | No external data | PASS | grep clean |
| 11 | No country lists or features | PASS | the countries loop comes from the data; `country` is not in features.json |
| 12 | Writer, LF, validator | PASS | 0 CR, record validator_pass true |
| 13 | Diagnostics plausible | PASS | see (e) |
| 14 | Gain plausible | PASS | recall +0.0031 → realized +0.0013 (India +0.0029, US flat); consistent |
