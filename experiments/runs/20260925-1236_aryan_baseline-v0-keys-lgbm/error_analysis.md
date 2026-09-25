# Error analysis — 20260925-1236_aryan_baseline-v0-keys-lgbm (mini, 87,952 S1, t=0.675)

Rebuilt from `val_pred.parquet` with `assign_best_s1` + t=0.675 → F0.5 **0.905722** (exactly the logged value).
All gains below are **macro-F0.5 deltas on mini**, computed by recomputing per-entity F0.5 with the bucket's
errors removed (FP) or added (FN). Analysis ran on 2 cores alongside the test job; scripts in the session scratchpad
(rule definitions reproduced exactly in §5 so they can be re-implemented).

## 1. Loss decomposition (total loss = 1 − 0.9057 = 0.0943)

| error source | pairs | entities hit | macro gain if fixed | India | US |
|---|---:|---:|---:|---:|---:|
| **Blocking miss** (true match not in candidates) | 42,456 | 25,936 | **0.0704** | 0.0381 | 0.0323 |
| FN in candidates, scored < t | 12,669 | 11,152 | 0.0164 | 0.0082 | 0.0082 |
| FN lost to another mini S1 in `assign_best_s1` | 416 | 405 | 0.0007 | 0.0005 | 0.0002 |
| FP (all) | 3,633 | 3,201 | 0.0105 | 0.0054 | 0.0051 |
| ↳ FP on true singletons (each costs 1.0) | 275 | 241 | 0.0027 | 0.0016 | 0.0012 |
| **All in-candidate errors (FP + FN)** | 16,718 | 14,255 | **0.0275** | 0.0140 | 0.0135 |

- **Blocking accounts for 75% of the loss** (being bucketed separately). 3,439 non-singletons get an empty prediction (cost 0.039);
  **2,941 of them have zero true matches in candidates**. So "empty on non-singleton" is almost entirely a blocking problem.
- **Decision vs scoring (in-candidate):** an oracle choosing the best top-k per S1 with the *current* prob ordering reaches
  **0.9300 (+0.0243)**, against +0.0275 for fixing every in-candidate error. So the ranking inside each S1 is almost right; the loss is
  **probabilities that are not comparable across contexts** (name-only records, co-located records), not ordering.
- A top-1 fallback for S1s with no prediction **hurts** at every t2 (−0.0045 at 0.2 … −0.00002 at 0.6): those entities are mostly blocking misses or true singletons.
- FP probs: 47% in 0.675–0.8, 12% > 0.97. FN probs: 48% in 0.4–0.675, 14% < 0.05. India is weaker (0.877 vs 0.925), and its gap is mostly blocking (0.038 vs 0.032).

## 2. Taxonomy of in-candidate errors (ranked by macro cost)

Bucket rules are applied in priority order. The TP comparisons use a 20k TP sample (×12.45).

| # | bucket | n | share | singleton FPs | macro cost | IN | US | median p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FN1 | **candidate address empty (name-only record)** | 5,419 | 41% of FN | – | **0.0067** | 0.0027 | 0.0040 | 0.44 |
| FN5 | house-number noise (803↔484, 216↔916, 1060↔1055), name ≥ 80 | 2,890 | 22% | – | 0.0042 | 0.0020 | 0.0023 | 0.28 |
| FP5 | extra / substituted business word (Services, Group, Exports, Holdings; VI↔VW, LC↔GC) | 1,135 | 31% of FP | 90 | 0.0033 | 0.0019 | 0.0013 | 0.89 |
| FP1 | **candidate address empty, exact name, owned by a same-name S1 elsewhere** | 987 | 27% | 56 | **0.0027** | 0.0009 | 0.0018 | 0.74 |
| FN4 | co-located, unrelated name (addr ≥ 90, name < 60): "Iririza", "Drexflux" are TRUE matches | 1,904 | 15% | – | 0.0024 | 0.0013 | 0.0011 | 0.45 |
| FP3 | co-located different business (addr ≥ 90, name < 60) | 630 | 17% | 40 | 0.0017 | 0.0009 | 0.0009 | 0.78 |
| FP6 | house number differs, same name | 541 | 15% | 40 | 0.0016 | 0.0008 | 0.0008 | 0.89 |
| FN7 | name typo / word swap, address OK (Neetwork, Knnp, P1atinum, PRlVATE) | 857 | 7% | – | 0.0011 | 0.0006 | 0.0006 | 0.35 |
| FN8 | both name and address degraded | 827 | 6% | – | 0.0011 | 0.0006 | 0.0005 | 0.39 |
| FN2 | native script (Telugu/Bengali/Devanagari transliterations) | 488 | 4% | – | 0.0011 | 0.0011 | 0 | 0.30 |
| FP7 | other near-duplicates | 283 | 8% | 46 | 0.0009 | 0.0007 | 0.0002 | 0.91 |
| FN0 | lost to another mini S1 (assignment) | 416 | 3% | – | 0.0007 | 0.0005 | 0.0002 | 0.03 |
| FN6 | address variant (name OK, addr < 70) | 132 | 1% | – | 0.0002 | | | 0.22 |
| FN3 | domain / handle | 152 | 1% | – | 0.0002 | | | 0.20 |
| FP2/FP4 | empty-address distractor / same name, different address | 57 | 2% | 3 | 0.0001 | | | |

**Grouped by root cause:**
- **(A) Name-only records (empty candidate address), FN1 + FP1 = 0.0094.** 44% of in-candidate FNs and 27% of FPs have an empty candidate
  address, against 1.5% of TPs. The model sees identical features for "the true record" and "the same-name record of a sibling S1"
  (966 of 987 FP1 are *exact feature ties* with the true owner). **About 50% of all S1 share their exact core name with another S1**
  (train IN 54%, US 48%; test FR 52.5%, IN 54%, US 40%). So a name-only record is resolvable only through S1-side name uniqueness.
- **(B) House-number ambiguity, FN5 + FP6 = 0.0058.** The generator puts ±k / edit-1 house numbers on both true matches and distractors.
- **(C) Business-word edits, FP5 + FN7 = 0.0044.** The injected words in FPs (services, center, partners, group, public) are the *same* words injected
  into true matches (TP extra-word top list: center, services, service, partners). Name tokens alone cannot separate them.
- **(D) Co-location, FN4 + FP3 = 0.0041.** True matches with a totally unrelated name at the exact S1 address exist (random-name positives).
  FPs of the same shape belong to *another* S1 at that address.

## 3. Competition: the val protocol is not test-like
- **55% of FPs (1,985 / 3,633) are records owned by another S1 that is NOT a mini query** (only 36 owners are in mini). In val,
  `assign_best_s1` only sees the 88k mini S1s. On test, all 1.73M S1s compete.
- In 839 of those, the true owner dominates this S1 on both name and address similarity (FP3: 422/543, FP5: 224/251, FP6: 89/101). If full
  competition removes them on test, that is **+0.0021**. The reverse effect (true records stolen by non-mini S1s on test) is **not measured** by the current protocol.
- Consequence: the tuned t=0.675 is tuned under weaker competition than test. For FP1 (exact ties), competition does not help at all.

## 4. Samples (stratified 44 FP / 44 FN; representative excerpts)
```
FP1  p=.682 S1 "Shir's Appliance" | 3803 Belvoir Dr, Huntsville AL     C "Shir's Appliance" | ''      owner: "Shir's Appliance Associates", Middleborough MA
FP1  p=.764 S1 Orris Dealcomm Private Limited | Panchkula HR            C "Orris Dealcomm Private" | ''  owner: same name, Pune MH
FP3  p=.745 S1 Fernandez Trust | 2384 Greenvale Rd, Cleveland OH       C "Drusi" | same address        owner: "Drusi K. Nilsen ... PLLC", same address
FP3  p=.683 S1 Premier Syntex Limited | Godrej Millennium ... Pune      C "RI" | same address           owner: Rose Industries Pvt Ltd, same address
FP5  p=.954 S1 Lombard Environmental Society | 801 Pinebrook Dr         C Lombard Nexlum Society | same (distractor)
FP5  p=.995 S1 Education Center VI | 8108 Jones Ave, Lincoln NE        C Education Center VW Inc. | same (distractor)
FP5  p=.961 S1 Industrial Complex Dundahera Foundation Pvt Ltd         C ... Dundahera India Private (Limited)   owner: the "India" S1, same address
FP6  p=.845 S1 West Translational | 200 11th Ave Unit 234 Nashville   C West Translational (Co) | 211 11th Ave Unit 234 (distractor)
FP-singleton p=.980 S1 Mym Academy Mumbai | ... Cross Rd No 4          C Mym Academy Mumbai Pvt Ltd | ... Cross Rd No 15 (distractor)
FN1  p=.338 S1 Shree Thai Pvt Ltd | B-9/287 Rohini, Delhi              C "Shree Thai Pvt Ltd" | ''
FN1  p=.208 S1 Nodyne | 1911 State Ln, Stillwater OK                   C "Nódyne" | ''
FN4  p=.634 S1 Developers Beni Delhi Pvt Ltd | Plot 330/2973 Lingaraj Ngr  C "Drexflux" | identical address (TRUE)
FN5  p=.261 S1 Upper Four Inc | 803 Eileen Ln, East Ridge TN          C UPPER FOUR | 484 EILEEN LN (TRUE)
FN5  p=.055 S1 Urgent Care Unified Partners PLLC | 1402 Chestnut St     C ... | 4402 Chestnut St (TRUE)
FN7  p=.330 S1 VD Pure Private Limited | Plot B-35 Mohali              C VD PMFURE PRlVATE LIMITED | same  (legal typo leaks into core: "vd pmfure prlvate")
FN2  p=.004 S1 First Laxmi Constructions Pvt Ltd                        C ফার্স্ট লক্ষ্মী কনস্ট্রাকশনস ... → "pharst lksmi knstraksns praibhet"
```
**Normalization leak:** honorifics and transliterated legal words stay in `n_core`: mr/dr/smt/"m s"/www/com/lnc/praibhet/piraivet/limirrd/
prinvbgate/privadte. Cleaning them makes 399 FN (and 60 FP) name-equal. Upper bound about +0.0005.

## 5. Measured fix: S1-context uniqueness (post-processing on existing probs, no retraining)
Keys are computed over **ALL S1 of the split and country** (train 2.2M for val, test 1.73M for test; unlabeled, so legal and transductive):
`n_key` = sorted `n_core` tokens, `a_key` = sorted `a_full` tokens, `s1_same_name` = #S1 with that n_key, `s1_same_addr` = #S1 with that a_key.
- **ruleA (add):** candidate address empty ∧ n_key(S1) = n_key(cand) ≠ '' ∧ s1_same_name = 1 → adds 2,194 pairs, 2,096 true: **+0.00234**.
  (With token_set = 100 instead of exact key: −0.0049. Exactness matters, because extra-word siblings are distractors.)
- **dropA (remove):** kept pair ∧ candidate address empty ∧ s1_same_name ≥ 2 → drops 1,608 pairs (719 true): **+0.00141**.
- **ruleB (add):** a_key(S1) = a_key(cand) ≠ '' ∧ s1_same_addr = 1 ∧ name_tset < 50 → adds 1,387 pairs, 1,106 true: **+0.00070**.
  (Without the name condition: −0.0005, because extra-word distractors sit alone at the S1's address. dropB (same address, ≥ 2 S1, name < 60): −0.00015, not used.)
- **A + dropA + B: 0.9057 → 0.9102 (+0.00443, paired bootstrap 95% CI [+0.0041, +0.0048], P(not better) = 0)**. India 0.8771 → 0.8817, US 0.9253 → 0.9296.
  With the threshold re-tuned to t=0.70: 0.91015.

## 6. Proposed fixes (top 3 + extras)

| rank | fix | targets | expected gain (mini) | cost | France risk |
|---|---|---|---|---|---|
| 1 | **EXP-016 S1-context features** in the LGBM: `s1_same_name` (count and ==1 flag), `self_name_unique` (S1's own n_key count), `n_key_equal`, `s1_same_addr` (count and ==1 flag) plus interactions with `cand_addr_empty` and `name_tset`. All counts over all S1 of the split and country. Ship the §5 rules as an immediate post-processing fallback. | A, D (0.0135 combined) | **+0.0044 measured as rules**; +0.005–0.008 expected as features | Low: two group_by counts per country (< 1 min, a few hundred MB), no blocking change | Low. France name-sharing is 52.5% (same as train). France address sharing is 19% (vs 6–7%), so co-location features will mostly say "ambiguous" there, which is the precision-safe direction. Use the ==1 flags rather than raw counts (US test S1 density is half of train). |
| 2 | **EXP-017 Competition-aware val + record-side features (extends EXP-010/EXP-003):** add as val "competitor queries" all train S1s that share a blocking key with any mini candidate, so `assign_best_s1` sees test-like competition. Add record-side features: best-other-S1 name_tset / addr_tset and margin. | FP3, FP5, FP6 owned-by-other (0.0021 at stake), plus a trustworthy threshold | +0.002 FP side (less if TP-stealing appears); main value is an honest val | Medium: the val query set grows by roughly 5–10×, and the reverse blocking pass | Neutral; the same code runs on test |
| 3 | **EXP-018 Number features:** house digit-edit distance, abs numeric diff, same-length flag, first/last digit changed, agreement of the non-house numbers (unit/plot), house present only on one side. Plus the **extra-token doc-frequency** (max S1-df of cand-only tokens, per country) for business-word edits. | B (0.0058), C (0.0044) | +0.001–0.002 (both are partly irreducible generator noise) | Low (vectorizable) | Low. Per-country df is computed from the data, and French legal words are already canonical. |
| 4 | Normalization v1: honorific stopwords (mr/mrs/dr/smt/m s/www/com), fuzzy legal-form canonicalization (edit ratio ≤ 0.34 to private/limited/pvt/ltd/inc/company → legal). Bump NORM_VERSION. | FN7 part | ≤ +0.0005 | Low | Must add French equivalents (M./Mme/Ets/Cie) and apply the same fuzzy rule to SARL/SAS typos |
| – | Blocking (separate track) | 0.070 | largest bucket | – | – |

**Recommended next experiment:** EXP-016. Add the S1-context uniqueness features and retrain, keeping the §5 rules as a zero-retrain fallback,
then measure mini, LOCO and India/US deltas. Also evaluate the rules on the *current* test predictions as a candidate resubmission
(needs `s1_same_name`/`s1_same_addr` from test S1), subject to the val-beats-best rule and validation-auditor review.
