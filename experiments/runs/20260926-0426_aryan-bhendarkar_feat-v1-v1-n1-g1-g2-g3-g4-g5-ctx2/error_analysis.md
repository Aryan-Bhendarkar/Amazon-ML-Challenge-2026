# Error analysis: 20260926-0426_aryan-bhendarkar_feat-v1-v1-n1-g1-g2-g3-g4-g5-ctx2 (mini, 87,952 S1, t=0.75)

I rebuilt the matches from `val_pred.parquet` with `assign_best_s1` and t=0.75. The result is F0.5 **0.980021**, which matches the logged value exactly, and the per-entity scores are identical to `val_entity_scores.parquet`.
Every number below is a **macro-F0.5 delta on mini**. Each one comes from recomputing per-entity F0.5 with that bucket's errors fixed (FN → TP, or FP removed).
The scripts are in the session scratchpad. Rule and bucket definitions are written out inline so they can be re-implemented.

## 1. Loss decomposition (total loss = 1 − 0.9800 = 0.0200)

| error source | pairs | entities | gain if fixed | India | US |
|---|---:|---:|---:|---:|---:|
| FN in candidates, assigned to this S1, prob < t | 8,607 | 7,874 | **0.0095** | 0.0040 | 0.0055 |
| FN lost in `assign_best_s1` to another mini S1 (own prob < t in 1,040 of 1,046) | 1,046 | 1,008 | 0.0012 | 0.0004 | 0.0008 |
| **Blocking miss** (true pair not in the 94 candidates/S1) | 4,862 | 4,073 | **0.0059** | 0.0040 | 0.0020 |
| FP, distractor (record matches no S1) | 553 | 547 | 0.0020 | 0.0010 | 0.0011 |
| FP, record owned by a NON-mini S1 (never competes in val) | 612 | 547 | 0.0017 | 0.0011 | 0.0006 |
| FP, owned by another mini S1 | 13 | 13 | 0.00003 | | |
| ↳ FP on true singletons (111 entities, each costs 1.0) | 130 | 111 | 0.0013 | 0.0007 | 0.0005 |

- **Scoring vs decision:** an oracle that picks the best top-k per S1 under the *current* prob order reaches 0.9919 (+0.0119). A perfect order plus decision inside the assigned candidates reaches 0.9933. Only 1,405 S1s have a negative ranked above one of their trues.
  So the in-candidate loss is again **calibration inside ambiguous cells, not ranking**. The per-cell analysis (§3) shows those cells are mostly irreducible at the pair level. This explains why stage-2/K-model/expected-F gave ≤ +0.0003 (decisions.md, 26 Sep).
- **Threshold:** the per-country optimum is 0.75 for both India (0.97471) and US (0.98365). A per-country threshold gains nothing.
- **Losses by cluster size:** n_true=1 entities have F0.5 0.940 and cost 0.0032 (16% of the loss from 5.3% of entities). 91 of their 141 in-candidate FNs are name-only records.
  There are 372 non-singletons with an empty prediction (cost 0.0042). Their FNs split 243 blocking, 256 low-prob, 41 lost-assign.
- **By source:** FNs and FPs are split roughly 45/55 between S2 and S3 in every bucket, so there is no source effect worth modelling.
- **Probability bands:** FN probs fall evenly across 0.01–0.75 (747 below 0.01; 1,562 in 0.6–0.75). FP probs: 219 in 0.75–0.8, 386 in 0.8–0.9, 334 in 0.9–0.97, and 239 above 0.97.

## 2. Taxonomy of in-candidate errors (ranked by macro cost)

Bucket rules are applied in priority order on the model's own features: name-only → native → domain/handle → co-located (addr_tset≥90 & name_tset<60) → house differs (name_tset≥80 & house_rel∉{equal,zeros,miss}) → …

| # | bucket | n | share | gain | IN | US | median p | notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| FN-A | **name-only record (empty cand address)** | 5,666 | 59% of FN | **0.0061** | 0.0022 | 0.0039 | 0.20 | **5,415 (cost 0.00585) are ambiguous: another S1 has an identical or at-least-as-close name** (§3a) |
| FN-E | house number differs (name ≥ 80) | 1,451 | 15% | 0.0019 | 0.0007 | 0.0012 | 0.41 | 656↔655, 471↔472, 1-↔78. Calibrated (§3b) |
| FP-E | house number differs | 367 | 31% of FP | 0.0013 | 0.0006 | 0.0007 | 0.88 | 44 on singletons. 236↔237, 142↔143, 254↔25, 2525↔252 |
| FN-D | co-located, unrelated name ("Veoquo", "Drexquoonyxio" are TRUE) | 931 | 10% | 0.0011 | 0.0005 | 0.0006 | 0.28 | |
| FP-G | extra/substituted word | 230 | 20% | 0.0007 | 0.0004 | 0.0003 | 0.92 | 70/95 owned ones have a dominating owner |
| FP-A | name-only (same-name S1 elsewhere) | 202 | 17% | 0.0006 | 0.0002 | 0.0003 | 0.86 | 88% owned; 138/178 are feature ties with the owner |
| FP-K | same name + same house, address/legal variant only | 140 | 12% | 0.0005 | 0.0004 | 0.0002 | 0.94 | 31 singletons ("Entity Trust Limited" 113/1→113/10) |
| FN-G | name edit (word/typo), address OK | 481 | 5% | 0.0005 | 0.0003 | 0.0002 | 0.39 | "Smt Rising Pvt Ltd Center", "Brand Consulting of Saman" |
| FP-D | co-located different business | 150 | 13% | 0.0004 | 0.0003 | 0.0001 | 0.87 | 87% owned by the S1 at that address |
| FN-F | house missing on one side | 262 | 3% | 0.0003 | | | 0.45 | |
| FN-C/B/I/H/Z | domain/handle 220, native 166, both degraded 228, addr variant 98, other 150 | 862 | 9% | 0.0011 | | | | |
| FP-B | native script (city differs: "एस बिजनेस Limited", Pune vs Latur, p=.996, singleton) | 63 | 5% | 0.0002 | 0.0002 | 0 | 0.93 | |

**Against the baseline-v0 analysis (25 Sep):**
- name-only FN 0.0067 → 0.0061, and name-only FP 0.0027 → 0.0006.
- house FN+FP 0.0058 → 0.0032.
- business-word FN+FP 0.0044 → 0.0012.
- co-location 0.0041 → 0.0015.
- blocking 0.0704 → 0.0059.

The ctx features closed everything that was *reducible* in those buckets. What remains is mostly the generator's symmetric noise.

## 3. Why the big in-candidate buckets are (mostly) irreducible

**(a) Name-only records.** I ran a reverse retrieval: TF-IDF char-3gram over ALL train S1 names per country, top-20, then `fuzz.ratio`. It gives, for each (S1, name-only record) pair, the margin between this S1's name similarity and the best OTHER S1's.

| cell | FN | FP | TN | TP | reading |
|---|---:|---:|---:|---:|---|
| exact key, unique S1 name (s1_name_c=1) | 19 | 56 | 55 | 4,621 | solved |
| exact key, shared by 2 S1 | 876 | 25 | 993 | 57 | coin flip (48% true) |
| exact key, shared by ≥5 S1 | 1,885 | 0 | 23,781 | 0 | 7% true |
| non-exact, margin ≤ 0 (another S1 ≥ as close) | 1,815 | 101 | 12,134 | 259 | 13% true |
| non-exact, margin > 0 | 209 | 18 | 320 | 1,179 | already mostly kept |

Adding the uncovered margin>0 pairs **hurts** at every setting: −0.00003 to −0.00019 across margin > 5/10/20 × ratio ≥ 80/90.
S1/record IDs carry no signal either (Spearman −0.001).
**Conclusion: about 0.0059 of the 0.0061 is a hard floor. Stop spending on name-only records.**

**(b) House numbers** (assigned pairs with name_tset≥90 and street_tset≥90):

| relation | P(true) | mean prob |
|---|---:|---:|
| edit1 | 0.679 | 0.673 |
| near (±10) | 0.688 | 0.672 |
| diff | 0.879 | 0.878 |
| prefix | 0.960 | 0.962 |

The model is calibrated per relation, and the sibling-consensus features (`sib_s1house_agree`) barely move P(true). The generator puts ±k and edit-1 numbers on true copies and distractors alike ("Heart Foundation 655→656" is TRUE; "Oncology Associates 236→237" is a distractor).

**(c) Co-location.** In the ambiguous cell (the exact a_key of the cand matches no S1) there are 766 FN, 3,087 TN and 5,933 TP.
A set-based a_key did not separate them. Neither did a fuzzy count (same house, addr tset ≥ 90, S1s of the country), which gave 639 FN / 1,925 TN in its zero cell. Random-name distractors sit at the S1's own address, so this is irreducible at the pair level.

**(d) Val competition artifact.** Using a name + address + house proxy, the true owner dominates this S1 in 325 of the 625 owned FPs (cost 0.0009): house 83/92, word 70/95, co-located 65/130, native 40/51.
I removed all 824 owner-dominated assigned negatives (p≥0.4) to simulate test-time competition:

| t | 0.65 | 0.70 | 0.75 | 0.80 |
|---|---:|---:|---:|---:|
| F0.5 | 0.9807 | 0.9809 | 0.9809 | 0.9806 |

**The optimal threshold stays at 0.70–0.75, so no retune is needed.** Val understates test by ≤ 0.0009 on the FP side. The reverse effect (true records stolen by non-mini S1s on test) is still unmeasured.

## 4. Blocking misses (4,862 pairs, 0.0059)

The overall pair miss rate is 1.6%. It is 4.1% for **native-script records** (vs 1.4% for Latin) and **10.8% for name-only records**.

| bucket (S1 vs missed record, norm_v1 sims) | n | gain | IN | US |
|---|---:|---:|---:|---:|
| **native script** (830/905 have name_tset ≥ 90 after token-map transliteration; 85% have an exact n_key) | 905 | **0.0016** | 0.0016 | 0 |
| name-only | 1,430 | 0.0015 | 0.0006 | 0.0010 |
| name ≥ 80 & addr ≥ 80 but not retrieved (S1 at the 100-cap; "Star Exports 98 vs 97 DL", "Real Business 47 vs 0047") | 811 | 0.0012 | 0.0009 | 0.0003 |
| name ok, address variant | 625 | 0.0009 | 0.0007 | 0.0003 |
| addr ok, name degraded / domain / co-located / both degraded | 1,091 | 0.0013 | | |

Native misses are **generic Indian names with short native addresses** ("न्यू टेक्नोलॉजीज प्राइवेट लिमिटेड" | "305, Pune, MH"). The `tf_na` top-30 (address weight 0.65) is crowded by same-name records, and the name-only retrievers only cover empty-address records.

**Measured key (no model run):** `nkey_num` = (country, sorted unique n_core tokens [norm_v1], any number token from a_numbers split on space, / and -, leading zeros stripped), with a pool block cap of 50.
- **It recovers 938 of 4,862 misses (914 India). The upper bound is +0.0015 mini (India +0.0037).**
- It adds 1.09 NEW pairs/S1 (+1.2% scoring). India gets 2.6/S1, US 0.07/S1.
- On test it adds 2.7/S1 for India and **0.76/S1 for France**. France is generic and dense, so crowding is plausible there. Test has no labels, though, and the "unique-name" label-free proxy is too weak to size the France gain.
- In-candidate analogue (n_key_equal & shared number): true-pair recall 0.997.

## 5. Side finding: ctx features read norm_v0 for native-script names

`ber.ctx_features.NORM_V = 0` is never overridden (`build_cache`/`v1_test` only patch `baseline_v0.NORM_V`). So the G1/G3 keys use untransliterated v0 names, while the base features use v1. For native candidates that are TRUE with name_tset=100:

| | native | Latin |
|---|---:|---:|
| `n_key_equal` | 1.4% | 84% |
| `ex_n>0` | 98.6% | 7% |
| `s1_name_c=0` | 98.5% | 14% |

This is consistent between train and test, so there is no skew. It is a lost signal though: native FP+FN cost 0.00046, and fixing it matters for the pairs `nkey_num` would add (mostly native).

## 6. Samples (stratified 35 FP / 40 FN; representative excerpts, raw → norm)
```
FP-E  p=.980 S1 Oncology Associates L.L.C. | 236 Newton St, Edwardsville KS      C Oncology Associates LLC | 237 NEWTON ST (distractor, S1 has 20+ same-name)
FP-E  p=.803 S1 Medina Kongsberg Center | 254 Americus Rd, Taylortown NC          C Medina Kongsberg Center Ltd | 25 AMERICUS RD (distractor)
FP-K  p=.968 S1 Entity Trust | A.N.113/1, Hamidpur, Varanasi                      C Entity Trust Limited | A.N.113/10, HAMIDPUR (distractor)
FP-B  p=.996 S1 Ace Business Limited | 506 Bhoi Galli, Kallam, Latur (singleton)  C एस बिजनेस Limited | FLAT NO- A-506, PUNE  -> "ace business" (distractor)
FP-D  p=.995 S1 Tadaify Corporation | 10704 Admirals Lassie Ln, Berlin MD         C Zetakor Corporation | same address (distractor)
FP-D  p=.902 S1 Apollo Dia Pvt Ltd | House 12583, Ward 8 ... Ludhiana             C FAYEHALO | HUSE NO. 12583/6 ...  owner: Value Software Care, same addr
FP-A  p=.773 S1 Mumbai Narayan Pvt Ltd | Mira Rd (singleton)                     C Mumbai Narayan Private Ltd | ''   owner: same name, Fancy Chambers
FP-G  p=.943 S1 Tech Producer Pvt Ltd | Nesco IT Park 10th fl                     C Tech Infraastdructure Pvt Ltd | same  owner: Tech Infrastructure (dominates)
FN-A  p=.009 S1 Anand Media Pvt Ltd | Janakpuri, Delhi (n_true=1 -> F=0)         C Anand Media Private Limited & Co | ''  (20+ S1 share "anand media")
FN-A  p=.598 S1 One Electronic Pvt Ltd | Gandhinagar                              C One électronic Pvt Ltd | ''   (2 S1 share the key)
FN-E  p=.003 S1 Heart Foundation PC | 655 Van Alstyne Rd, Webster NY              C ... Heart Fóundation-Inc | 656 Van Alstyne Rd (TRUE)
FN-E  p=.740 S1 Mercado and Gray, Inc | 471 Townsend Harbor Rd                    C MERCADO MERCADO AND GRAY, INC | 472 ... (TRUE)
FN-D  p=.057 S1 Brand Fund (India) LLP | A-285 Shastri Nagar, Jodhpur             C Veoquo | JODHPUR, SHASTRI NAGAR, A-285 (TRUE)
FN-G  p=.750 S1 Rising Retail Pvt Ltd | 23/1 Lavelle Rd, Bangalore                C Smt Rising Pvt. Ltd. Center | 23/1, Bangalore (TRUE)
BLK-B        S1 New Technologies Pvt Ltd | 305, 306 A-Wing ... Pune               C न्यू टेक्नोलॉजीज प्राइवेट लिमिटेड | 305, Pune, MH  -> "new technologies" | "305 pune mh"
BLK-B        S1 Lakshmi Products LLP | Milkat 3556 ... Phaltan                    C लक्ष्मी प्रोडक्ट्स एलएलपी | MILKAT NO-556 ...  -> "laxmi products" (translit variant)
BLK-E        S1 Real Business Pvt Ltd | Plot 47, Faridabad                        C private real business limited | Plot No.-0047, Faridabad, HR
```

## 7. Proposed fixes

| rank | fix | targets | expected gain (mini) | cost | France risk |
|---|---|---|---|---|---|
| 1 | **EXP-030 `nkey_num` blocking key**: exact join on (country, sorted unique n_core tokens, number token with leading zeros stripped), pool block cap 50. Union it into the v1 retrievers (new rbits bit) and exempt it from the 100-cap, or give it reserved slots. Rebuild the mini cache and rescore with the frozen ctx2 model, then retrain | 938 of 4,862 blocking misses (native 85%) | **+0.0012 to +0.0015** (upper bound 0.0015; in-cand analogue recall 0.997); India ≈ +0.003 | +1.1 pairs/S1 on mini (+1.2%); test +2.7/S1 IN, +0.76/S1 FR, ≈ 0 US. The key join itself takes seconds | Low. The key is language-agnostic and exact, and the model still decides. It adds 0.76 new pairs/S1 in France (dense generic names, where tf_na crowding is expected) |
| 2 | **EXP-031 ctx features on norm_v1** (`ctx_features.NORM_V=1`, or a `--ctx-norm` flag), recompute ctx2 train/mini, retrain | Native-script G1/G3 keys are wrong for 98% of native candidates; native FP 63 + FN 166 | +0.0002 to +0.0005 alone. Also needed so the pairs from #1 get correct `n_key_equal`/`s1_name_c` | One ctx rebuild (train+mini ~1 h) and a retrain (30 min). The test ctx must be rebuilt too | None (France has no native script; Latin v0 = v1 byte-identical) |
| 3 | **EXP-017 (re-scoped) competition-aware val**: add competitor S1 queries so that `assign_best_s1` sees owners. Main purpose: measure the unmeasured TP-stealing side before the final threshold choice | 325 owner-dominated FPs (0.0009) + unknown TP stealing | Measurement only. FP-side simulation: +0.0009 on test with t unchanged (0.70–0.75 flat) | Medium (reverse retrieval) | Neutral |
| – | **Do NOT pursue:** name-only margin/uniqueness rules (−0.00003 to −0.00019), house ±k rules, co-location key variants. These cells are calibrated and ambiguous (§3); the combined floor is ≈ 0.011 of the 0.020 loss | | | | |
| – | Remaining blocking (cap-crowded high-sim 811 + addr-variant 625, 0.0021): try the India cap 100→150 only after #1, and measure recall per extra pair | | ≤ +0.001 | +50% India scoring | Low |

**Recommended next experiment:** EXP-030 + EXP-031 as one run.
1. Add the `nkey_num` key to the v1 union for mini.
2. Recompute ctx on norm_v1.
3. Retrain, and report mini, LOCO and India/US against 0.9800.

Expected mini ≈ 0.9813–0.9818. Then test-time timing: the key join is seconds, and scoring +1.2% pairs.
