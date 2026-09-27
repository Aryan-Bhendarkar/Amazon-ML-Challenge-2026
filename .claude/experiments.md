# Experiments log (HANDOFF final push, 26–27 Sep 2026)

Baseline for every comparison: **20260926-0710_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3**
- v1_n2 blocking + ctx v3 G1–G5 LightGBM, t = 0.775
- LB sub 20260926-0953: public 0.967
- clean mini 0.9813, clean fold0x 0.98117
- Uses no pretrained weights (LightGBM only).

Orphan-sim protocol:
- Pipeline: `pipelines/orphan_sim.py`; masks from `ber.orphan.removal_mask` (md5(s1_id + "|orphan") mod 1000 < 190).
- Pruning covers all 2.2M train S1. All S2/S3 records are kept.
- G1/G2/G3 split statistics are recomputed on the surviving S1.
- Only surviving eval S1 are scored and assigned.
- Paired bootstrap: 1,000 resamples on identical surviving S1.

## EXP-A Orphan-simulated validation + diagnosis (20260926-1154_aryan-bhendarkar_orphan-sim-mini, commit 1f6e7c2, 26 Sep 12:24–12:44 IST)
- change: none (eval only). The current model 0710 is scored at its frozen t=0.775 on mini under 19% S1 pruning.
  - Uniform mask: md5(s1_id|orphan) mod 1000 < 190.
  - Biased mask: ×2 for S1 whose name or address key is shared, ×0.5 otherwise, rescaled to 19%.
  - Removed S1: 418.5k / 420.3k of 2.21M (19.0% / 19.1%).
  - Records per S1 after pruning: 5.77 (test 5.76; per country, test FR 5.53 / IN 5.82 / US 5.76).
- sanity: clean recomputed through the new code path = 0.98129, identical to the run's logged mini score.
- clean mini: 0.98129 (P 0.9963, R 0.9537; IN 0.97779, US 0.98369)
- orphan-uniform mini: 0.97760 (P 0.9948, R 0.9473; IN 0.97119, US 0.98199)
  - **Δ vs clean on the same 71,270 surviving S1: −0.00374, 95% CI [−0.0041, −0.0034]**
- orphan-biased mini: 0.97739 (P 0.9949, R 0.9471; IN 0.97345, US 0.98006)
  - **Δ −0.00479, CI [−0.0051, −0.0045]**
- loss split (FP on singletons / FP on non-singletons / FN), as a share of macro F:
  - clean: 0.0012 / 0.0023 / 0.0152
  - u19: 0.0017 / 0.0032 / 0.0175
  - b19: 0.0016 / 0.0032 / 0.0178
  - So about 40% of the drop is FP and about 60% is FN.
- FP sources:
  - u19: 297 of 1,212 FPs are orphan records; clean has 1,070 FPs on 88k S1.
  - b19: 388 of 1,202.
  - FPs per S1 rise about 40%.
- mechanism, from the micro diagnosis (LightGBM pred_contrib, clean vs u19 on identical pairs):
  - 180 TP→FN flips and 33 TN→FP flips.
  - Flipped TPs lose −1.82 logit on `ex_lfrac_cmax` and −0.97 on `ex_lfrac_cmin`. These G3 v2 features are log(df/n_S1) for common tokens.
  - Their values change on 100% of flipped pairs, while `s1_name_*` change on only 11%. The model uses the exact frequency value as a token-identity code, so sampling noise in df/n crosses its fine splits.
  - This brittleness is present on test regardless of orphans: test df/n differ from train, and France's tokens differ entirely.
- label-free test comparison (predicted matches per S1 / empty-prediction rate):
  - test: FR 3.18 / 6.1%, IN 3.27 / 6.0%, US 3.41 / 5.7%
  - sim u19: IN 3.27 / 6.0%, US 3.32 / 5.9%
  - clean val: IN 3.31 / 6.0%, US 3.32 / 5.8%
  - India test matches the sim exactly. US test has about +0.09 matches per S1 more than any sim, which suggests an extra US-test FP source (or denser true clusters) that orphans don't explain.
- verdict: **NEEDS-REVIEW.**
  - The uniform drop (0.0037) is below the handoff's 0.004 bar ("orphans are not the main gap"). The biased drop (0.0048) is in the 0.004–0.006 "continue B + start E" zone.
  - The simulation explains at most about 0.004–0.005 of the 0.014 val→LB gap. The rest is most likely France (15% of test, no labels) plus the US-specific excess.
  - The largest piece of the sim drop is lfrac feature brittleness, which is cheap to fix and useful even if the orphan hypothesis is wrong.
- test cost: none. The EXP-A eval costs about 20 min per mini run (3 scenarios, 7 threads).

## DIAG-US Label-free US over-match check (current model 0710; 26 Sep 13:00 IST; no run_id, analysis only)
- question: test US predicts 3.41 matches/S1 vs 3.32 on val. Is that FPs?
- predicted match-count distribution: test US has more S1 with 5–8 matches than val US predictions (5: 14.3% vs 13.4%, 6: 7.2% vs 6.6%). Its shape is close to val US *ground truth* (14.7%, 7.4%). The same shift appears, weaker, for India.
- assigned-pair probability mass per S1 in the uncertain band, test vs clean val:
  - US: 0.7–0.775: 0.033 vs 0.0125 (2.6×); 0.775–0.85: 0.039 vs 0.015 (2.6×); 0.85–0.9: 0.035 vs 0.015 (2.3×); 0.9–0.95: 1.9×; 0.95–0.98: 1.7×.
  - India: 1.5–1.7× in the same bins. France: like the US.
  - On val these bins are only 72–88% precise.
- the orphan sim reproduces only part of it:
  - u19: 1.2–1.3× (US) and 1.3–1.4× (India)
  - b19 US: 1.6–1.8×
  - So 19% orphans explain roughly a third to half of test's uncertain-band inflation. A stronger or chain-biased pruning, or another negative shift, would explain the rest.
- a test-like scenario `t19` (thin each country to test's pre-pruning S1 density: US ×0.62, IN ×1.0; then u19) was added to orphan_sim.
  - micro: −0.0035 vs clean (u19 −0.0038). Density thinning adds nothing, so US density is not the cause.
- verdict: the US "over-match" is uncertain-band inflation, consistent with orphan-type negatives at a higher rate than the uniform 19% sim. Supports keeping b19 as a guardrail and preferring models that shrink the uncertain band under pruning.

## DIAG-FR France token audit (EXP-F prep; 26 Sep 13:05 IST; analysis only, no dictionary applied)
- method: test pairs assigned with p ≥ 0.3 by the current model. Extra/missing core-name and address tokens per 1k pairs, per country, confident (p ≥ 0.9) vs uncertain band (0.3–0.775).
- addresses:
  - S1 carries the region ("hauts de france", "nouvelle aquitaine", "pays de la loire") while candidates carry the department ("nord", "gironde", "loire atlantique", "pas de calais") or nothing, in 25–30% of FR pairs.
  - **It does not matter:** among 457k FR pairs with an equal name key and equal address once region/department tokens are ignored, the share below t is 0.007% with a mismatch vs 0.004% without (median p 0.9999 in both). **No dept↔region map needed.**
- names:
  - The uncertain band is dominated by French distractor words: développement 88/1k, groupe 67, participations 62, holding 40, france 28, cie 27, international 22, fils 19, et 14 (confident pairs: ≤5/1k each).
  - Missing words (the candidate drops them): club, école, amicale, comité, sportive, amis, maison, primaire.
  - These are the French versions of the train distractors ("X Holdings", "X Group"), so the band is expected. Their labels are unknown, so no rule is defensible.
- small gaps (not worth a norm bump and a full test re-featurization at this point):
  - legal-form OCR typos "5arl"/"5as" (0.5/1k)
  - "et" is not a name stopword ("and" is)
  - q↔quai, crs↔cours
- verdict: no French normalization change is justified by label-free evidence. EXP-F.2 is dropped for the region map.
- Alias-mining proposals (EXP-F.3) would only cover these low-volume tokens. Not written; tell me if the human still wants the review file.

## EXP-E1 prototype: in-candidate sibling rescue (26 Sep 13:15 IST; analysis only)
- recall buckets on clean mini (FN pairs / oracle F0.5 gain if fixed):
  - below-t in candidates: 9,143 / +0.0101
  - blocking miss: 3,924 / +0.0045
  - stolen by another S1: 1,036 / +0.0012
  - 97% of below-t FNs are in S1s that already have ≥1 TP. Below-t FN prob median is 0.34, so most are genuinely ambiguous.
- rule tested: add a candidate assigned to A with p ∈ [lo, t) if it is near-identical to one of A's anchors (p ≥ 0.98): token_sort ≥ 90–100 on name and address, house equal.
  - Precision was 0.55–0.87 on only 38–216 pairs; ΔF between −0.00014 and +0.00002, clean and u19 alike.
- verdict: **KILL.** These near-sibling low-prob candidates are mostly distractors, and G5 sibling features already capture the signal. The remaining recall lever is blocking misses (+0.0045 oracle): reverse retrieval / 2-hop over records not in the candidate list.

## EXP-E2 sizing: reverse retrieval, record → top-k S1 (26 Sep 13:10 IST; analysis only)
- setup: the 3,924 mini true pairs missing from the v1_n2 cache. Each missed record queries all train S1 of its country (TfViews: name char3 w .35 + address word w .65, the same as tf_na).
- recovered, name+address view: top3 758 (19%), **top5 897 (23%)**, top10 1,108 (28%), top20 1,330 (34%).
  - India top5 376/2,241; US top5 521/1,683.
  - Name-only records are barely recovered (98/1,430 at top5).
- name-only view: top5 288 (7%).
- speed: 1.7–4.6k queries/s at 2 threads. The test query set (~4.3M unassigned records) takes about 15–30 min at 8 threads.
- value: the oracle for all blocking misses is +0.0045, so top5 gives an upper bound of about +0.001 and realistically +0.0005–0.0008. That sits right at the E gate (≥ +0.0005 on orphan-sim), and every new pair is also a new orphan-FP opportunity.
- cost: `rbits` is a model feature, so this needs the full nkey_num-style cycle (augment train/mini/fold0x/test, recompute ctx, retrain), about 3–4 h of box time.
- verdict: **DEFER** to 27 Sep morning, and only if B0/B finish tonight. Lower priority than the B-track.

## DIAG-ORPH Label-free orphan-cluster test (26 Sep 13:20 IST; analysis only). **Contradicts the 19% orphan hypothesis**
- idea: an orphaned S1 leaves its whole cluster (about 3.5 near-identical records) unassigned. Distractors are singletons. So count unassigned records that share (sorted core-name key, house number without leading zeros) with ≥1 / ≥2 other unassigned records of the same country.
- calibration on train (GT-unmatched records + records of S1 removed by the uniform md5 mask + a random FN share):

| train scenario | IN grp2+ | IN grp3+ | US grp2+ | US grp3+ |
|---|---|---|---|---|
| no removal, no FN | 0.96% | 0.08% | 0.59% | 0.01% |
| no removal, 5% FN | 1.81% | 0.18% | 1.06% | 0.04% |
| 5% removal, 5% FN | 6.84% | 3.26% | 5.53% | 2.47% |
| 10% removal, 5% FN | 10.98% | 5.78% | 9.17% | 4.46% |
| 19% removal, 5% FN | 16.92% | 9.46% | 14.43% | 7.32% |

- test (records not matched by 0710): **IN 2.43% / 0.39%, US 5.32% / 0.22%**, FR 8.05% / 2.44% (FR: dense templated names, no train reference).
- reading:
  - The grp3+ rate is the clean orphan signature, because orphaned clusters of ≥3 are common. Test US/IN sit at the no-removal level (0.2–0.4%), far below even 5% removal (2.5–3.3%).
  - US grp2+ (5.3%) comes from *pairs*, not clusters. That suggests distractor pairs, e.g. one near-copy present in both S2 and S3.
  - Orphan records absorbed by another S1 cannot explain the gap: in the u19 sim only about 300 orphan records per 71k S1 were absorbed.
- conclusion: **test is not train with 19% of S1 removed.** The extra ~1.1 records/S1 are unmatched singletons or pairs, i.e. more near-copy distractors per S1. That also explains test's 2–2.6× inflated uncertain band (DIAG-US) and the largest US test sets, which are true clusters plus house ±1 and business-word near-copies at p 0.8–0.98.
- implications:
  1. EXP-B (orphan-pruned training) targets a shift that is not present at the assumed scale. Expected value is low; the u19/b19 metrics are robustness guardrails, not a test proxy.
  2. The B0 lfrac fix stays relevant: it is robustness to any statistics shift.
  3. The real test shift is a higher distractor density in the uncertain band. That calls for more precision in the band: e.g. a threshold chosen with the band's negatives up-weighted to test's density, or features against near-copies. The first is a form of test-score-based threshold adjustment, which the handoff §6 forbids, so it **needs a human decision**.

## EXP-B0a Drop the 4 lfrac features (20260926-1237_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3-nolfra, commit 37c33a4, 26 Sep 12:37–13:35 IST)
- change: retrain the 0710 recipe (v1_n2, ctx3 G1–G5, same LightGBM params and ES) without ex_lfrac_cmax/cmin and mi_lfrac_cmax/cmin. t = 0.75 (0710: 0.775).
- clean mini: 0.98014 vs 0.98129, **Δ −0.00116 [−0.0014, −0.0009]** (P 0.9954, R 0.9529; IN 0.97619, US 0.98283)
- orphan-uniform mini: 0.97943 vs 0.97760, **Δ +0.00183 [+0.0014, +0.0022]**. Its own pruning loss is −0.00076 (0710: −0.00374). Recall under pruning is kept (0.9536 vs 0.9473).
- orphan-biased mini: lost. I accidentally deleted the running eval's artifact dir. It is re-scored in the queued B0b eval.
- LOCO: IN→US 0.96461 (at source t 0.96351); US→IN 0.92307 (at source t 0.91813). The 0710 reference is running (locoref).
- verdict: **KILL as standalone** (clean guardrail fails; u19 +0.0018 is below +0.003). It confirms that lfrac is the pruning-sensitivity mechanism; the coarsened variant (B0b) is the candidate.
- test cost: rescore from saved feats (--from-feats), minutes.

## EXP-A2 probe: twin-targeted orphan removal (micro, 26 Sep 13:35 IST; smoke only, full mini queued in the B0b eval)
- scenario tw50: remove 50% of the S1 that have a same-name S1 at a different house (twin share of S1: train 50.5%, test 48.2%; US test 40% vs train 48%). That is 25% of all S1, rec/S1 6.25.
- 0710 on micro: 0.97863 vs clean 0.98259, **Δ −0.0066 [−0.0081, −0.0052]**. FP-singleton loss 0.0012→0.0020, FP-nonsingleton 0.0025→0.0032, FN 0.0137→0.0162.
- about 1.6× the harm per removed S1 of uniform removal, and it matches the label-free test signature: twin pairs of unassigned records at shifted houses, and 3–4× more assigned pairs with such a twin partner on test than in train.

## DIAG-TWIN What the extra test records are (26 Sep 13:30 IST; analysis only)
- US unassigned records that come in same-(name key, house) pairs: 76k. 93% share the name key with a present S1; only 1% share its house.
  - They are **twin entities**: two records (usually one S2 + one S3) of the S1's name at a shifted house number. Examples: "007 Barclays" at 11331 vs S1 at 11328 Effie Way; "090 Liberty" at 58 vs 37 Calle Cienega.
  - Best-S1 prob: median 0.04, but 9.8k are in 0.5–0.775, right under t.
- Assigned pairs where the candidate's house differs from the S1 house AND an unassigned record shares the candidate's (name key, house):
  - test US 0.256% of predicted pairs vs train true pairs 0.059%
  - test IN 0.72% vs 0.25%
  - test FR 0.18% (house differences are rare in FR)
  - The excess is about 4.5k US and 12.5k IN pairs, likely FPs. That is an upper bound, since test "unassigned" also holds FNs.
- mechanism: the twin's two records agree on the shifted house, so the G5 sibling features read like "the S1 has a house typo". With the twin S1 absent, s1_name_c = 1 and the name looks unique. Train saw this pattern mostly with the twin S1 present.
- implication: the relevant shift is **twin-targeted orphaning** (or injected twin-entity distractors), not uniform pruning. The fix is EXP-B with twin-removal views, and tw50 as a validation scenario.

## DM-VAL Density-matched validation, threshold re-tune of 0710 (20260926-1413_aryan-bhendarkar_dm-val-mini (canonical; re-run of the deleted 1342 record, identical ρ/w/t), 26 Sep 14:13 IST; code `src/ber/dmval.py`, `pipelines/dm_val.py`)
- **how the DM weighting is applied** (monitor note 10, approved):
  1. Band = [0.2, t_ship + 0.2] = [0.2, 0.975]. Mass = assigned pairs (each record kept only at its best S1) with p in the band, per S1 of the eval set.
  2. ρ = test band mass per S1 / mini band mass per S1, where test uses the 0710 `test_pred.parquet` over all 1.73M test S1. One global scalar; the per-country ratio is reported, never used.
  3. The excess test band mass is assumed to be negatives (positives unchanged), so each val band NEGATIVE gets weight w = (ρ·band − band_pos) / band_neg.
  4. Inside the per-entity F0.5, each assigned val band-negative row is replicated c = floor(w) + Bernoulli(w − floor(w)) times (5 seeded draws, averaged).
     - A replicated negative predicted at threshold t adds c FPs: F = 1.25·tp / (1.25·tp + 0.25·fn + FP_weighted).
     - A true singleton with any FP scores 0; positives and out-of-band pairs have weight 1.
  5. The global t is re-tuned on DM-mini (grid 0.600–0.975, step 0.025), then frozen ρ/w/t are confirmed on DM-fold0x.
- numbers:
  - test band 0.4512 pairs per S1; mini band 0.2769 (of which positives 0.1773, band precision 0.640)
  - **ρ = 1.630** (report only: US 1.80, IN 1.39); **w = 2.75**
- curves (clean / DM):

| t | clean | DM |
|---|---|---|
| 0.700 | 0.98112 | 0.97878 |
| 0.750 | 0.98127 | 0.97942 |
| **0.775** | **0.98129** | **0.97967** |
| 0.800 | 0.98108 | 0.97962 |
| 0.825 | 0.98085 | 0.97959 |
| 0.850 | 0.98059 | 0.97951 |
| 0.900 | 0.97965 | 0.97898 |

- **DM-optimal t = 0.775, the shipping threshold.** The DM curve is flat from 0.775 to 0.85.
  - Re-thresholding the 0953 predictions changes nothing, so there is **no DM-threshold submission**.
- DM vs clean at t = 0.775: −0.0016. So the band-density shift explains only about 0.0016 of the 0.014 val→LB gap (normal gap ≈ 0.005). Roughly 0.007 remains unexplained: France is the prime suspect (15% of test, no labels), plus any twin FPs beyond the band model.
- DM-fold0x baseline for 0710 (frozen ρ/w/t): running (tmux dmfx). It serves as the reference for the model gates.
- verdict: DM-val is kept as the primary metric for model changes (gate: DM-fold0x Δ ≥ +0.002). Threshold re-tune: **no change**.

## LOCO reference for 0710 (20260926-1337_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3-locore)
- IN→US 0.95425 (at source t: 0.95194)
- US→IN 0.92988 (at source t: 0.92932)
- EXP-B0a (drop-lfrac) vs this: IN→US +0.0104, US→IN −0.0068, so LOCO is mixed. Its KILL stands.

## DIAG-TWIN2 Can near-copy (twin) features or a twin-specific cutoff fix the test twin FPs? (26 Sep 14:10 IST; analysis only)
- in train/val the twin structure is a POSITIVE signal. Definition: candidate house ≠ S1 house, same name key, and ≥2 of the S1's candidates share the candidate's (name key, house).
  - Band precision is 0.78–0.82 (the S1's own house is usually the mistyped one); above t it is 0.953.
  - So a learned feature cannot pick up the test pattern (the same conclusion amlc-07 reached for the G6 twins).
- structural density (candidates p ≥ 0.01, assigned; per S1), val mini → test:
  - band [0.2, 0.975]: 0.0141 → 0.0379, **ρ_twin = 2.7** (overall band ρ = 1.63)
  - predicted part [0.775, 0.975]: 0.0096 → 0.0192
  - above 0.975: 0.197 → 0.191 (unchanged)
- if the test excess is negatives, test precision of predicted twin-structure pairs is about 0.48, i.e. about 16k FPs (matching the DIAG-TWIN excess of about 17k).
  - A twin-specific stricter cutoff would gain about +0.0017 from FP removal and lose about −0.0007 from TPs: **≲ +0.001 net**, below the +0.002 gate. It would also need a second ρ, beyond the approved single global ρ.
- verdict: **not pursued without approval.**
- gap budget: normal ≈ 0.005 + band/twin density ≈ 0.002 leaves ≈ 0.006 unexplained. It is most consistent with France cross-country transfer: LOCO IN→US 0.954 vs 0.984 in-country, and France is 15% of test.

## EXP-B0b lfrac rounded to 0.5 log-units (20260926-1344_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3-lfracq, commit c7568a7, 26 Sep 13:44–14:20 IST)
- change: the 4 lfrac features are replaced by `*_q05` (round(x/0.5)·0.5, via `ctx_features.DERIVED`; applied identically in training, orphan_sim and test rescoring). Everything else as in 0710.
- clean mini: 0.98068 @ t = 0.75 (Δ −0.0006 vs 0.98129); P 0.9959, R 0.9536; IN 0.97680, US 0.98333
- **DM-mini** (frozen ρ = 1.63 / w = 2.75 from 1413; t re-tuned, 20260926-1420): best 0.97911 @ t = 0.775–0.80, **vs 0710 0.97966 → −0.0006**
- LOCO vs 0710 reference: IN→US 0.96221 (+0.0080), US→IN 0.92448 (−0.0054). The drop arm showed the same split (+0.0104 / −0.0068).
- orphan guardrail: not run. The chain was stopped because the primary metric already fails.
- verdict: **KILL** (DM-mini below baseline; LOCO mixed).
- side finding: both lfrac arms improve India→US by +0.008–0.010. That direction (Latin-script target) may be the closer France proxy, but US→India drops by 0.005–0.007. Which LOCO direction speaks for France is a judgment call for the human.
- test cost: n/a.

## DM-VAL fold0x confirmation for 0710 (20260926-1441_aryan-bhendarkar_dm-val-fold0x; frozen ρ/w/t from 1413)
- 353,503 S1. Clean at t = 0.775: 0.98117 (reproduces the logged fold0x). **DM-fold0x 0.97963.**
- DM curve on fold0x: 0.750 0.97948 | 0.775 0.97963 | **0.800 0.97972** | 0.825 0.97967 | 0.850 0.97956.
  - That is flat (+0.0001 at 0.80), so the mini decision (no threshold change) is confirmed.
- This is the **reference for the submission gate** (DM-fold0x Δ ≥ +0.002).

## NOTE13 Per-source thresholds t_S2 / t_S3 (20260926-1512_aryan-bhendarkar_dm-persource, 26 Sep 15:12–15:25 IST)
- change: decision layer only. Separate global thresholds for S2 and S3 candidates (grid 0.70–0.90 × 0.70–0.90, step 0.025) on the 0710 predictions. Tuned on DM-mini (frozen ρ/w from 1413); frozen for DM-fold0x.
- DM-mini optimum: **t_S2 = t_S3 = 0.775**, the global t. DM-mini, DM-fold0x, clean mini and clean fold0x Δ are all exactly 0.
- verdict: **KILL** (no gain; the source split carries no threshold signal).

## EXP-A3a Per-source competition features "ps" (20260926-1511_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3-ps, commit 1b9b1ba, 26 Sep 15:11–16:10 IST)
- change: 0710 recipe + 13 features from `ctx_features.PS_FEATS`, computed per S1 within each candidate source (S2 / S3) from existing scores:
  - rank and gap-to-best of name_tset / addr_tset / cos_na
  - count of same-source candidates ≥ 90 on name / address; same-source list size
  - the other source's best name / address score
  - Rescorable from the saved test_feats (no test re-featurization).
- clean mini 0.98135 @ t = 0.775 (0710: 0.98129).
- **DM (frozen ρ/w; t_DM = 0.825 from mini, 1538)**, paired vs 0710 @ 0.775:
  - DM-mini 0.97990 vs 0.97966: **+0.00024 [−0.0001, +0.0005]**
  - **DM-fold0x 0.97999 vs 0.97963: +0.00036 [+0.00019, +0.00053]** (p ≈ 0)
- clean at t_DM, paired: mini −0.00017 (n.s.); fold0x **−0.00004 [−0.0002, +0.0001]**. At t = 0.775, clean fold0x is 0.98152 (+0.0004).
- LOCO vs the 0710 reference: IN→US 0.95504 (+0.0008), US→IN 0.93508 (**+0.0052**), **avg +0.003**.
- verdict: **KEEP as the new base; submission status NEEDS-LEAD.** It fails the model gate (DM-fold0x +0.00036 < +0.002) but passes the transfer gate (LOCO-avg ≥ +0.002, clean ≥ −0.001). Expected LB effect is small positive (≈ +0.0004 non-FR plus the FR transfer share).
- test cost: `--from-feats` rescore (minutes). Prepared as a submission (below; not requested).

## LB-PROBE France-empty (sub 20260926-probe-fr-empty; uploaded 26 Sep; **public 0.834**)
- setup: the 0953 predictions (public 0.967) with all 259,452 France S1 emptied (FR share of test S1 = 0.1497; any random public subset keeps it within ±0.0005).
- read-out: F_FR = (0.967 − 0.834)/0.1497 + s_FR, with s_FR ≈ 0.056 ± 0.01 → **F_France ≈ 0.944 ± 0.01**.
  - Hence **F_non-France ≈ (0.967 − 0.1497·0.944)/0.8503 ≈ 0.971** vs val 0.981.
- gap decomposition (LB 0.967 vs val 0.981):
  - non-FR ≈ 0.85 × 0.010 = **0.0086** (the larger part)
  - France ≈ 0.15 × (0.98 − 0.944) = **0.0054**
- label-free mass shift, assigned pairs per S1, test vs val mini:

| country | >0.999 | 0.975–0.999 | 0.775–0.975 | 0.3–0.775 | total ≥ 0.3 |
|---|---|---|---|---|---|
| US | 2.674 vs 2.702 | 0.529 vs 0.518 | **0.205 vs 0.103** | **0.201 vs 0.112** | **+0.175** |
| India | **2.312 vs 2.494** | 0.724 vs 0.667 | 0.238 vs 0.144 | 0.146 vs 0.118 | **≈ 0** |
| France | 2.488 | 0.492 | 0.198 | 0.235 | – |

- reading:
  - **US** gains new mass in the band with unchanged top mass → extra negatives (distractors / twins): a precision problem. The DM model fits here.
  - **India** keeps the same total but moves 0.18 pairs/S1 from >0.999 into the band → noisier true copies on test: a recall problem. The DM "excess = negatives" assumption is **wrong for India**; stricter thresholds would hurt there.
  - France looks like the US on the high side and has the most mass in 0.3–0.775.
- implications:
  1. Global-threshold tuning can't serve both regimes; country-keyed thresholds are forbidden.
  2. The non-FR loss (~0.010) is split between US precision and India recall.
  3. Features that make noisy true copies score high (India) and features that separate distractors (US) are the levers. bge-m3 cosines target the India case (native / transliteration noise).

## DIAG-IN India recall shift: what moved into the band? (26 Sep 19:10 IST; label-free, 0710 predictions)
- transliteration coverage (norm_v1 n_full of native-script pool names; token in the same split's Latin-S1 vocabulary):
  - train: tokens 97.5%, names fully covered 90.4%
  - test: tokens 96.2%, **names fully covered 85.2%**
  - Native share of the India pool is the same (18.2% / 18.4%).
  - Real but small: about 1% of India pool records get a worse transliteration on test (train coverage is flattered, since the map was learned on train).
- composition of India assigned pairs in the band (p 0.2–0.975), test (10% S1 sample) vs val mini:

| | test | val |
|---|---|---|
| native candidate | 7.4% | 9.7% |
| name-only candidate | 15.3% | 27.0% |
| house_rel = 0 | 27.6% | 22.5% |
| n_key_equal | 0.415 | 0.488 |
| extra name tokens (ex_n) | 0.72 | 0.60 |

  - Top bucket (p > 0.999): n_key_equal 0.764 vs 0.713, ex_n 0.22 vs 0.29.
- reading: the extra India band mass is **not** native-script noise (the native share in the band is lower on test). It is pairs *with* an address whose name has extra tokens (business-word additions / variants): the same family as the US distractors. Whether these are true copies (recall) or near-copies (precision) cannot be resolved label-free.
- implication:
  - A transliteration-map extension or skeleton feature targets at most ~1% of India records; it is low value for the remaining time.
  - Better lever: word-addition discrimination (G3 ex_* features are already in; bge-m3 in psemb already helped India, mini India 0.9794).
- verdict: India transliteration work **deprioritised** (small ceiling). NEEDS-LEAD only if the lead disagrees.

## SUB-C0 / SUB-C1 Submission #4 candidates (lead decisions 18:50; built 26 Sep 18:48–20:45 IST)
- model: 20260926-1737_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3-psemb (ps + frozen bge-m3 band cosines), t = 0.775 (DM-selected, 1802).
  - val: clean fold0x 0.98192 (+0.00075 vs 0710), DM-fold0x 0.98101 (+0.0014); LOCO 0.95383 / 0.93387.
- **C0** = submissions/files/20260926-2034_aryan-bhendarkar
  - Test rescored from the saved test_feats + emb join (90 min).
  - 5,700,127 matches for 1,627,862 S1; validator `--check-ids` **PASS**; matches ⊆ candidates (5,700,127/5,700,127).
  - Empty rate FR 6.30% / IN 6.19% / US 5.77%; matches/S1 FR 3.150 / IN 3.242 / US 3.404.
- **C1** = submissions/files/20260926-2040_aryan-bhendarkar = C0 + France norm v2 (laneC, merged d26c5d6)
  - 24,240,674 France pairs re-featurized: base on norm_v2 (refeat_norm) + ctx on norm_v2 statistics (SplitContext test, norm 2); same model, same t.
  - France pairs outside the bge-m3 band stay NaN.
  - Validator **PASS**; matches ⊆ candidates (5,704,167/5,704,167).
  - **US/India rows byte-identical to C0 (0 of 1,473,092 differ)**; candidate_pairs.tsv byte-identical.
  - 28,101 of 259,452 France rows differ.
- France label-free before/after (C0 → C1):
  - matches/S1 3.150 → 3.166; empty 6.30% → 6.26%
  - S1 with a p ≥ 0.99 candidate 0.928 → 0.927
  - **band pairs/S1 (p 0.2–0.975) 0.821 → 0.698 (−15%)**
- expected LB: C0 ≈ 0.967 + gains from ps/emb (non-FR DM +0.0014 → ≈ +0.001); C1 adds the France normalization effect (lead's estimate 0.969–0.972).

## D1 Adversarial validation, val vs test (pipelines/adv_val.py; psemb features; pairs with p ≥ 0.5; 26 Sep 21:35 IST)
- US+IN: 296,826 val (mini) vs 300,000 test pairs. LightGBM 3-fold **AUC 0.797**: val and test pairs are clearly separable.
- top gain shares: mi_lfrac_cmax 0.206, ex_lfrac_cmax 0.164, ex_lfrac_cmin 0.086, mi_lfrac_cmin 0.044 (**lfrac = 50%**); then sib_n 0.040, emb_cos_full 0.034 (band NaN rate 0.94 vs 0.89), ex_ldf_max 0.034, s1_name_self 0.027.
- The lfrac quantiles are nearly identical between val and test, so the classifier uses the *exact* log(df/n) values as a split fingerprint. This is direct evidence for the memorization hypothesis (lead 19:00), consistent with EXP-A's brittleness.
- action: psemb without lfrac (tmux qNL): train + LOCO → DM-mini → DM-fold0x → D1 on the new feature set. Keep if DM/LOCO hold and the AUC drops.
- India-only: **AUC 0.799** (120k val vs 300k test). lfrac = 60% of gain (mi_lfrac_cmax 0.31); ex_ldf_max/min 0.10 (token df, also split-specific).
- US-only: **AUC 0.834** (177k vs 300k). lfrac = 47%; then sib_n 0.056, s1_name_self 0.049, sib_s1house_frac 0.033.

## EXP-NL psemb minus the 4 lfrac features (20260926-2135_…-psemb-nolfrac; DM 2200 / 2201; run 26 Sep 21:35–22:30 IST, logged 27 Sep 02:40)
- motivation: D1 (lfrac = 50% of val-vs-test separability).
- clean mini 0.9811 @ t = 0.70. DM-mini best 0.97995 @ t = 0.775.
- **DM-fold0x 0.98010 vs psemb 0.98101 (−0.0009)**; clean fold0x 0.98101 vs 0.98192 (−0.0009).
- LOCO IN→US 0.96408 (+0.0103 vs psemb), US→IN 0.92727 (−0.0066): the same direction-flip as B0a/B0b.
- post-drop D1 re-run crashed (polars collect error in adv_val on the dropped-feature set); not re-run, since it is not decision-relevant after the val drop.
- verdict: **KILL.** lfrac is split-fingerprinting but still carries in-domain signal; the LOCO transfer gain is not bidirectional.

## EXP-XS-A xenc stack: psemb + existing cross-fitted MiniLM logit as a feature (20260927-0234_…-ctx3-a; DM 0247 / 0249; 27 Sep 02:34–03:25 IST)
- change: psemb recipe + `xenc_minilm` (the amlc-xenc ckpts m0/m1, MIT, scored on Kaggle as amlc-xenc2-score) as a LightGBM feature. NaN where the pair was not scored (stage-1 p < 0.01).
  - Train rows use OOF logits (cross-fit halves, cf = s1_id.hash(42) % 2, verified 0 mismatches). es / val / test rows use the mean of both models.
  - Leak check: train-role OOF AUC 0.9932 = never-trained es rows 0.9934. Train coverage: 99.4% of positives; 100% of every OOF-prob band.
  - Stack method: GBDT feature (preferred). The sigmoid fallback was not needed.
- clean mini 0.9856 @ t = 0.775 (P 0.9980, R 0.9612; IN 0.9845, US 0.9865). DM-mini t_DM = 0.825.
- **paired vs psemb (0.775) at t_DM = 0.825:**

| | overall | India | US |
|---|---|---|---|
| **DM-fold0x** | **+0.00361 [+0.00341, +0.00383]** | +0.00548 | +0.00237 |
| clean fold0x | +0.00318 [+0.00301, +0.00338] | +0.00488 | +0.00206 |
| DM-mini | +0.00412 | +0.00605 | +0.00280 |
| clean mini | +0.00362 | +0.00546 | +0.00237 |

  - Absolute DM-fold0x 0.98441, clean fold0x 0.98511.
- LOCO IN→US 0.98379 / US→IN 0.98114: **not a transfer measure**. The xenc ckpts were trained on both countries, so the LOCO models get a feature fitted on the held-out country. Info only.
- verdict: **PASSES the model gate** (DM-fold0x ≥ +0.002, both countries up). The transfer risk for France is unmeasured: the ckpts never saw France. The earlier LOCO for an xenc trained on one country was −0.0015…−0.0026, but France is 15% vs +0.0036 on the 85%. Check the FR test diagnostics before the request.

## EXP-XS-B xenc stack minus the 4 ldf features (20260927-0257_…-ctx3-b; DM 0312 / 0313; 27 Sep 02:57–03:50 IST)
- change: A (psemb + xenc_minilm) with ex_ldf_min, ex_ldf_max, mi_ldf_min, mi_ldf_max dropped (Lane C transfer-safe list; Lane C LOCO for the ldf drop without xenc: +0.0022).
- clean mini 0.9856 @ t = 0.75 (IN 0.98455, US 0.98639). t_DM = 0.825.
- **paired vs psemb (0.775) at t_DM = 0.825:**

| | overall | India | US |
|---|---|---|---|
| **DM-fold0x** | **+0.00367 [+0.00347, +0.00389]** | +0.00549 | +0.00246 |
| clean fold0x | +0.00323 [+0.00306, +0.00343] | +0.00488 | +0.00214 |
| DM-mini | +0.00407 | +0.00598 | +0.00276 |

  - Absolute DM-fold0x 0.98447, clean fold0x 0.98516.
- LOCO (contaminated by xenc; the ldf-drop effect only, B − A): IN→US 0.98367 (−0.0001), US→IN 0.98156 (+0.0004).
- verdict: **KEEP. B is the submission candidate** (DM ≥ A and transfer-safer). C1-path test build started 03:52 (tmux bxB, logs/buildC1x.sh).

## SUB-XB C1-path test files for EXP-XS-B (27 Sep 03:52–04:50 IST)
- **XB** = submissions/files/20260927-0354_aryan-bhendarkar: B (psemb + xenc_minilm − ldf), t = 0.825, France rows on norm v2 (C1 path).
  - Validator `--check-ids` PASS; matches ⊆ candidates 5,755,308/5,755,308; 1,630,487 non-empty S1.
  - Splice check: non-FR rows identical to B's own full rescore.
- vs psemb-C1 (LB 0.968), label-free:

| country | matches/S1 | empty rate | rows changed | pairs added / removed per S1 |
|---|---|---|---|---|
| US | 3.404 → 3.371 | 5.77% → 5.80% | 11.7% | +0.051 / −0.084 |
| India | 3.242 → 3.308 | 6.19% → 5.95% | 12.9% | +0.105 / −0.038 |
| France | 3.166 → 3.239 | 6.26% → 5.93% | 21.9% | +0.163 / −0.090 |

  - The US net pruning and India net additions match LB-PROBE (US = band negatives, India = copies pushed into the band).
  - France churns most and is unmeasured (the ckpts never saw France).
- **HEDGE** = B for US/IN, France rows = psemb-C1 (the LB 0.968 file's France predictions); building (logs/bx_hedge_sub.log). It isolates the France effect of xenc.

## EXP-XS-C XB + Lane G MiniLM-5x as a second xenc feature (20260927-0449_…-ctx3-c; DM 0458 / 0459; 27 Sep 04:49–05:40 IST)
- change: XB (psemb + xenc_minilm − ldf) + `xenc_mlm5x`.
  - Lane G MiniLM (MIT) trained on 8M pairs, Ditto-style "[COL] name [VAL] … [NUM] house" on norm_v2, light augmentation.
  - Cross-fit halves md5(s1_id) % 2; verified every train-role pair is in exactly one file with 0 in-sample; es rows = mean of both, val/test = mean.
- clean mini 0.9859 @ t = 0.825; t_DM = 0.825.
- **paired vs XB (0.825):**

| | overall | India | US |
|---|---|---|---|
| **DM-fold0x** | **+0.00035 [+0.00025, +0.00045]** | +0.00052 | +0.00025 |
| clean fold0x | +0.00031 [+0.00021, +0.00039] | +0.00045 | +0.00022 |
| DM-mini | +0.00046 | +0.00069 | +0.00031 |

- verdict: **below the +0.001 keep gate** (a significant but small gain, both countries up). D (mlm5x replacing minilm) is pending.

## EXP-XS-D XB with Lane G MiniLM-5x replacing the old MiniLM (20260927-0508_…-ctx3-d; DM 0518 / 0519; 27 Sep 05:08–05:55 IST)
- change: psemb + `xenc_mlm5x` (no `xenc_minilm`) − ldf.
- clean mini 0.9858 @ t = 0.75; t_DM = 0.825.
- **paired vs XB (0.825):**

| | overall | India | US |
|---|---|---|---|
| **DM-fold0x** | **+0.00038 [+0.00026, +0.00051]** | +0.00044 | +0.00035 |
| clean fold0x | +0.00032 [+0.00021, +0.00042] | +0.00037 | +0.00028 |
| DM-mini | +0.00030 (n.s. per country) | | |

- verdict: **below the +0.001 gate. XB stays the candidate.** C ≈ D ≈ +0.0004. The new MiniLM-5x alone matches the old MiniLM + a small gain, so the two cross-encoders are largely redundant.
- Note for the lead: D's cross-encoder was trained on norm_v2 text with augmentation, so it may transfer to France better than the old MiniLM (which XB uses). That is unmeasurable label-free; D is a reasonable swap if the lead weighs France robustness over the n.s. difference (D − C ≈ 0).

## EXP-XS-E XB + Lane G XLM-R as a second xenc feature (20260927-0651_…-ctx3-e; DM 0659 / 0700; 27 Sep 06:51–07:35 IST)
- change: XB + `xenc_xlmr`: xlm-roberta-base (MIT), two cross-fitted halves delivered as separate dirs, joined with '+'.
  - Verified: every train-role pair in exactly one file, 0 in-sample, es rows in both.
- clean mini 0.9859 @ t = 0.725; t_DM = 0.825.
- **paired vs XB (0.825):**

| | overall | India | US |
|---|---|---|---|
| **DM-fold0x** | **+0.00029 [+0.00019, +0.00040]** | +0.00041 | +0.00022 |
| clean fold0x | +0.00022 [+0.00013, +0.00030] | +0.00032 | +0.00015 |
| DM-mini | +0.00024 (US +0.00005 n.s.) | | |

- verdict: **below the +0.001 gate; not kept.** C (+mlm5x) +0.00035, D (mlm5x instead) +0.00038, E (+xlmr) +0.00029: the cross-encoders are largely redundant given XB.
- next: F = XB + mlm5x + xlmr (all three), a final complementarity check.

## EXP-XS-F XB + MiniLM-5x + XLM-R (all three xenc; 20260927-0709_…-ctx3-f; DM 0718 / 0719; 27 Sep 07:09–07:50 IST)
- clean mini 0.9859 @ t = 0.75; t_DM = 0.825.
- paired vs XB: **DM-fold0x +0.00035 [+0.00024, +0.00046]** (IN +0.00048, US +0.00026); clean fold0x +0.00029.
- verdict: **not kept.** No complementarity: F ≈ C ≈ D ≈ E ≈ +0.0003–0.0004. The first cross-encoder (in XB) captured the xenc signal; extra xenc models add ~+0.0004 at most.
- Summary for the lead: the gate (+0.001 vs XB) is not reached by any Lane G combination. XB / HEDGE remain the candidates.
  - If a Lane G model is wanted for France robustness, D (MiniLM-5x on norm_v2 text with augmentation, replacing the old MiniLM; +0.00038 in-domain) is the natural swap, but its France effect can only be measured on the LB.

## EXP-BAG-XB Seed bag x5 of XB (20260927-0806_…-ctx3-xbbag; DM 0838 / 0839; 27 Sep 08:06–09:15 IST)
- change: members = XB (seed 42, reused: deterministic) + seeds 43..46 (LightGBM derives bagging/feature_fraction seeds from `seed`). Probabilities averaged (`ber.bag`; bag.json). best_iter 543 / 606 / 500 / 552.
- clean mini 0.9858 @ t = 0.75; t_DM = 0.825.
- **paired vs XB (0.825):** DM-fold0x **+0.00005 [−0.00002, +0.00012]** (IN +0.00007, US +0.00004; n.s.); clean fold0x +0.00002; DM-mini +0.00016.
- verdict: **not kept** (bar: +0.0005). The single deterministic LightGBM already has negligible seed variance at this data size.

## EXP-BAG-D Seed bag x5 of D (20260927-0905_…-ctx3-dbag; DM 0937 / fold0x run after it; 27 Sep 09:05–10:05 IST)
- members = D (seed 42) + seeds 43..46 (best_iter 607 / 648 / 492 / 515). clean mini 0.9859; t_DM = 0.825.
- **paired vs D (0.825):** DM-fold0x **+0.00002 [−0.00005, +0.00009]** (IN −0.00004, US +0.00006; n.s.); clean fold0x −0.00002; DM-mini +0.00002.
- verdict: **not kept.** Seed bagging is a no-op for both XB and D.
- test-build pick (task 3): **D** (20260927-0508). Best DM-fold0x of {XB, XB-bag, D, D-bag}: D 0.98485 vs XB 0.98447 (+0.00038 [+0.00026, +0.00051], both countries up). Its cross-encoder was trained on norm_v2 text with augmentation.

## SUB-D C1-path test files for D + HEDGE (27 Sep 10:05–10:50 IST)
- **D-C1** = submissions/files/20260927-1037_aryan-bhendarkar: run 20260927-0508 (psemb + xenc_mlm5x − ldf), t = 0.825, France rows on norm v2.
  - Validator `--check-ids` PASS; matches ⊆ candidates 5,761,349; splice non-FR identical.
- **D-HEDGE** = …/20260927-1043_aryan-bhendarkar: D-C1 US/IN + psemb-C1 France rows (byte-verified: 0 / 0 differ). Validator PASS; 5,736,610 matches.
- per country (matches/S1, empty rate):

| country | D-C1 | XB | psemb-C1 (LB 0.968) |
|---|---|---|---|
| France | 3.261 / 5.83% | 3.239 / 5.93% | 3.166 / 6.26% |
| India | 3.309 / 5.96% | 3.308 / 5.95% | 3.242 / 6.19% |
| US | 3.370 / 5.80% | 3.371 / 5.80% | 3.404 / 5.77% |

- churn, D-C1 vs XB: US 2.5% of rows (±0.013 pairs/S1), IN 2.4%, **FR 14.5% (+0.092 / −0.070 per S1)**.
  - D differs from XB mainly in France. LB(D) − LB(XB) ≈ the France effect of the norm_v2/augmented MiniLM-5x vs the old MiniLM, plus ~+0.0003 US/IN.

## LC Learning curve on the XB feature set (lead 08:05 task 2; no new featurization; 27 Sep 10:49– IST)
- same recipe and features as XB; training S1 subsampled (`--n-train-s1`; es role subsampled alike); xenc NaN where unscored as before; t re-tuned on DM-mini (frozen ρ/w).
- **25% (45k S1)** (20260927-1049_…-lc4500; DM 1053 / 1054): clean mini 0.9847.
  - Paired vs XB (100%): **DM-fold0x −0.00112 [−0.00124, −0.00099]** (IN −0.00116, US −0.00109); clean fold0x −0.00091.
- **50% (90k S1)** (20260927-1100_…-lc9000; DM-mini 1104): clean mini 0.9853.
  - Its DM-fold0x run 1106 was interrupted (the chain's tmux session ended at 11:07 when the lead launched the F build). Re-run as DM-fold0x 20260927-1138_…dm-val-fold0x.
  - Paired vs XB: **DM-fold0x −0.00054 [−0.00064, −0.00044]** (IN −0.00060, US −0.00050); clean fold0x −0.00048.
- **curve (DM-fold0x): 45k 0.98335 → 90k 0.98393 → 180k 0.98447.**
  - Gain per doubling: **+0.00058 (45→90k), +0.00054 (90→180k)**, i.e. about +0.0005–0.0006 per doubling and diminishing only slowly.
  - Extrapolation: 4× (720k S1) ≈ +0.0011; all ~1.76M fold 2–4 S1 (≈ 3.3 doublings) ≈ +0.0017.
  - That needs new blocking + ctx + bge-m3 + xenc scoring for the added S1: several hours of CPU + GPU. Not started (lead instruction).

## SUB-F F C1 + F-HEDGE (lead's bldF, 11:07–11:40 IST; F-HEDGE completed by amlc-49)
- F-C1 = submissions/files/20260927-1131_aryan-bhendarkar (built by the lead's chainBuildF; validator PASS; 5,758,122 matches ⊆ candidates; FR matches/S1 3.241).
- The chain aborted before F-HEDGE: `logs/buildC1x.sh` was edited during the run and the new line 12 `: keepalive kept (lead)` had unquoted parentheses (syntax error). Fixed to `: 'keepalive kept (lead)'`, same intent.
- **F-HEDGE** = …/20260927-1138_aryan-bhendarkar, built with the lead's exact commands (hedge_fr + make_submission). Validator PASS; byte-verified: France rows = psemb-C1 (0 differ), US/IN = F-C1 (0 differ); 5,738,594 matches.

