SUBMIT-REQUEST: none. No prediction-level change clears the gate (DM-fold0x ≥ +0.0005, CI > 0, no per-country drop). Ship D as built.

# AUDIT REPORT: run D (box2, branch `audit`, 27 Sep 2026)
Updated 10:50 IST. D = `20260927-0508_…-ctx3-d`, t = 0.825. Reproduced here exactly: DM-fold0x **0.98485** (draws 5), clean fold0x **0.98548**, DM-mini 0.98511.
Code: `pipelines/audit_d.py` (tables + DM scorer for arbitrary selections), `audit_errors.py` (part A), `audit_rules.py` (part C), `audit_hop2.py` (2-hop). Outputs are in `artifacts/audit_d/`.

## TL;DR (ranked by expected gain per hour)
1. **Nothing cheap is left at the decision layer.** Every threshold/segment/post-rule is ≤ +0.00003 on DM-fold0x. t = 0.825 is exactly the fold0x DM optimum (curve: 0.8 → 0.98495, 0.825 → 0.98497, 0.85 → 0.98488).
2. **The loss is 86% recall.** Ceilings on DM-fold0x if a bucket were fixed completely:
   - in-candidate rejects **+0.0070**
   - blocking misses **+0.0045**
   - FPs +0.0021
   - "lost to another S1" +0.0020

   The points toward 0.99 are in the *model* (hard true copies scored < t) and *blocking*, not in thresholds. Neither can be built and validated before 13:30 without test re-featurization on box1.
3. **Package gaps (must fix before the zip; see §4):**
   - no filled methodology document exists anywhere in the repo
   - REPRODUCE.md describes XB, not D
   - Lane G kernels + ckpt sha256 live on branch laneC (sha256 below)
   - requirements.txt pins unused packages from the Windows laptop

## 1. Part A: where D loses (fold0x, t = 0.825)
Per-S1 loss = 1 − F0.5, total DM loss 5,358 (mean 0.01516). Attribution: FP loss = F(no FP) − F; the rest is FN, split over the S1's missing true records by fate.

| bucket | DM % of loss | clean % |
|---|---|---|
| **FN side** | **86.2** | 90.0 |
| · in-candidate reject (true pair scored, p < t) | **44.8** | 46.8 |
| · blocking miss (true pair not in candidates) | **28.6** | 29.9 |
| · lost to another S1 (record's best S1 is a different eval S1) | **12.8** | 13.3 |
| **FP side** | **13.8** | 10.0 |
| · record owned by another S1 (not same name+addr) | 5.7 | 3.9 |
| · unowned near-copy: house ±k | 3.4 | 2.7 |
| · unowned near-copy: extra/missing word, same address | 2.7 | 2.0 |
| · unowned other | 1.5 | 1.1 |
| · chain/co-located twin (owned by other S1, same name+addr) | 0.5 | 0.4 |

- **Other slices (DM %):**
  - non-singleton predicted EMPTY (scores 0): 20.2
  - singleton with an FP: 3.2
  - FP by source: S2 6.4 / S3 7.4
  - FN by source: S2 39.0 / S3 47.2
  - FP with empty candidate address: 2.6
- **Per country, mean loss:** India 0.0165, US 0.0143. The India gap is recall.

**Patterns (5 examples per bucket in `artifacts/audit_d/fold0x_errors.json`):**
- **Lost to another S1:** almost all are **empty-address records** whose name matches several S1s in different cities (e.g. "Jai Enterprises Private Limited | ⟨empty⟩" → the Madras S1 instead of the Delhi one). Text cannot identify the owner, and their p is low, so they cost recall, not precision. This is not fixable from text.
- **In-candidate rejects:** many are empty-address records (p 0.2–0.5), heavy name edits, native script, and records with unrelated trade names at the same address.
  - Empty-address records: only 60% of positives are accepted, but exact-name rejects are only 25–37% positive, so the model's rejections are correct on average.
- **Blocking misses:** trade name ≠ legal name at the same address ("My Ventures Pvt Ltd" vs "5ynnovi" at 0119 Jyotinagar), missing addresses, typo'd names. A record-side address-only retriever would be needed (≈ +0.0045 ceiling, test re-blocking ≈ 2.5 h: not today).
- **FPs:** near-copies with house ±k (482 vs 480 Big Creek Rd) and extra/replaced business words, plus records that belong to another S1 via a shared native-script legal name.

## 2. Part C: prediction-level rules (tuned on DM-mini, frozen on DM-fold0x, paired bootstrap)
| rule | tuned params | DM-fold0x Δ [95% CI] | clean Δ | per country | verdict |
|---|---|---|---|---|---|
| rescue empty S1 (top cand if p ≥ t2) | t2 = 0.8 (lower t2 hurts on mini) | +0.00002 [−0.00001, +0.00005] | +0.00002 | IN −0.00003 / US +0.00005 | KILL |
| per-source thresholds S2/S3 | 0.825 / 0.825 | 0 | 0 | 0 | KILL (global t already optimal) |
| empty-address threshold | 0.825 | 0 | 0 | 0 | KILL |
| candidate-count bucket threshold | no effect | 0 | 0 | 0 | KILL |
| 2nd-S1 margin (m < 0.3 → need p ≥ 0.96) | m 0.3, t_hi 0.96 | **+0.00003 [+0.00001, +0.00004]** | +0.00001 | IN +0.00006 / US +0.00001 | significant but 17× below the gate |
| non-Latin-script threshold (country-agnostic proxy) | t_nat 0.7 | 0.00000 [−0.00002, +0.00003] | +0.00002 | ≈0 | KILL |
| sibling rescue (S1 already matched; strong name+addr band rows) | t_lo 0.75, nt 100, at 95 | −0.00001 | +0.00002 | ≈0 | KILL |
| 2-hop record↔record (accept rejected rows near-identical to an accepted sibling / drop accepted outliers) | analysis only | no separation: rejected rows with hop_full ≥ 90 & p ≥ 0.6 are 0.71 positive, the same as p's calibration; accepted rows with low hop are 90–100% positive | – | – | KILL |

| **stage-2 stack** on D (LightGBM on mini labels: D logit, XLM-R logits [not in D], MiniLM-5x, 2-hop, margin, 2nd-S1 p, sims; 2-fold OOF threshold) | t 0.825 | **−0.00011 [−0.00022, +0.00001]** | −0.00013 [−0.00023, −0.00002] | IN −0.00014 / US −0.00008; LOCO US→IN −0.00031, IN→US −0.00050 | KILL (mini OOF already −0.00014: D absorbs these signals; gain dominated by margin and D's logit) |

| contested-record reassignment (record clears t for 2 S1 → give it to the 2nd S1 if that one is otherwise empty) | analysis only | only 467 contested rows on fold0x; argmax picks the true owner 85% of the time; for "2nd otherwise empty" the 2nd is right 1/38 | – | – | KILL |

**C4, India recall** (fold0x, per-S1 loss ×1e-4):

| | blocking miss | in-cand reject | lost to other S1 | FP |
|---|---|---|---|---|
| India | **63.7** | 61.2 | 17.4 | 22.9 |
| US | 29.8 | 72.4 | 20.7 | 19.5 |

- India's extra loss is **blocking** (2.1× US), not the threshold: its in-candidate rejects are lower than US.
- Only 6% of India's rejects are native-script records.
- In both countries, 59–67% of in-candidate rejects are **empty-address** records.
- The lever for India is blocking recall (native-script / trade-name retrieval), not a threshold proxy.

**Threshold robustness** (fold0x, DM-optimal t as the test/val band-density ratio rho varies; w re-derived):

| rho | best t | loss at t = 0.825 vs best |
|---|---|---|
| 1.39 (India estimate) | 0.8 | 0.00002 |
| 1.63 (used) / 1.80 (US) / 2.2 / 2.6 | 0.825 | 0 |
| 1.0 (test as clean as val) | 0.75 | 0.00019 |

→ t = 0.825 is safe for any plausible density; no per-segment threshold is needed.

Why rescue fails:
- Among S1s predicted empty, true singletons dominate every p band (fold0x p 0.5–0.65: 102 rescuable vs 233 singletons that would score 0).
- The 262 correct top candidates of empty non-singletons at p 0.5–0.825 cannot be separated from singletons without a singleton detector.

## 3. Part B: code audit (read-only)
| # | area | finding | severity | action |
|---|---|---|---|---|
| B1 | test candidate cache v1_n2 | all 1,732,544 test S1 present (0 with no candidates); 164.7M pairs, 0 duplicate pairs; 0 null similarity features; 0 null/NA S1 names; max 215 cands/S1 (nkey_num exempt from the 100 cap, by design) | OK | – |
| B2 | `assign_best_s1` tie-breaking | pandas default (unstable) sort → exact ties broken arbitrarily. Measured on fold0x: **0 exact ties** between best and 2nd S1 | OK (latent) | optional: `kind="stable"` |
| B3 | `scripts/make_submission.py` | records matched to > 1 S1, and matched pairs missing from candidates, only print a WARNING (exit code still 0 if the validator passes); the validator does not check the one-owner structure | low (splice_fr asserts uniqueness; predict_test assigns first) | make it a hard failure before the final zip |
| B4 | xenc parity | pairs never scored have NaN in both train and test, but the selection differs: train used v1_n1 OOF p ≥ 0.01, val/test use psemb p ≥ 0.01. **nkey_num-only positives (0.33% of positives): xenc coverage 0% in train vs 99% in mini/fold0x/test** | low (the model never saw a scored xenc on those; true-copy values are high → benign direction) | document in the methodology |
| B5 | France splice | France matches from test_n2fr (norm v2) replace only France S1; uniqueness asserted; candidate set unchanged (refeat keeps it) → candidate_pairs.tsv = v1_n2 test cache stays the scored set | OK | – |
| B6 | xenc test text | G3 text built on norm_v2 (France rules) → consistent with the C1 France path; US/IN identical to v1 | OK | – |
| B7 | Lane G poller | one-shot `.pulled` flag re-pulled the OOM-killed MiniLM v2 output; the lead used `amlc-g3-minilm_latest`. Stale dir renamed `…_STALE_v2_oom_pull` (README inside) | fixed / documented | make sure box1 joined `minilm_latest` (s3 share/g3logits/minilm5x) |
| B8 | val competition | one-owner assignment in val only lets **eval** S1 compete, but test has all S1 → the "lost to another S1" bucket (12.8% of val loss) is understated for test | known (monitor note 4) | none today |

## 4. Final package: what is missing
- **Methodology document (Documentation_template.md filled): MISSING.**
  - Only `student_resource/Documentation_template.md` (template) exists.
  - Draft written: `docs/METHODOLOGY_DRAFT.md` (branch `audit`, box2). Follows the template sections; team name/members + final LB to fill.
- **docs/REPRODUCE.md describes XB, not D:**
  - §3 needs `--xenc mlm5x` (MiniLM-5x) instead of `minilm`, and D's run id.
  - TODO 4 (Lane G sha256) values:
    - MiniLM-5x h0 `182e5c94e29243b7f4464291c25b01d372f21aef71b01efd2f4230d3b47209b9`, h1 `f63ed5ad964809a78cac96db14c0fb02871d4062a6bd8c4d887f16fd305aa1fb`
    - XLM-R h0 `0a06715eac8c30109a3562203b49c287fc8c74350d5b3a95cbe212888910e004`
    - XLM-R h1 `ac94b96c833a58e7fddf71006de3125dfd74470665519f06695abe45ff9d1c92`
  - MiniLM-5x training kernel: acc5 `darshanbagadeycce/amlc-g2-minilm` (2×T4, 128 min); scoring `amlc-g3-minilm` v3 (T4x2).
- **Lane G code is not in box1-main:**
  - `pipelines/xenc_g1_export.py`, `pipelines/xenc_g3_prep.py`
  - `kaggle/xenc_g2*.py`, `kaggle/xenc_g3*.py`, `kaggle/gen_g2_variants.py`, `kaggle/gen_g3_variants.py`
  - `kaggle/launch_lane_g.sh`, `kaggle/poll_lane_g2.sh`
  - `pipelines/refeat_norm.py` (France)
  - These live on branch `laneC` (pushed to origin) and must be in the zip.
- **requirements.txt:**
  - Frozen on the Windows laptop. `faiss-cpu`, `sentence-transformers`, `psutil`, `tqdm` are not imported anywhere in src/pipelines/scripts.
  - `transformers` is used only inside Kaggle kernels.
  - Re-freeze from the box `.venv` (Python 3.13.15) and add a Kaggle-image note (torch 2.10 / transformers from kernel logs).
- **candidate_pairs.tsv = scored set:** OK by construction (B5), and make_submission checks that matches ⊆ candidates.
