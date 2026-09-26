# HANDOFF PROMPT: AMLC 2026 Business Entity Resolution, final push (26–27 Sep 2026)

You are the lead ML engineer on our 4-person team for the Amazon ML Challenge 2026 (business entity resolution). Work in this repo on the AWS box. A separate Claude (Cowork) monitors your progress through the files you write, so keep `claude/STATUS.md` and `claude/experiments.md` up to date (formats at the end).

**Goal:** raise our public LB score from **0.967** (val 0.9813) as far as possible toward ~0.987. The main target is the suspected **orphan distribution shift**. Every change is judged on a validation set built to look like test, not on the public leaderboard.

---

## 0. Before you write any code
1. Read `claude/rules_and_gotchas.md`, `claude/strategy.md`, `claude/cloud_setup.md`, `claude/research_brief.md` and any existing `claude/experiments.md`.
2. Map the actual repo:
   - the pipeline entry points
   - where the saved pair features and out-of-fold (OOF) scores live
   - how `assign_best_s1` and threshold tuning work
   - **how record-side competition features are computed at validation time**: which S1s compete for a record (eval S1s only, or the full pool?)

   Write a short map in `claude/STATUS.md`. Reuse the existing code and saved features wherever you can. Do not rewrite working pipelines.
3. Confirm the current best model and its run_id (val 0.9813 / LB 0.967). Every experiment below is compared against it with a **paired bootstrap on identical S1 entities** (1,000 resamples, report the mean Δ and 95% CI).

---

## 1. Problem summary (you need this to make good choices)
- Records have 4 fields: `entity_id, business_name, business_address, country`. There are no IDs, phones, coordinates or websites.
- S1 is a deduplicated reference set. For each S1, predict the set of S2/S3 records that are the same business (0..n).
- **Star structure:** every S2/S3 record belongs to **at most one** S1. So each record goes to its single best S1, or to none.
- **Metric: macro F0.5 per S1**, averaged over all S1s including singletons.
  - A true singleton predicted empty scores 1.0; any prediction on it scores 0.
  - A non-singleton predicted empty scores 0.
  - False positives cost roughly twice as much as false negatives.
- Scale:
  - train: 2.21M S1 + 10.32M S2/S3 (US, India)
  - test: 1.73M S1 + 9.97M S2/S3 (US, India, **France**: 15% of test S1, no labels)
- **Suspected shift ("orphans"):**
  - Records per S1 are 4.67 in train but ~5.76 in test.
  - This fits test being built at train scale, then **~19% of S1 being removed while their S2/S3 records were kept** (4.67 / 0.81 = 5.77).
  - Orphaned records of chain branches, co-located or templated businesses then look like matches for the remaining similar S1s.
  - In training, the true owner always competed for these records; in test, the owner is missing.
- Current state:
  - val 0.9813 (P 0.996, R ~0.952)
  - blocking pair recall 0.987
  - val→LB gap grew from −0.005 to −0.014
- Remaining failure modes:
  1. orphans
  2. France
  3. in-candidate false negatives (the largest recall bucket) plus 1.3% of pairs lost in blocking
  4. adversarial near-copies

## 2. Hard rules (never break these)
- **Pretrained weights:** MIT or Apache-2.0 only, ≤ 8B params. Check the license on the model card before using any weights and record it in `claude/experiments.md`. No Llama, Gemma or Qwen-licensed models.
- **No external data:** no geocoders, registries, gazetteers, postcode lists, ER APIs or internet augmentation.
  - Hand-written normalization dictionaries in code are fine; document them.
  - **Do not open or use other teams' competition code or repos** (e.g. any public "amazon-ml-challenge" repo). Our code is audited.
- `country` must never be a model feature. Normalization rules keyed on country need a generic fallback.
- **No leakage:**
  - Labels come only from training folds; stacking is out-of-fold.
  - Unsupervised statistics on test are allowed.
  - Never tune on the leaderboard.
- **Output format:**
  - header `source1_entity_id\tmatched_entity_ids`
  - comma-separated IDs with no spaces
  - empty rows still contain the tab (`S1-xxx\t`)
  - always write with `lineterminator="\n"`
  - read with `dtype=str, keep_default_na=False`
  - every S1 ID exactly once
- `candidate_pairs.tsv` must be **exactly the pair set that was scored**, and final matches ⊆ candidates. If any step adds candidates (step 4), they must go into this file.
- Run `utils/validate_submission.py` (with `--check-ids` on the box) on every submission file.

## 3. Submission policy (strict)
- **Today (26 Sep): only 2 submissions left.** Tomorrow (27 Sep): 5. Submit only when there is a real, validated improvement.
- **You never submit.** You prepare the file, validate it, and write a submission request in `claude/STATUS.md` with:
  - run_id and git commit
  - path to `matching_results.tsv`
  - clean-val, orphan-val and LOCO numbers vs baseline, with CIs
  - the expected LB direction

  A human uploads it from the single designated laptop.
- **Gate for submission #1 today:**
  - orphan-sim **fold0x** Δ ≥ +0.003 vs the current LB model (CI excludes 0)
  - clean fold0x Δ ≥ −0.001
  - LOCO Δ ≥ −0.0005
- **Gate for submission #2 today:** a further ≥ +0.002 on orphan-sim fold0x over submission #1, with the same side conditions. Otherwise hold it for tomorrow.
- Log every submission (run_id, commit, config, val numbers, LB score once known) in `claude/experiments.md`.

## 4. Timeline (IST)
- **26 Sep:** steps A–D; up to 2 submissions.
- **27 Sep:**
  - steps E–F (+ optional G), final stack
  - **feature freeze 18:00**
  - full test rescoring and validation by 20:00
  - final zip (`output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}`, documentation) by **22:00**
  - hard deadline 23:59
- Long jobs run in `tmux`. `touch ~/.keepalive` for unattended runs and remove it afterwards.
- Compute:
  - CPU box: 8 vCPU / 61 GB
  - Kaggle 2×T4 for GPU work
  - full test blocking ≈ 2.5 h; full featurize + score ≈ 3.5 h; rescoring saved features takes minutes, so prefer that
- Prefer experiments that reuse saved pair features.

---

## 5. Experiment plan (in order; each step has a keep/kill gate)

### EXP-A: Orphan-simulated validation + diagnosis (do first, ~1–2 h)
Build the evaluation that looks like test:
1. **Removal mask:** remove S1 where `md5(s1_id + "|orphan") mod 1000 < 190` (19%, reproducible). Also build a **biased variant**:
   - removal rate ×2 for S1s whose core name or normalized address is shared with another S1 in the same country
   - rate ×0.5 elsewhere
   - rescaled so the overall rate is still ~19%
2. Apply the removal to the **whole competing S1 pool**, not only to the eval queries. Use out-of-fold scores for non-eval S1s so that record-side competition is realistic. If the current code only lets eval S1s compete, document this and fix it in both clean and orphan validation.
3. Keep all S2/S3 records. Records whose owner was removed become unmatched (label 0 against every surviving S1).
4. Score only the surviving eval S1s. **Recompute all context and competition features after the removal:**
   - rank within the record
   - margin to the best other S1
   - number of S1s that retrieved the record
   - sibling consensus
   - S1-side name/address uniqueness counts
5. Run the **current** model on clean mini and on orphan mini (uniform and biased). Report:
   - macro F0.5 overall and per country
   - precision and recall
   - the share of lost F0.5 caused by FPs on true singletons, FPs on non-singletons, and FNs
   - how many FPs are orphan records
   - the simulated records-per-S1 ratio vs test's 5.76

**Decision:**
- Orphan drop ≥ 0.006: hypothesis confirmed; continue with B–D.
- Drop < 0.004: orphans are **not** the main gap. Write this in STATUS.md, skip to E (recall) and F (France), and flag it for human review.

From now on, the **primary metric is orphan-sim mini (uniform), confirmed on orphan-sim fold0x**. Clean val and LOCO are guardrails.

### EXP-B: Retrain on pruned pools (the main fix)
1. Build training data from folds 2–4 (early stopping on fold 1) using a **mixture of drop rates {0, 0.19, 0.30}**, half uniform and half biased removal. If memory is tight, use {0, 0.22}.
2. For each pruned view, recompute every competition/context feature on the pruned pool (the same code path as EXP-A). Subsample pairs if needed to fit in RAM.
3. Add two features that don't depend on competitors:
   - the absolute stage-1 probability with no rank normalization
   - the margin to the second-best S1 computed over surviving S1s only
4. Retrain LightGBM with the same hyperparameters. Evaluate on clean, orphan-uniform, orphan-biased and LOCO (train US → eval India, train India → eval US).
5. **Gate:** orphan-mini Δ ≥ +0.003 (CI excludes 0), clean ≥ −0.001, LOCO ≥ −0.0005. If it passes, confirm on orphan fold0x and rescore test from saved features. For test, compute the features exactly as in inference; no removal is needed because test is already pruned.
   → Candidate for **submission #1**.

### EXP-C: Record-record sub-cluster features (orphan clusters without their owner)
For each S1, take its top-k candidates by current probability (k = 10–12).
1. Compute record-to-record name and address similarity (rapidfuzz `token_set_ratio` plus char-3gram TF-IDF cosine) for all pairs inside that set.
2. Run single-link clustering at a high threshold. **Require both name and address similarity**, because French names are templated ("Association …", "Club …").
3. Add these per-candidate features:
   - same sub-cluster as the S1's top-1 candidate (bool)
   - size of the candidate's own sub-cluster
   - number of sub-clusters of size ≥ 2
   - mean similarity to its own sub-cluster vs to the top-1 sub-cluster
   - gap in S1-similarity between the two largest sub-clusters
4. Compute these on the pruned training views from EXP-B and retrain.
5. **Gate:** orphan-mini Δ ≥ +0.001 over EXP-B, guardrails unchanged. Test cost is about 1–1.5 h of featurizing the top-k sets only.

### EXP-D: Threshold + singleton guard on orphan-sim out-of-fold scores
1. Produce OOF probabilities on orphan-sim folds from the best model so far.
2. Re-select the global threshold to maximize orphan-sim macro F0.5. Rerun the existing isotonic + per-S1 expected-F0.5 selector on these OOF scores and keep whichever wins.
3. Add a **singleton guard**: when the predicted set size is 1, require p ≥ t₁ (> t) and a margin over the second candidate. Tune t₁ and the margin on orphan mini and confirm on orphan fold0x.
4. **Gate:** ≥ +0.001 on orphan-sim and neutral on clean.
   → B+C+D is the candidate for **submission #2** (only if it clears the §3 gate).

### EXP-E: Recall (27 Sep morning)
Most lost recall comes from in-candidate false negatives, so start there.
1. **2-hop sibling rescue:**
   - Anchors are pairs (A, r′) with p ≥ 0.98 where r′ is A's top-1.
   - For each record r whose best-S1 p < t, find its top-5 nearest records (the existing sparse char-3gram top-k over name + address).
   - If r is very similar to an anchor r′ **on name and address**, and r's best other S1 is below a low bar, create the candidate (A, r).
   - Score it with the main model plus two features: anchor similarity and anchor probability.
   - Require the anchor to pass the EXP-C sub-cluster check.
2. **Reverse retrieval:** for each unassigned S2/S3 record, retrieve its top-5 S1s (name char-3gram + address-word TF-IDF) and add only the new pairs.
3. **Multilingual embedding blocker (optional):** `minishlab/potion-multilingual-128M` (static embeddings, CPU; **verify the MIT license on the model card first**). Build name and name+address views and retrieve the top-20 S1s for low-confidence or no-candidate records. **Measure how many missed true pairs it recovers on mini first.** Keep it only if it recovers ≥ 30% at a reasonable candidate cost.
4. Add every new pair to the candidate set, and therefore to `candidate_pairs.tsv`.
5. **Gate:** each sub-step must be ≥ +0.0005 on orphan-sim (new candidates bring new orphan false positives), with guardrails unchanged.

### EXP-F: France (27 Sep, in parallel with E)
1. **Test-pool statistics:** recompute token document-frequency/IDF and S1 name/address density on each country's own test pool, so words like "Association" and "Club" become common in France.
2. **Normalization fixes** (hand-written, generic fallback, documented):
   - NFD accent-stripping plus explicit ligature maps `œ→oe`, `æ→ae` (NFD does not decompose these); make sure both sides of every comparison are normalized the same way
   - French street types: `r/r./rue`, `bd/bd./boulevard`, `av/av./avenue`, `all/all./allée`, `imp/impasse`, `ch/chemin`, `rte/route`, `pl/place`, `fg/faubourg`, `N°`, and `bis/ter` as house-number suffixes
   - in addresses: `St → saint`, `Ste → sainte`
   - in names: `Sté → société` (a legal-form word, **not** "sainte")
   - legal forms: SARL / S.A.R.L., SAS / S.A.S., SASU, EURL, SCI, SNC, SA; strip "(France)"
3. **One-pass alias mining from test (proposal only):**
   - Take test pairs with p ≥ 0.99 that are mutual-best.
   - Align their tokens and count differing-token pairs across distinct S1s.
   - Keep pairs with support ≥ 50 and high PMI.
   - Write the top 150–200 to `claude/review/alias_candidates.tsv` for **human review**. Do not apply them automatically.
   - Only approved rows go into the dictionary. One iteration only; no self-training loops.
4. Finish the in-progress **name-twin competition** feature.
5. Re-featurize the France pairs only (~15% of test). **Gate: LOCO Δ ≥ 0** and clean val neutral.

### EXP-G: Optional cross-encoder (Kaggle 2×T4, only if E and F are on track)
- Retrain MiniLM-L12 (MIT) on **both countries**, with hard negatives balanced so the rate of exact-address and exact-name agreement among negatives is similar across countries. Upsample co-located negatives toward a ~19% shared-address rate and add orphan-sim negatives.
- Use field-structured input: `[NAME] … [ADDR] …`.
- Use it **only as an add-rule** inside the band p ∈ [0.3·t, t) with CE ≥ 0.995. Never as a veto, never as a general feature.
- **Kill it if LOCO Δ < 0.**

### EXP-H: Hardening before the freeze
- Bag 5 seeds.
- Add monotone-increasing constraints on the core name/address similarity features.
- Run an adversarial check (classifier trained to separate val pair features from test pair features). Clip or drop features that separate val from test extremely well, but only if orphan-val is not hurt.

---

## 6. Do NOT do (already tested or judged low value)
- Hungarian / min-cost flow / global assignment. Each record takes its best S1 or none; this is already enough.
- EM / BBSE / GMM / Youden threshold recalibration on test scores. The orphan shift changes *which* negatives appear, so these can't see it.
- Test-time BatchNorm adaptation, Noisy Student, iterative pseudo-labeling, a larger mDeBERTa cross-encoder, LLM judges.
- Fellegi–Sunter LLR features (LightGBM already learns these), more fuzzy-string variants, clustering (HAC/Leiden), focal loss, class re-weighting.
- Any feature using coordinates, phones or geography lookups (we have none, and external data is banned).

## 7. Final model selection (27 Sep, 18:00–20:00)
- Pick the model with the best **orphan-sim fold0x** score that is neutral on clean fold0x and LOCO. The public LB is a sanity check only; don't pick by LB.
- Regenerate both output files from one run and validate:
  - `candidate_pairs.tsv` equals the scored set
  - matches ⊆ candidates
  - `validate_submission.py --check-ids` passes
- The code must regenerate both outputs from the raw train/test data, with pinned `requirements.txt`.
- Update the methodology document. Explicitly cover:
  - orphan-simulated training and validation
  - all hand-written dictionaries
  - the human-reviewed test alias proposals
  - model licenses

---

## 8. Reporting formats (the monitoring Claude reads these)

**`claude/experiments.md`**: append one entry per experiment:
```
## EXP-<id> <short name> (run_id, git commit, date/time IST)
- change: ...
- clean mini / fold0x: x.xxxx (Δ ±, 95% CI)
- orphan-uniform mini / fold0x: ...
- orphan-biased mini: ...
- LOCO US→IN / IN→US: ...
- per-country, P / R: ...
- verdict: KEEP / KILL / NEEDS-REVIEW + one-line reason
- test cost: ...
```

**`claude/STATUS.md`**: overwrite it after every step with:
```
Updated: <IST time>
Current best: <run_id>, orphan fold0x X, clean fold0x Y, LOCO Z
Running now: <job, tmux session, ETA>
Next: <step>
Blockers / needs human: <e.g., alias review, submission request, decision after EXP-A>
Submission requests: <none | run_id + path + numbers + gate check>
Submissions used: today k/2, tomorrow k/5
```

If a gate result is borderline, a decision is ambiguous, or something breaks a rule, **stop and write it under "needs human"** rather than guessing.

Start now with §0, then EXP-A.