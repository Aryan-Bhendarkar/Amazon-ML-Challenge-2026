# ML Challenge 2026: Business Entity Resolution (methodology DRAFT, box2 audit, 27 Sep 2026)

> Draft built from the repo's run records and decision log. The team must fill in the name/members and confirm the
> final run id and LB figures before copying it into `Documentation_template.md` for the zip.

**Team Name:** [Team name]
**Team Members:** [members]
**Submission Date:** 27 Sep 2026

---

## 1. Executive Summary
We solve the task as **blocking → pairwise scoring → one-owner assignment → a global threshold tuned for macro F0.5**.
- **Blocking:** a multi-retriever candidate generator (TF-IDF on name char-3grams + address words, exact normalized keys, and a name-key + house-number retriever) keeps 98.7% of true pairs at ~95 candidates per S1.
- **Scoring:** one LightGBM model scores every pair from ~100 country-agnostic similarity, difference, context and competition features, plus two MIT-licensed neural signals: frozen bge-m3 embedding cosines and a cross-fitted, fine-tuned multilingual MiniLM cross-encoder.
- **Decision:** each S2/S3 record goes to its best S1 only, and the threshold is chosen on a density-matched validation that mimics the test's distractor density.

France (no labels) is handled with country-agnostic features, hand-written French normalization rules, and leave-one-country-out validation.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Structure:** every S2/S3 record matches at most one S1 (0 violations in 7.6M train pairs). S1 match counts run 0–11 (mode 3); 5.6% of S1 are singletons. About 25–27% of S2/S3 records match nothing, and many of those are **adversarial near-copies** of a real S1 (an extra business word, a house number ±k, a swapped legal suffix).
- **Noise:** 7+ Indic scripts, domains/handles as names, junk tags (`(ID: 96415)`, `#70318`, `>>`), injected accents, "formerly known as"/dba aliases, legal-form swaps; address component reordering, state abbreviation ↔ full ↔ native script, `<NULL>`, missing or typo'd numbers, PMB/Unit additions.
- **Test shifts** (label-free analysis):
  - test has 5.8 records/S1 vs 4.7 in train, and 1.6–2.6× more assigned pairs per S1 in the uncertain probability band (near-copy distractors / pruned owners);
  - **France** (15% of test S1, no labels) is 3× more co-located (19% of S1 share an address vs 6%) and 3–6× denser in generic name words (Association, Club, École);
  - French S1 addresses always end with a region, while pool records carry the region, the department, or neither.

### 2.2 Solution Strategy
**Approach Type:** Blocking + gradient-boosted pair classifier with stacked neural pair signals, then a one-owner assignment and a global threshold.
**Core Innovations:**
1. Context/competition features that transfer across countries: name/address uniqueness counts within the split, typed token edits, house-number relations, sibling consensus, and per-source competition ranks.
2. Density-matched validation (DM-val): clean validation re-weighted so that uncertain-band negatives per S1 match the test's label-free band mass; it selects the threshold.
3. A cross-fitted cross-encoder trained on 8M hard pairs; its train-row logits are strictly out-of-fold.
4. France-keyed normalization with a generic fallback (US/India byte-identical), validated label-free.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys / retrievers** (normalized text; learned native-script → Latin token map; anyascii transliteration, ISC license):
  1. TF-IDF cosine on name char-3grams (max_df 0.01, weight 0.35) + address word tokens (max_df 0.02, weight 0.65), top-30 per S1 (sparse top-k, `sparse-dot-topn`, Apache-2.0);
  2. exact normalized keys (sorted core-name tokens, sorted address tokens, house + street);
  3. `nkey_num`: exact sorted-unique name key + a shared address number (block cap 50).

  Candidates are capped at 100 per S1 by retrieval score (nkey_num is exempt). Blocking is per country, taken from the data (open set).
- **Candidate pairs generated:** test 164,745,367 pairs for 1,732,544 S1 (every S1 has candidates; 0 duplicate pairs). `candidate_pairs.tsv` is exactly this scored set, and final matches ⊆ candidates.
- **How we kept true matches:** pair recall is measured on every blocking change against the full train pool (validation S1 only as queries). Mini recall is 0.9871 (0.9815 → 0.9840 → 0.9871 across versions), and 0.18% of non-singletons have no true pair in the candidates. The remaining misses are mostly trade names unrelated to the legal name at the same address, or empty addresses.

---

## 4. Matching Model
**Features used (98, all country-agnostic; `country` is never a feature):**
- **Name:** token-set / token-sort / ratio / partial ratio on core names (legal forms and stopwords removed), Jaro-Winkler and partial ratio on compact names (matches domains/handles), alias ("formerly/dba") similarity, legal-form relation, extra/missing/common token counts.
- **Address:** token-set and ratio on full address, street-only token-set, house-number relation (same / leading-zero / prefix / suffix / differ / missing), number-set Jaccard, secondary-number and postcode relations, state relation (state/region codes; France: region/department → region).
- **Context (unsupervised, computed on the split being scored):**
  - G1/G2: how many S1 / pool records share the name, address, or house+street key;
  - G3: extra/missing token rarity (capped document frequency, density-robust), a hand-written generic business-word flag, prefix/subset relations, typo-vs-substitution;
  - G4: house-number digit edits;
  - G5: sibling consensus within the S1's candidate list;
  - per-S1 ranks and gaps, and per-source (S2/S3) competition ranks/gaps.
- **Neural pair signals (MIT, ≤ 8B):**
  - frozen `BAAI/bge-m3` cosines (name / address / full) for pairs in the uncertain band;
  - `microsoft/Multilingual-MiniLM-L12-H384` cross-encoder fine-tuned on Kaggle GPUs on 8.0M train pairs (all positives + hard negatives), cross-fitted on two md5 halves of the train S1, so every train-row logit is out-of-fold; validation/test get the mean of both halves.
- **Removed after audit:** 4 raw log-document-frequency features. They are density-sensitive and failed transfer checks, and removing them is neutral in-domain.

**Model type:** LightGBM binary classifier (lr 0.05, 127 leaves, min_data 100, feature/bagging fraction 0.8, early stopping on fold 1, seed 42, deterministic).
**Decision:**
1. Each S2/S3 record is assigned to its highest-probability S1 (the at-most-one-owner structure).
2. It is kept if p ≥ t.
3. t = 0.825 is chosen by maximizing macro F0.5 under **density-matched validation**, frozen from the mini set and confirmed on a disjoint 353k-S1 fold.

**Threshold selection method:** macro-F0.5 optimization on DM-val (above); per-country thresholds are never used.

---

## 5. Results & Error Analysis
- **Validation protocol:** md5 folds over train S1; queries = the held-out fold's S1 against the **full** train S2/S3 pool (preserves distractor density); models trained on the other folds; a leave-one-country-out (US ↔ India) score as a France proxy.
- **F0.5 (macro), final model D:**
  - mini (88k S1): 0.9858 clean (India 0.9847, US 0.9866; pair precision 0.998, recall 0.962)
  - fold0x (353k S1, disjoint): **0.98548 clean, 0.98497 density-matched** at t = 0.825
- **Public LB progression:** 0.901 (baseline) → 0.967 (blocking v1 + context features) → 0.968 (+ neural signals + France normalization) → [final].
- **Where the remaining loss is** (fold0x, DM): 86% recall, 14% precision.
  - in-candidate rejects 45%: hard true copies scored below t, e.g. empty-address copies, heavy transliteration;
  - blocking misses 29%;
  - records whose name matches several S1 in different cities while their own address is empty: 13%. Text cannot identify the owner.
- **Common false positives:** near-copies with a house number ±k (480 vs 482 Big Creek Rd); a generic business word added or substituted at the same address; records of a different S1 sharing a native-script legal name.
- **Common false negatives:** empty-address copies, trade name ≠ legal name at the same address, strong typos in both name and number.

---

## 6. Conclusion
A carefully validated blocking + GBDT pipeline with transfer-safe context features gets most of the way. The largest gains came from:
- better blocking recall;
- competition/uniqueness context features (+0.019);
- neural pair signals stacked as features;
- choosing the threshold under a test-like distractor density.

Lessons: measure everything on a validation set built to look like test (full-pool retrieval, density matching, leave-one-country-out). Neural cross-encoders help most when stacked into the GBDT with strictly out-of-fold logits.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/`: `ber/` (library: io, normalize, blocking, ctx_features, decision, metric, dmval, submission), `pipelines/` (blocking, features, training, DM-val, test inference, France splice, cross-encoder export/join), `scripts/` (data preparation, normalization caches, submission writer + official validator), `kaggle/` (GPU kernels).
Entry points and exact commands: `README.md` (from `docs/REPRODUCE.md`): prepare → normalize (v0/v1/v2) → blocking (cache v1_n2) → context features → Kaggle GPU steps (bge-m3 cosines; MiniLM cross-encoder train + score) → train LightGBM → DM-val threshold → test inference (+ France norm v2 splice) → `make_submission.py` (writes both TSVs via `ber.submission`, LF line endings; runs the official validator with `--check-ids`).

### B. Compliance
- **No external data:** no geocoding, registries, gazetteers, postcode lists or entity-resolution APIs. Unsupervised statistics are computed on the given test data only.
- **Hand-written dictionaries** (general knowledge, in code):
  - legal forms (US/India/France);
  - street-type abbreviations;
  - US and Indian state names/codes;
  - French regions (13 + pre-2016 names) and 96 departments → region;
  - French name rules: strip "(France)", cie/compagnie → co, et → and, frs → freres, st → saint, OCR typos 5arl/5as/5asu, legal form `ei`;
  - generic business words (G3 flag).
- **Learned from train labels only:** native-script → Latin token map (folds 1–4).
- **Models:**
  - `microsoft/Multilingual-MiniLM-L12-H384`: MIT, 117M params (tokenizer files from `FacebookAI/xlm-roberta-base`, MIT)
  - `BAAI/bge-m3`: MIT, 568M params, frozen
  - `FacebookAI/xlm-roberta-base`: MIT, 278M params, trained, not used in the final model
- **Libraries:** permissive (MIT/BSD/Apache/ISC; anyascii instead of the GPL unidecode).
- `country` is used only to select per-country normalization rules (with a generic fallback) and to split blocking; it is never a model feature.

### C. Checkpoint hashes (sha256)
- MiniLM cross-encoder (25 Sep): m0 `cbf19475…691d6`, m1 `54a248ea…fe96ac`
- MiniLM-5x (Lane G, used by D): h0 `182e5c94e29243b7f4464291c25b01d372f21aef71b01efd2f4230d3b47209b9`, h1 `f63ed5ad964809a78cac96db14c0fb02871d4062a6bd8c4d887f16fd305aa1fb`
- XLM-R (not in D): h0 `0a06715e…e004`, h1 `ac94b96c…1c92`
