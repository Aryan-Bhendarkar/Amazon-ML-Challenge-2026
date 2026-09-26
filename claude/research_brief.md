# Research request: Amazon ML Challenge 2026, Business Entity Resolution

You are an expert in entity resolution, record linkage and ML competitions. We are a 4-student team. Our current public leaderboard score is **0.967**; the top of the leaderboard is **~0.987**.
The deadline is **27 Sep 2026, 23:59 IST (~36 h left)**, with a feature freeze ~6 h before that.

**What we need from you:** independent research on modern papers, Kaggle/competition write-ups, and ER libraries, turned into **concrete, ranked techniques** that can close the gap. The constraints below are hard. For each idea, say:
- which failure mode in section 6 it targets
- the evidence (a paper or write-up)
- expected gain and confidence
- compute cost at our scale
- risk on an unseen country

Don't repeat ideas listed in section 5 unless you have a better variant.

---

## 1. Problem
- There are three sources of business records. Every record has only 4 fields: `entity_id`, `business_name`, `business_address`, `country`. There are no shared IDs, phones, tax IDs, websites, or coordinates.
- **Source 1 (S1)** is a deduplicated reference. For every S1 entity, predict the set of S2/S3 record IDs that are the same real business (0..n).
- **The structure is a star.** Every S2/S3 record belongs to **at most one** S1 (verified on 7.6M training pairs). So this is an assignment problem, not transitive clustering.
- **Metric:** **macro-averaged F0.5 per S1 entity** (precision weighted 2× recall).
  - A true singleton predicted empty scores 1.0, and any prediction on it scores 0.
  - A non-singleton predicted empty scores 0.
- **Scale:**
  - train: 2.21M S1 + 10.32M S2/S3 records (US, India)
  - test: 1.73M S1 + 9.97M S2/S3 records (US, India, **France: 15% of test S1, zero labels**)
- **Submission:**
  - `matching_results.tsv`
  - `candidate_pairs.tsv`: must be exactly the pair set we scored, and final matches ⊆ candidates
  - both are audited, along with our code and a methodology document
- The public leaderboard is a subset of test; final ranking uses the private remainder.

## 2. Data characteristics (verified)
- Train S1 match counts: 0 matches = 5.6%, mode 3, mean 3.5, max 11. About 25–27% of training S2/S3 records match nothing.
- **Adversarial distractors:** near-copies of a real S1 with an extra business word ("Anurag Academy **Holdings**", same address), or a shifted house number (31 vs 32). True matches also contain number typos (407→07) and added words.
- **Name noise:**
  - typos, word reordering, legal-suffix swaps (Pvt Ltd / Private Limited / LLC / Inc)
  - 7+ Indic scripts with transliteration variants
  - junk tokens (`--`, `>>`, `(ID: 96415)`, `#70318`), domains, @handles, injected accents, "formerly known as"
- **Address noise:** reordering, state abbreviation ↔ full ↔ native script, `<NULL>`, missing numbers, PMB/Unit additions, city variants. About 3.4% of records have an empty address.
- Chains and generic names: ~50% of S1 share their exact core name with another S1.
- **France (test only):**
  - SARL/SAS/SASU/EURL/SCI/SNC/S.A.R.L., "(France)" insertions
  - `R.`/`BD.`/`ALL.`/`N°`/`Bis`/`Ter`, department vs region naming
  - dense cities with generic names ("Association", "Club")
  - shared addresses in ~19% of records (vs 6–7% in train)
- **NEW, important: a test distribution shift.**
  - Records per S1: train 4.67 (US and India alike); test 5.76 US, 5.82 India, 5.53 France.
  - Our predicted matches per S1 on test (3.3) are the same as on validation, so the extra ~1.1 records per S1 match no S1.
  - The numbers are consistent with the test set being built at train scale (~2.1M S1) and then **~19% of S1 being removed while their S2/S3 records were kept** ("orphans").
  - Orphans of chain branches and co-located businesses look like valid matches for the remaining similar S1. In training, their true owner S1 always competed for them; in test, that competitor is missing.

## 3. Hard constraints
1. Pretrained models must be **MIT or Apache-2.0 licensed and ≤ 8B params** (e.g. Multilingual-MiniLM, XLM-R, mDeBERTa-v3, multilingual-e5). No Llama/Gemma/Qwen, nothing non-commercial.
2. **No external data:** no geocoders, registries, gazetteers, postcode lists, or ER APIs. Hand-written normalization dictionaries are allowed.
3. `country` must not be a model feature (the set is open; France is unseen). Country-keyed normalization rules need a generic fallback.
4. **No leakage.**
   - Labels only from training folds; out-of-fold stacking.
   - Unsupervised statistics may be computed on test.
   - We do not tune on the leaderboard beyond a handful of submissions.
5. **Compute:**
   - one AWS CPU box, 8 vCPU / 61 GB (a second one is possible)
   - Kaggle 2× T4 (~30 GPU-h/week, ≤12 h per run)
   - AWS GPU quota is 0
   - a full test blocking pass takes ~2.5 h; full test featurization + scoring takes ~3.5 h; rescoring from saved test features takes minutes
6. 12 leaderboard submissions left (5/day).

## 4. Validation
- Folds = md5(s1_id) mod 5.
  - Eval on fold 0: `mini` 88k S1 for iteration; `fold0x` 353k S1 for confirmation.
  - Train on folds 2–4, early stopping on fold 1.
- Eval queries retrieve from the full country pool (same as test).
- Keep/kill by paired bootstrap on identical entities.
- **France proxy:** leave-one-country-out (train US → eval India and vice versa).
- **Val→LB gap:** −0.005 for our first two submissions, then **−0.014** for the latest. We attribute the extra gap to the orphan shift (section 2) plus France.

## 5. What we built and measured
| Stage | Val macro F0.5 | Public LB |
|---|---|---|
| Baseline: exact-key blocking + 28 rapidfuzz features + LightGBM + global threshold | 0.9057 | 0.901 |
| + post-rules (name-only uniqueness, co-location) | 0.9107 | 0.9046 |
| Blocking v1 (below) + same features | 0.9611 | – |
| + context features G1–G5 | 0.9800 (India 0.975, US 0.984; P 0.996, R 0.952) | – |
| + nkey_num retriever + density-robust context features v3 | **0.9813** (fold0x 0.9812) | **0.967** |

**Blocking v1** (Sparkly-style top-k per query, union capped at 100 per S1; pair recall **0.987**):
- TF-IDF char-3gram on the name + address-word TF-IDF, top-30 per S1 via sparse top-k
- exact normalized-address keys and house+street keys
- a phonetic consonant-skeleton key for transliterated names
- k=50 name-only retrieval for empty-address records
- rarest-2-name-token keys
- an exact-name-key + shared-number retriever
- a native-script→Latin token map learned from training pairs

**Features:**
- name/address similarities: token set/sort, JW, partial ratio, TF-IDF cosines
- house-number relations
- typed token edits: extra/missing core tokens and their document frequency; a hand-written business-word list, English + French
- S1-side uniqueness: how many S1 share this name / address (density-robust version)
- co-location flags
- sibling consensus: agreement with the S1's other high-probability candidates
- rank within the S1
- retriever-hit bits

**Tried and killed (measured):**
- **Set-level decision layer:** 4-fold OOF stage-1 → isotonic calibration → stage-2 LightGBM / cardinality (k*) model / per-S1 expected-F0.5 selection. Gain was only +0.0003 over a global threshold, even though the per-S1 prefix oracle is 0.992; 80% of the remaining loss is on entities where we only miss true matches.
- **Cross-encoder** (Multilingual-MiniLM-L12, MIT, trained on Kaggle T4s on hard negatives, AUC 0.9986) as a stage-2 feature: **+0.004 in-country** (confirmed on fold0x) but **−0.0015 to −0.0026 in the cross-country (LOCO) test**, so it's shelved because of France risk.
- **Already rejected:** tax/phone vetoes (no such fields), HAC/Leiden/correlation clustering (star structure), focal loss / class re-weighting (breaks calibration), graph embeddings, SPLADE, weighted MinHash, LLM-as-judge at 160M pairs.

**In progress:**
- a "name-twin competition" feature (another S1 sharing the candidate's name fits its address better), aimed at France templated same-name records on different streets
- French normalization rules
- an orphan-simulated validation and training regime (remove ~19% of S1, keep their records)

## 6. Remaining failure modes (where the next points are)
1. **Orphan / distribution shift on test (largest suspected loss, ~−0.008).** A record whose true S1 is absent gets assigned to the most similar remaining S1: chain branches, co-located businesses, templated names. The model learned to reject such records mainly through competition from their real owner, which test doesn't always have.
2. **France.** Unseen formats, dense cities, generic names, shared addresses; no labels, so we cannot measure it directly.
3. **Recall on hard true matches (in-candidate FN):** heavy name edits, native script, unrelated trade names at the same address. Blocking misses are ~1.3% of true pairs.
4. **Adversarial near-copies** (one extra business word, house ±1) that remain.

## 7. Questions for you
1. **Open-world / missing-entity matching:** how do we match against a reference set where some true owners are absent (open-set ER, "NIL"/unlinkable detection in entity linking, abstention)? What training-time simulations and features make a matcher robust when competitor entities are missing? Cite papers or production systems.
2. **Covariate / label shift without test labels:** which methods (prior-shift EM re-calibration, importance weighting, transductive threshold selection, self-training on high-confidence test pairs) are trustworthy for choosing the decision threshold under a changed distractor ratio? What are the risks?
3. **Zero-shot transfer to a new country/language (France):** which features and normalizations transfer? Are there self-supervised or transductive techniques that use unlabeled test pairs (e.g., learning abbreviation/synonym maps from high-confidence matches) without breaking the "no external data" rule?
4. **Cross-encoders that transfer across countries:** why might a cross-encoder help in-country but hurt cross-country? What training recipes (multi-country hard negatives, field-structured inputs, masking of country-specific tokens, calibration, distillation into GBDT features) fix that?
5. **Recovering the last recall** at fixed precision in ER pipelines with GBDT scoring. What did winners of Kaggle **Foursquare Location Matching (2022)**, **Shopee Product Matching (2021)** and similar POI/business dedup contests do beyond 2-stage GBDT? Which of their tricks fit data with only name + address and no coordinates?
6. **Macro per-entity F-beta optimization** with a one-owner-per-record constraint: is global assignment (min-cost flow / Hungarian with an abstain option) better than per-S1 thresholding when some owners are missing?
7. **Anything else** from 2023–2026 ER literature (LLM-distilled matchers, Ditto/HierGAT successors, AnyMatch, Unicorn, zero-shot ER benchmarks) that is feasible within the license and compute limits above.

**Output format:** a ranked list (by expected gain per hour of work). For each item give:
- method
- why it fits our failure mode
- evidence and links
- expected gain
- implementation sketch (≤ 5 steps)
- cost
- France/shift risk
- rule check

Skip anything that needs external data, non-permissive weights, or country as a feature.
