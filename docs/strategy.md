# Strategy (living document — update when a decision changes)

## Pipeline shape
```
raw → normalize (v0 → learned maps) → BLOCKING (union of retrievers, per country)
    → candidate pairs (s1_id, cand_id)  ← this exact set = candidate_pairs.tsv
    → pair features → LightGBM (stage 1) → [optional cross-encoder on uncertain band]
    → stage-2 context features (ranks, margins, S2↔S3 consistency) → calibrated prob
    → assign each record to its best S1 → per-S1 decision (threshold / expected-F0.5)
    → matching_results.tsv
```

## Blocking (the recall ceiling)
Target: **pair recall ≥ 0.99 on `mini` at ≤ 40 candidates per S1** (report per country).
Union of:
1. **Char-3gram TF-IDF on the normalized core name**, top-k per S1 within the country (sparse matmul in chunks).
2. **Address keys**: (house no., first street token) and (postcode), within the country.
3. **Compact-name key** for domains/handles (`energyvrtextile`) and alias names ("formerly known as").
4. **Dense retrieval** (fine-tuned multilingual MIT/Apache bi-encoder + FAISS). This catches native-script names and heavy typos.
5. **Record-centric reverse retrieval**: for each S2/S3 record, the top-3 S1s. This fits the at-most-one-owner structure and supplies competition features.

## Matching features (country-agnostic!)
- Name similarity on several views (full, core, compact, alias): ratio, token_set/sort, partial, Jaro-Winkler, char-3gram cosine, embedding cosine.
- **Difference features** (these are what separate the hard negatives):
  - extra/missing core tokens (count and IDF-weighted)
  - whether the extra token is a business word (Holdings/Partners/Exports…)
  - legal form equal / compatible / missing
- Address features:
  - house-number relation: equal / numeric diff / one is a prefix or suffix of the other (407 vs 07) / missing
  - street similarity, secondary numbers (unit/PMB), state/postcode equality or missing
- Record meta: source (2/3), name kind (domain/handle/empty), script (native vs Latin), empty address, token counts.
- **Competition/context features**:
  - rank of this candidate within the S1
  - rank of this S1 within the record
  - prob margin to the second-best S1
  - how many S1s retrieved the record
  - S2↔S3 agreement: a candidate similar to other high-confidence matches of the same S1

## Validation protocol
- Folds come from md5(s1_id) mod 5; eval = fold 0. `mini` (~88k S1) is the default, `micro` (~9k) is for smoke tests only, and `fold0` (~440k) is for final confirmation.
- Queries = eval S1s. **Candidate pool = the full train S2/S3 of that country**, which preserves distractor density. Models train on folds 1–4 only.
- Caveat: at eval time only eval S1s compete for records, so "assign to best S1" is slightly weaker than on test. For final numbers, score all S1s (OOF) so that competition is complete.
- **France proxy = leave-one-country-out**: train on US only and evaluate India (and the reverse). Features that collapse under LOCO will hurt France.
- Threshold / decision tuning: tune on the eval subset, and confirm on a disjoint subset (fold0 minus mini) to limit optimism.

## Submission plan (15 total)
- Day 1 (25 Sep): 1–2 submissions. The baseline end-to-end validates format + the val↔LB relationship. Record the gap.
- Day 2 (26 Sep): 2–3 submissions, only for major val gains (blocking v1 + features v1, cross-encoder).
- Day 3 (27 Sep): 2–3 submissions: best model; best + conservative France variant. **Freeze features by 18:00 IST**, final full run, package the zip by 22:00 IST. Keep 2 submissions in reserve.

## Compute plan (updated 25 Sep: cloud-first, see infra/aws/README.md)
- EC2 r7i.xlarge per member (32 GB) for all dev. Resize to 2xlarge/4xlarge for full test runs. g6.xlarge (L4) for encoders. Kaggle 2×T4 as the GPU fallback.
- Laptop: editing + micro smoke tests only (~5 GB free RAM).
- SageMaker: full train/test blocking + features (ml.m5.4xlarge / r5.4xlarge, 64–128 GB), GPU embedding/cross-encoder inference (ml.g5.xlarge/2xlarge, A10G 24 GB). Every member has their own credits, so parallelize across accounts.
