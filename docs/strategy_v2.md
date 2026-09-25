# Strategy v2: problem → solution map to reach 0.98+ (25 Sep, 15:30 IST)

Evidence comes from `experiments/runs/20260925-1236_aryan_baseline-v0-keys-lgbm/error_analysis.md`, measured on mini. Loss = 1 − 0.9057 = 0.094:

| Source of loss | Macro loss |
|---|---|
| blocking misses | 0.070 |
| in-candidate FN | 0.016 |
| FP | 0.011 |

Best-top-k oracle with the current ranking: **0.930**. The ranking inside each S1 is mostly right; the per-S1 *cutoff* is wrong.

Leaders are at ~0.985, which means they have near-perfect recall AND precision. That is only possible with (1) ~99% blocking recall, (2) a decision layer that knows *how many* matches each S1 has, and (3) features that resolve the four ambiguity families below. **No leakage tricks** (no ID/row-order exploitation, no test labels): the private LB is what counts.

---

## P1. Blocking recall 0.86 → 0.99 (worth up to +0.07)
**Inspiration:** Sparkly (VLDB 2023) showed that a plain TF-IDF/BM25 blocker on character 3-grams, taking the **top-k per query (not a threshold)**, beats deep blockers on large dirty data. IDF is what matters; TF barely does. Concatenating the identity attributes works well.

**Retrievers** (each logs its marginal recall; the union is capped at ≤100 per S1 by a cheap score):
1. **BM25 / TF-IDF 3-gram on `n_core`**, per country, k≈30. Handles typos, word swaps, dropped words.
2. **3-gram on `n_core + a_street`**, k≈30. Handles generic names and chains.
3. **Exact normalized-address key** (sorted `a_full` tokens incl. house number, and a house+street-token key). This catches the **"unrelated name at the identical address" positives** (DBA/trade names; FN4 shows they are real).
4. **Name-only records (empty address, 3.4% of S2/S3):** only name retrievers apply, so give them k≈50 on name.
5. **Phonetic consonant-skeleton key** for native-script records:
   - anyascii, then lowercase, drop vowels, then merge `ph→f, bh→b, kh→k, gh→g, th→t, dh→d, sh/s→s, v/w→v, z/j→j`
   - key = skeleton of the first two core tokens

   "blu impeks praivet" and "blue impex private" collapse to the same key. It is cheap, and the transliteration literature says the same thing: phonology beats orthography across scripts.
6. **Reverse retrieval** (record → top-3 S1 over ALL S1 of the country). This fits "each record has one owner" and yields honest competition features.

**Gate:** pair recall ≥ 0.98 on mini, and **zero-candidate non-singletons < 0.3%** (they are 3.4% today and cost 0.039).

## P2. The per-S1 cutoff: "how many matches does this S1 have?" (worth up to +0.024 at today's ranking)
Out-of-the-box: treat this as **set prediction**, not independent pairs.
- **Cardinality model:** an S1-level LightGBM regressor that predicts the number of true matches `k̂` from:
  - the S1's candidate-prob profile: sorted top-10 probs, gaps between consecutive probs
  - counts above thresholds, per source (S2 vs S3 separately; true clusters usually span both sources)
  - S1 name/address uniqueness (P3)

  Output the top-`k̂` (or expected-F0.5 over the calibrated probs, with `k̂` as a prior). Train it on OOF probs from folds 2–4, early stopping on fold 1, eval on mini.
- **Singleton head:** P(no match) for the S1. Predicting empty is worth 1.0 per true singleton, and a singleton FP costs 1.0.
- **Relative-rank features** in stage 2: rank within the S1 and within the source, gap to the S1's best, and gap to the next candidate. The error analysis says probabilities aren't comparable across contexts; relative features fix exactly that.

## P3. Name-only records and name collisions (worth ~0.009)
About 50% of S1s share their exact core name with another S1. A name-only candidate is only resolvable via **S1-side uniqueness**:
- `s1_same_name` counts over ALL S1 of the split and country (unlabeled, so it's legal)
- `n_key_equal`
- the uniqueness flag × `cand_addr_empty`

Already measured as rules: **+0.0044 (CI +0.0041…+0.0048)**. As model features, expect +0.005–0.008.

## P4. Co-location (same address, different business) (worth ~0.004)
- `s1_same_addr` (how many S1 share this normalized address) and `a_key_equal`.
- Rule B measured +0.0007.
- **Competition-aware assignment:** a record goes to the S1 that dominates on name+address. Validation must include competitor S1s (all train S1 that share a key with a mini candidate), otherwise val is easier than test (55% of FPs are owned by non-mini S1).

## P5. Adversarial distractors: business-word edits + house ±k (worth ~0.010, partly irreducible)
- **Token-level edit typing:**
  - extra/missing core token counts
  - the extra token's S1 document frequency (per country; generic "services" vs rare)
  - whether the S1 name is a *prefix* of the candidate name
  - substitution vs insertion
- **House-number relation features:**
  - digit edit distance, absolute and relative numeric diff
  - same length, first-digit vs last-digit change, leading-zero only
  - unit/plot agreement
- **Leave-one-out consensus:** does this candidate's house number / token set agree with the S1's *other* high-prob candidates? True copies carry independent noise; a distractor deviates from both the S1 and its siblings.

## P6. Native script (India, worth ~0.004 now and more after blocking)
- The consonant skeleton (P1.5) as both a blocking key and a feature: skeleton similarity of names.
- A learned token map from folds 1–4 pairs: `praibhet/piraivet/limirrd → private/limited`, and state names in 7 scripts.
- Normalization cleanup: honorifics and fuzzy legal forms (measured upper bound +0.0005).

## P7. France (15% of test, no labels): protect it, don't gamble
- The "country-specific address patterns" tip applies here. Write a French rule set:
  - rue/r., bd, av, all., imp, ch, rte, n°, bis/ter
  - SARL/SAS/SASU/EURL/SCI/SNC with dotted variants
  - drop "(France)"
  - department↔region map for the ~15 test cities
- France addresses are shared much more often (19% vs 6–7%), so co-location features must use ==1 flags, which is the precision-safe direction.
- Every feature group needs a LOCO check (US→India and India→US), plus test diagnostics per country.

## P8. Stronger scorer (after P1–P3)
- Stage-2 LightGBM on OOF stage-1 probs, with context features.
- CatBoost as a second model, averaged by rank.
- A cross-encoder (mDeBERTa-v3-base, MIT) on the uncertain band, only if the GPU quota arrives.

---

## Execution plan (parallel tracks, one heavy job per box)

| Track | Owner / box | Work | Gate |
|---|---|---|---|
| A: Blocking v1 | Claude Code #1, amlc-box (already running Phase 0/1) | P1 retrievers 1–6 + candidate cache | recall ≥ 0.98, zero-cand < 0.3% |
| B: Features | Claude Code #2 (teammate box, or amlc-box when A is idle) | P3 + P4 + P5 features on the keys_v0 cache, then re-run on the v1 cache | paired-bootstrap gain, LOCO OK |
| C: Decision | Claude Code #2 after B | P2 cardinality + singleton head + expected-F; competition-aware val | gain confirmed on fold0 minus mini |
| D: France + norm | teammate | P6 normalization/token map, P7 French rules | LOCO + test diagnostics |

**Quick win today:** apply the measured rules (A + dropA + B, +0.0044) to the baseline's saved `test_pred.parquet` and submit as sub #2. Run the validation-auditor first.

**Submission ladder** (only when confirmed on fold0 minus mini):
1. baseline
2. baseline + rules
3. blocking v1
4. v1 + features
5. v1 + features + decision layer

Then France/ensemble polish on Day 3. Record the val↔LB gap every time.

## Sources
- Sparkly, TF/IDF top-k blocker (VLDB 2023): https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf
- One-to-one matching algorithms for ER (VLDB J. 2023): https://dl.acm.org/doi/10.1007/s00778-023-00791-3
- Blocking & filtering survey: https://arxiv.org/pdf/1905.06167
- Transliteration by orthography vs phonology (Hindi/Marathi→English): https://www.researchgate.net/publication/261725308
- Foursquare Location Matching winning pipelines (multi-retriever + 2-stage GBDT): https://github.com/TheoViel/kaggle_foursquare
