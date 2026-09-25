---
name: er-playbook
description: Grandmaster playbook for this Business Entity Resolution challenge — blocking recipes, pair features that beat adversarial hard negatives, model recipes (LightGBM, bi-/cross-encoders, license-safe), decision rules optimal for macro F0.5, the France/unseen-country strategy, and scaling to 10M records on a laptop. Load before designing or changing any blocking, normalization, feature, model, or decision logic.
---

# ER playbook (read this first, then the relevant reference file)

## The 8 truths of this dataset
1. **Record → at most one S1.** Always resolve competition: assign each S2/S3 record to its best S1, and use "margin to second-best S1" as a feature and a decision signal.
2. **Distractors are adversarial near-copies** (extra business word; house number ±k; same address with a different business). Similarity features alone plateau. **Difference features** (what differs, and how much it matters) win the precision battle.
3. **Positives are noisy in the same dimensions** (number typos 407→07, added words, missing address, native script). The model must learn *which* differences are noise. So give it the typed relation, not just a similarity: e.g. `house_rel ∈ {equal, suffix, prefix, off_by_small, different, missing}`.
4. **Macro F0.5 per S1**: singletons are 5.6% of train and score 1 only if empty. An empty prediction on a non-singleton scores 0. Predicting one confident match is worth ~0.6 even when the entity has 4 matches.
5. **Blocking recall is the ceiling.** Native-script names, domains/handles and empty addresses are the classic blocking misses. Measure them by bucket (`/blocking-audit`).
6. **France has no labels.** Everything must transfer: similarity/difference/rank features, generic normalization, and no `country` feature. Validate with leave-one-country-out.
7. **Scale**: 1.7M test queries against about 10M records. Anything O(n·k) with vectorized C loops is fine. Python loops over pairs are not (use rapidfuzz `cpdist`, numpy, polars).
8. **15 submissions**: the val protocol is the real leaderboard. Keep it honest (see `docs/strategy.md`).

## Recommended architecture (build it in this order)
1. Normalized cache → multi-retriever blocking (keys + TF-IDF char-3gram + compact/alias keys → later dense + reverse). Target ≥0.99 recall at ≤40 cands.
2. LightGBM stage-1 on about 60 pair features. Negatives = ALL blocking candidates of sampled train S1s (hard by construction).
3. Stage-2 LightGBM with context features computed from stage-1 probs: rank within S1, rank within record, margin, n_candidates, S2↔S3 agreement, the S1's max prob.
4. Decision: assign_best_s1 → global threshold (tuned) vs expected-F0.5 top-k (after isotonic calibration). Keep whichever wins on `mini` and confirms on fold0\mini.
5. Upgrades by expected ROI:
   1. learned token map (native script)
   2. cross-encoder on the uncertain band
   3. bi-encoder retrieval + cosine feature
   4. France normalization + synthetic French pairs

## Reference files (load as needed)
- `reference/blocking.md`: retrievers with code (keys, TF-IDF top-k, dense+FAISS, reverse), caps, recall accounting
- `reference/features.md`: the full feature catalogue, including difference, number-relation and competition features
- `reference/models.md`: LightGBM recipe, sampling, calibration, bi-/cross-encoder recipes, license-safe checkpoints
- `reference/decision.md`: metric math, thresholding vs expected-F0.5, singleton handling, France hedge
- `reference/france.md`: unseen-country strategy, LOCO validation, synthetic pairs, French normalization list
- `reference/scaling.md`: memory/time recipes for 10M records on 16 GB RAM / 4 GB VRAM, Windows gotchas
