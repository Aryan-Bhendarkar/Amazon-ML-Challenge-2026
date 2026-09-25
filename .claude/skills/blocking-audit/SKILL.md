---
name: blocking-audit
description: Audit candidate generation (blocking) — pair recall, full-coverage rate, candidate-count distribution, reduction ratio, per-country recall, and a categorized sample of MISSED true matches with a recommendation for which retriever would recover them. Use after any blocking change or when recall is below 0.99.
argument-hint: "<run_id or candidates parquet> [subset=mini]"
---

# Blocking audit: $ARGUMENTS

1. Load the candidates (s1_id, cand_id) for the eval subset. Use `harness.EvalContext.load(subset)` + `blocking_eval.blocking_report`. Report:
   - pair_recall (overall + by country)
   - entity_full_coverage
   - cands mean/p50/p95/max
   - empty_cand_rate
   - reduction_ratio
2. Find the missed true pairs: `pairs[~label_candidates(...)]` restricted to eval S1s. Sample 60, stratified by country and source (S2/S3).
3. For each sampled miss, show the S1 and the missed record, raw AND normalized (`data/features/norm_v*`). Put each miss in ONE bucket:
   - native script / transliteration
   - domain / handle / compact name
   - alias (fka/dba)
   - empty or partial address
   - heavy name typo
   - name word order / extra words
   - house-number mismatch
   - city/state variant
   - key block too big (capped)
   - other
4. Output a table: bucket → count in sample → estimated share of all misses → the retriever/normalization fix that would catch it (TF-IDF char n-grams, compact key, learned token map, dense bi-encoder, reverse retrieval, raising k).
5. Check the cost side: pairs per S1 and the biggest blocks (France cities!). Recall gains that multiply pairs by more than 2× need justification.
6. Write the findings into the run's `notes.md` and add concrete items to `docs/ideas_backlog.md`.
