# Ideas backlog (ordered by expected gain / cost). Update the status + run_id when done.

Status: `todo` · `doing(<who>)` · `done(<run_id>, Δ)` · `killed(<reason>)`

## P0: Day 1 (get on the board)
- [ ] **SETUP**: `python scripts/prepare_data.py` → `python scripts/build_norm_cache.py` (all). Check `data/samples/`.
- [ ] **EXP-001 Baseline end-to-end**: IMPLEMENTED in `pipelines/baseline_v0.py` and tested end-to-end on synthetic data. Next: run `--subset micro` locally, then `mini` + `--test` on SageMaker/Kaggle, then `/submit`.
  - Blocking: key-based within country:
    - k1 = (a_house, first 4 chars of the first a_street token)
    - k2 = (first core token ≥3 chars, first 3 chars of the second core token)
    - k3 = n_compact[:8]

    Drop key blocks with >300 records per side. Report blocking recall on `mini`.
  - Features: rapidfuzz `cpdist` on n_core/n_full/n_compact/a_full/a_street (ratio, token_set_ratio, token_sort_ratio, partial_ratio, JaroWinkler), house equal/diff, legal equal, numbers Jaccard, state equal, empty flags, source, n_kind, n_script≠latin, token-count diffs, extra/missing core token counts.
  - Model: LightGBM on pairs from ~200k S1 of folds 1–4. Predict on `mini` → `assign_best_s1` → `tune_threshold` → report.
  - Then run on the full test, `/submit` it as the first LB anchor.
- [ ] **EXP-002 TF-IDF char-3gram name blocking** (per country, chunked sparse top-k, k≈20–50) unioned with the keys. Goal: recall ≥ 0.99.
- [ ] **EXP-003 Reverse retrieval** (record → top-3 S1) and union. Also yields competition features.

## P1: Day 2 (big gains)
- [ ] EXP-010 Difference + competition features (see strategy.md). Expect the biggest precision gain against hard negatives.
- [ ] EXP-011 Learned token map from train pairs: align tokens of native-script/transliterated names with Latin names ('praivet'→'private', state names in 7 scripts, 'kampani'→'company'). Apply in normalization (bump NORM_VERSION).
- [ ] EXP-012 Bi-encoder: fine-tune `intfloat/multilingual-e5-small` (MIT, 118M) with in-batch + hard negatives on "name | address" strings from folds 1–4. Use it for FAISS retrieval + a cosine feature.
- [ ] EXP-013 Decision rule: expected-F0.5 top-k vs global threshold vs per-source threshold. Calibrate first (isotonic on OOF).
- [ ] EXP-014 LOCO evaluation for every feature group. Drop features that don't transfer.
- [ ] EXP-015 Cross-encoder (`microsoft/mdeberta-v3-base` MIT or `xlm-roberta-base` MIT) trained on blocking hard negatives. Run on the uncertain band only (0.05<p<0.95).

## P2: Day 2–3 (France + polish)
- [ ] EXP-020 French normalization dictionary: rue/r/r., bd/boulevard, av, all/allée, imp, ch, rte, pl, fg, n°, bis/ter, "eme etage"; legal forms; strip "(france)"; region↔department map for the cities seen (general knowledge, hand-written).
- [ ] EXP-021 Synthetic French pairs: apply noise operators learned from train (case, legal swap, token drop, typo, reorder, accent injection, abbreviation) to test-France S1 records → extra positives. Hard negatives: +business word / house ±k. Train or fine-tune with them; check LOCO.
- [ ] EXP-022 Conservative France: a higher threshold for France is a free precision hedge. Decide via val of the LOCO proxy.
- [ ] EXP-023 Stage-2 stacking with graph/context features on stage-1 probs.
- [ ] EXP-024 Ensembling (LGBM seeds/folds + cross-encoder), rank-average then re-calibrate.

## Parking lot
- LLM judge (Qwen2.5-7B-Instruct, Apache) on the few hardest France pairs. Probably too slow and costly; only if time allows.
