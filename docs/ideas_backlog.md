# Ideas backlog (ordered by expected gain / cost). Update the status + run_id when done.

Status: `todo` · `doing(<who>)` · `done(<run_id>, Δ)` · `killed(<reason>)`

## P0: Day 1 (get on the board)
- [ ] **SETUP**: `python scripts/prepare_data.py` → `python scripts/build_norm_cache.py` (all). Check `data/samples/`.
- [ ] **EXP-001 Baseline end-to-end**: IMPLEMENTED in `pipelines/baseline_v0.py` and tested end-to-end on synthetic data. Next: run `--subset micro` locally, then `mini` + `--test` on the EC2 box (or Kaggle), then `/submit`.
  - Blocking: key-based within country:
    - k1 = (a_house, first 4 chars of the first a_street token)
    - k2 = (first core token ≥3 chars, first 3 chars of the second core token)
    - k3 = n_compact[:8]

    Drop key blocks with >300 records per side. Report blocking recall on `mini`.
  - Features: rapidfuzz `cpdist` on n_core/n_full/n_compact/a_full/a_street (ratio, token_set_ratio, token_sort_ratio, partial_ratio, JaroWinkler), house equal/diff, legal equal, numbers Jaccard, state equal, empty flags, source, n_kind, n_script≠latin, token-count diffs, extra/missing core token counts.
  - Model: LightGBM on pairs from ~200k S1 of folds 1–4. Predict on `mini` → `assign_best_s1` → `tune_threshold` → report.
  - Then run on the full test, `/submit` it as the first LB anchor.
- [x] done(blocking-v1-mini-n1, recall 0.861→0.9815): name char3 + address-word TF-IDF, keys, skeleton, empty-addr retrievers. **EXP-002 TF-IDF char-3gram name blocking** (per country, chunked sparse top-k, k≈20–50) unioned with the keys. Goal: recall ≥ 0.99.
- [ ] (deferred: gate met without it; cost ≈ a full forward pass) **EXP-003 Reverse retrieval** (record → top-3 S1) and union. Also yields competition features.

## P1: Day 2 (big gains)
- [ ] EXP-010 Difference + competition features (see strategy.md). Expect the biggest precision gain against hard negatives.
- [ ] EXP-011 Learned token map from train pairs: align tokens of native-script/transliterated names with Latin names ('praivet'→'private', state names in 7 scripts, 'kampani'→'company'). Apply in normalization (bump NORM_VERSION).
- [ ] EXP-012 Bi-encoder: fine-tune `intfloat/multilingual-e5-small` (MIT, 118M) with in-batch + hard negatives on "name | address" strings from folds 1–4. Use it for FAISS retrieval + a cosine feature.
- [ ] doing(amlc-07, pipelines/decision_v1.py) EXP-013 Decision rule: expected-F0.5 top-k vs global threshold vs per-source threshold. Calibrate first (isotonic on OOF).
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

## From error analysis of 20260925-1236_aryan_baseline-v0-keys-lgbm (25 Sep, see its error_analysis.md)
- [x] done(20260925-2121_aryan-bhendarkar_feat-v1-v1-n1-g1-g2-g3-g4-g5, +0.0189 on v1_n1 together with EXP-018; LOCO +0.014/+0.017) **EXP-016 S1-context uniqueness features** (TOP): `s1_same_name` = #S1 (same split+country) with identical sorted n_core; `s1_same_addr` = same for sorted a_full; ==1 flags; `n_key_equal`. Targets name-only (empty cand address) records and co-location, which cost 0.0135 macro. **Measured as post-processing rules (no retraining): +0.0044 mini (CI [0.0041, 0.0048]), IN +0.0046, US +0.0043.** Expected +0.005–0.008 as LGBM features. Cheap (group_by counts). France: 52.5% of S1 share names (same as train), and 19% share addresses, which makes the flags more conservative there.
- [ ] **EXP-017 Competition-aware val**: 55% of val FPs are records owned by a non-mini S1 that never competes in val `assign_best_s1`. Add competitor S1 queries (all S1 sharing a key with a mini candidate) and record-side best-other-S1 sim/margin features. +0.002 FP-side at stake, plus a threshold that matches test conditions.
- [x] done(same run as EXP-016: ber.ctx_features G3/G4/G5) **EXP-018 Number + extra-token features**: house digit-edit distance / abs diff / other-number agreement; max S1-doc-freq of cand-only name tokens. Targets house noise (0.0058) and business-word edits (0.0044); expected +0.001–0.002.
- [ ] **NORM v1 (small)**: honorific stopwords (mr/dr/smt/m s/www/com) + fuzzy legal canonicalization (praibhet/piraivet/limirrd/PRlVATE/lnc). ≤ +0.0005. Add French honorifics/legal typos too.
- Killed on arrival: top-1 fallback for empty predictions (−0.0045 … −0.00002 at every t2); these are blocking misses or singletons.
