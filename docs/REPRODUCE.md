# REPRODUCE (draft, 27 Sep 2026): regenerate `matching_results.tsv` + `candidate_pairs.tsv` from the raw data

Status: DRAFT written by the Lane A session. Commands are the ones actually used for the current submission candidates
(XB = run `20260927-0257_…-ctx3-b`, and its seed bag if kept). The TODO items at the end must be closed before the final zip.

## 0. Environment (CPU box)
- Python 3.13.15 (uv venv `.venv`). Pinned versions actually used:
  - polars 1.44.2, pandas 3.0.2, numpy 2.4.6, pyarrow 24.0.0, scipy 1.15.3, scikit-learn 1.7.0
  - lightgbm 4.6.0, rapidfuzz 3.14.6, anyascii 0.3.3 (ISC), sparse-dot-topn 1.2.0 (Apache-2.0)
  - torch 2.6.0 + transformers 5.17.0 (only for Kaggle kernels / CPU smoke tests)
  - `requirements.txt` must be regenerated from `.venv` (TODO 1).
- Hardware used: AWS r7i 8 vCPU / 61 GB. Peak RSS ≈ 31 GB (full test blocking).
- Seeds: every pipeline uses seed 42 (LightGBM `seed=42, deterministic=True, force_row_wise=True`). Seed bags use seeds 42–46; LightGBM derives `bagging_seed` / `feature_fraction_seed` from `seed`. Kaggle kernels use seed 42.
- Raw data is read only through `ber.io` from `student_resource/` (never modified).

## 1. CPU pipeline (order matters)
```bash
python scripts/check_env.py
python scripts/prepare_data.py                    # parquet copies of the raw TSVs + folds (md5 over train S1)
python scripts/build_token_map.py                 # learned native-script token map (folds 1-4 GT pairs only)
# normalization caches v0, v1, v2 (NORM_VERSION in scripts/build_norm_cache.py; TODO 2: make it a CLI flag)
python scripts/build_norm_cache.py --split all    # run once per NORM_VERSION in {0, 1, 2}
# blocking v1 on norm_v1 -> candidate cache v1_n1 (train tag = 180k fold 2-4 S1, mini, fold0x, test)
python pipelines/build_cache.py ...               # TODO 3: exact flags per tag (see docs/decisions.md 25 Sep entries)
python pipelines/augment_nkey_num.py ...          # + nkey_num retriever -> cache v1_n2 (the scored candidate set)
# context features ctx v3 (G1-G5) for train/mini/fold0x
python pipelines/features_v1.py --cache v1_n2 --ctx-ver 3 --featurize-only
```

## 2. GPU steps (Kaggle, private datasets/kernels, driven by `scripts/kaggle_gpu.py`)
All models are MIT-licensed and ≤ 8B (checks logged in docs/decisions.md). They are frozen or fine-tuned only on train folds 1–4.

| step | CPU export | Kaggle kernel | output → join |
|---|---|---|---|
| bge-m3 band cosines (frozen, zero-shot) | `pipelines/emb_export.py` | `kaggle/emb_bge_m3.py` (slug amlc-emb-bgem3) | `pipelines/emb_split.py` → `data/cands/v1_n2/emb_<tag>.parquet` |
| MiniLM cross-encoder, trained 25 Sep (cross-fit halves `s1_id.hash(42) % 2`) | `pipelines/xenc_export.py` | `kaggle/xenc_train_score.py` (slug amlc-xenc) | ckpt_m0/m1 |
| score the pair set p ≥ 0.01 with those ckpts | `pipelines/xenc2_export.py` | `kaggle/xenc_train_score.py`, score-only, ckpts in the dataset (slug amlc-xenc2-score) | `pipelines/xenc_join.py minilm=artifacts/kaggle/amlc-xenc2-score` |
| Lane G MiniLM-5x / XLM-R (G1 export → G2 train → G3 score; md5(s1_id) % 2 halves) | `pipelines/xenc_g1_export.py`, G3 prep (branch laneC) | `kaggle/xenc_g2*.py`, `kaggle/xenc_g3*.py` (branch laneC; `kaggle/launch_lane_g.sh`) | `pipelines/xenc_verify_halves.py`, `pipelines/xenc_join.py mlm5x=…` |

Checkpoint sha256 (MiniLM ckpts used by XB):
- ckpt_m0.pt `cbf19475a699976ac55825fcbd236dd72c58da18e5607804fad0d895a0f691d6`
- ckpt_m1.pt `54a248eaf2a50cf62dd93ef9ee9eff7b7172a17d6cc69bdec2ab4f1552fe96ac`
- Lane G ckpts: TODO 4 (sha256 on box2).

Leakage controls:
- Every xenc train-row logit is out-of-fold (cross-fit halves; `xenc_verify_halves.py` asserts 0 in-sample).
- es / val / test rows get the mean of both halves.
- Pairs never scored (stage-1 p < 0.01) have NaN in both train and test.

## 3. Final model + decision
```bash
# XB: psemb (ps + bge-m3) + xenc_minilm - 4 ldf features, LightGBM, trained on the v1_n2 train tag
python pipelines/features_v1.py --cache v1_n2 --ctx-ver 3 --groups G1,G2,G3,G4,G5 --threads 7 --ps --emb --xenc minilm \
    --drop-feats ex_ldf_min,ex_ldf_max,mi_ldf_min,mi_ldf_max --tag b
# (seed bag: add --seeds 43,44,45,46 --bag-parent <XB run>)
# threshold: density-matched validation (one global rho from label-free test/val band mass), t re-tuned on mini
python pipelines/dm_val.py --run <run> --tag mini --freeze-w 20260926-1413_aryan-bhendarkar_dm-val-mini
python pipelines/dm_val.py --run <run> --tag fold0x --freeze <dm-val mini run>          # confirmation
```
The global threshold is t = 0.825 (DM-selected). Decision rule: each S2/S3 record goes to its best S1 (`assign_best_s1`) and is kept if p ≥ t.

## 4. Test inference (C1 path: France rows on norm v2)
```bash
# one-time full test featurization (blocking + base + ctx) -> data/cands/v1_n2/test.parquet + test_feats_g15_ctx3.parquet
python pipelines/predict_test_v1.py --run <run> --cache v1_n2 --threads 7
# afterwards any model is rescored from the saved matrix (+ emb/xenc joins, ps derived features)
python pipelines/predict_test_v1.py --run <run> --cache v1_n2 --from-feats --threads 7 --threshold 0.825
# France: norm v2 re-featurization of France pairs only (US/IN byte-identical), then splice
python pipelines/refeat_norm.py --cache v1_n2 --tag test
python pipelines/splice_fr.py filter
python pipelines/predict_test_v1.py --run <run> --cache v1_n2 --src-tag test_n2fr --norm 2 --out-suffix _n2fr --threads 7 --threshold 0.825 --no-save-feats
python pipelines/splice_fr.py splice --run <run> --t 0.825
python scripts/make_submission.py --run <run> --matches artifacts/<run>/test_matches_c1.parquet --candidates data/cands/v1_n2/test.parquet
```
`make_submission` writes both TSVs through `ber.submission.write_id_lists` (LF, tab, empty rows kept) and runs the official validator with `--check-ids`. `candidate_pairs.tsv` is the exact scored set (v1_n2 test cache), and matches ⊆ candidates is asserted. Wrapper used: `logs/buildC1x.sh <run> <t> <val> "<note>"`.

## TODO before the final zip
1. `requirements.txt` from `.venv` (`uv pip freeze`) plus a Kaggle kernel environment note (Kaggle image torch/transformers versions from the kernel logs).
2. `build_norm_cache.py --norm-version` flag (it is currently a module constant).
3. Exact `build_cache.py` / `augment_nkey_num.py` / `blocking_v1.py` commands per tag (from the run records).
4. Lane G ckpt sha256 + kernel versions (box2 / branch laneC merge).
5. Runtime table: full test blocking ≈ 2.5 h; featurize ≈ 3.5 h; rescore 25 min; Kaggle scoring ≈ 1 h per model on 1× L4.
6. Merge branch laneC (Lane G kernels, `kaggle/README_lane_g.md`) into main.
