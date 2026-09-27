# REPRODUCE: raw `student_resource/` data → `matching_results.tsv` + `candidate_pairs.tsv` (FINAL pipeline, 27 Sep 2026)

Final model **D** = run `20260927-0508_…-ctx3-d` (LightGBM on base + ctx v3 + per-source + bge-m3 band cosines + MiniLM-5x
cross-encoder logit, minus 4 ldf features), threshold t = 0.825. Submitted as the data-driven **hedge**: countries seen in
the training labels use D; unseen countries (France) use the same pipeline without cross-encoder features (psemb run
`20260926-1737_…-ctx3-psemb`). LB: D-HEDGE 0.979503 (D alone 0.978025).
If F-HEDGE (all three cross-encoders) wins on the LB, replace step 6 with the F line and keep everything else.

## 0. Environment
- CPU: AWS r7i, 8 vCPU / 61 GB, Ubuntu; Python 3.13.15 (uv venv). `pip install -r requirements.txt` (pinned, imported packages only).
  Peak RSS ≈ 31 GB (full test blocking). All CPU steps use ≤ 8 threads.
- GPU: Kaggle notebooks (2× Tesla T4 or 1× L4, Python 3.12, torch 2.10.0+cu128; see `requirements-kaggle.txt`), private
  datasets/kernels driven from the box by `scripts/kaggle_gpu.py` (`data` / `run` / `wait` / `pull`).
- Seeds: 42 everywhere (LightGBM `seed=42, deterministic=True, force_row_wise=True`; kernels `torch.manual_seed(42+k)`;
  sampling in exports seed 42; cross-fit halves = md5(s1_id) % 2 for Lane G, polars `hash(42) % 2` for the 25 Sep MiniLM).
- Raw data is read only through `ber.io` from `student_resource/` (never modified). No external data.

## 1. Data, folds, normalization (≈ 20 min)
```bash
python scripts/check_env.py
python scripts/prepare_data.py            # parquet copies of the raw TSVs + md5 folds over train S1
python scripts/build_token_map.py         # native-script -> Latin token map, learned from folds 1-4 GT pairs only
# normalization caches: set NORM_VERSION in scripts/build_norm_cache.py to 0, 1, 2 in turn (≈ 11 min each, 2 jobs)
python scripts/build_norm_cache.py --split all
```
norm v1 = hand rules + token map; norm v2 = v1 + France-keyed rules (region/department → state, bis/ter, "(France)",
cie/compagnie, et, frs, st, legal `ei`); v2 is byte-identical to v1 for US/India (0 changed train rows).

## 2. Blocking → candidate caches v1_n2 (train/mini/fold0x ≈ 45 min; test ≈ 2.5 h)
```bash
python pipelines/build_cache.py --cand-ver v1 --norm-v 1 --subset mini       # train tag (150k + 30k es S1) + mini
python pipelines/build_cache.py --cand-ver v1 --norm-v 1 --subset fold0x --skip-train
python pipelines/v1_test.py --model-run <gate run> --save-feats              # test blocking + base features -> v1_n1/test.parquet
python pipelines/augment_nkey_num.py --src v1_n1 --dst v1_n2 --norm-v 1 --files train mini fold0x test   # + nkey_num retriever
```
Retrievers: TF-IDF name char-3grams (max_df .01, w .35) + address words (max_df .02, w .65), top-30/S1; exact normalized
keys; nkey_num (name key + shared number, cap 50). Cap 100/S1 (nkey_num exempt). Test: 164,745,367 pairs, all 1,732,544
S1 covered. Mini pair recall 0.9871. **`data/cands/v1_n2/test.parquet` is the exact scored set → `candidate_pairs.tsv`.**

## 3. Context + per-source features (≈ 30 min train/val; test inside step 7)
```bash
python pipelines/features_v1.py --cache v1_n2 --ctx-ver 3 --featurize-only           # ctx3_{train,mini,fold0x}.parquet
```

## 4. GPU step A: frozen bge-m3 band cosines (Kaggle)
```bash
python pipelines/emb_export.py                                    # band pairs + record texts -> data/kaggle/emb/
python scripts/kaggle_gpu.py data data/kaggle/emb --slug amlc-emb
python scripts/kaggle_gpu.py run kaggle/emb_bge_m3.py --slug amlc-emb-bgem3 --data amlc-emb --gpu t4x2
python scripts/kaggle_gpu.py wait --slug amlc-emb-bgem3 ; python scripts/kaggle_gpu.py pull --slug amlc-emb-bgem3 --out artifacts/kaggle/amlc-emb-bgem3
python pipelines/emb_split.py artifacts/kaggle/amlc-emb-bgem3/pair_cos.parquet v1_n2   # -> data/cands/v1_n2/emb_<tag>.parquet
```

## 5. GPU step B: Lane G MiniLM-5x cross-encoder (Kaggle; G1 → G2 → G3)
```bash
python pipelines/xenc_g1_export.py                               # 8.01M train pairs (all positives + hard negatives), cf = md5(s1_id)%2
python kaggle/gen_g2_variants.py                                 # per-job training kernels (CONFIG line; compile-checked)
bash kaggle/launch_lane_g.sh 5 kaggle/xenc_g2_minilm.py amlc-g2-minilm t4x2   # G2: one model per half, 2×T4, 1 epoch
python pipelines/xenc2_export.py                                 # scoring pair set: stage-1 p >= 0.01 (train OOF / val / test)
python pipelines/xenc_g3_prep.py                                 # G3 inputs: md5 halves (checked == G1), norm_v2 texts
python kaggle/gen_g3_variants.py
# upload data/kaggle/g3_up as dataset amlc-g3, then score with the G2 kernel output mounted as a kernel source:
python scripts/kaggle_gpu.py run kaggle/xenc_g3_minilm.py --slug amlc-g3-minilm --data amlc-g3 --kernels <user>/amlc-g2-minilm --gpu t4x2
python scripts/kaggle_gpu.py pull --slug amlc-g3-minilm --out artifacts/kaggle/amlc-g3-minilm_latest
python pipelines/xenc_verify_halves.py artifacts/kaggle/amlc-g3-minilm_latest        # every train pair OOF (0 in-sample)
python pipelines/xenc_join.py mlm5x=artifacts/kaggle/amlc-g3-minilm_latest --map data/kaggle/xenc2_map
#  (F variant also: minilm=artifacts/kaggle/amlc-xenc2-score  xlmr=artifacts/kaggle/amlc-g3-xlmr-h0+artifacts/kaggle/amlc-g3-xlmr-h1)
```
Training recipe (`kaggle/xenc_g2.py`): BCE, fp16 AMP, max_len 96, AdamW lr 5e-5 (MiniLM) / 2e-5 (XLM-R), warmup 5%,
linear decay, 1 epoch, time-boxed after a 200-step throughput probe, best checkpoint by fold-1 val logloss, positive-pair
augmentation p = 0.15 (token drop except the first token, adjacent swap, legal-suffix drop; no digit edits).
Input serialization: `[COL] name [VAL] <name> [COL] address [VAL] <address> [NUM] <house>`.

| checkpoint | sha256 | val logloss | runtime |
|---|---|---|---|
| MiniLM-5x h0 (used by D) | `182e5c94e29243b7f4464291c25b01d372f21aef71b01efd2f4230d3b47209b9` | 0.00629 | 128 min (2×T4, both halves in parallel, 492 pairs/s/GPU) |
| MiniLM-5x h1 (used by D) | `f63ed5ad964809a78cac96db14c0fb02871d4062a6bd8c4d887f16fd305aa1fb` | 0.00614 | (same run) |
| XLM-R h0 (F only) | `0a06715eac8c30109a3562203b49c287fc8c74350d5b3a95cbe212888910e004` | 0.00642 | 307 min (1×L4) + scoring 189 min |
| XLM-R h1 (F only) | `ac94b96c833a58e7fddf71006de3125dfd74470665519f06695abe45ff9d1c92` | 0.00628 | 304 min (1×L4) + scoring 178 min |
| MiniLM 25 Sep m0 (F only) | `cbf19475a699976ac55825fcbd236dd72c58da18e5607804fad0d895a0f691d6` | – | – |
| MiniLM 25 Sep m1 (F only) | `54a248eaf2a50cf62dd93ef9ee9eff7b7172a17d6cc69bdec2ab4f1552fe96ac` | – | – |

Scoring throughput: MiniLM ≈ 2.8k pairs/s per T4, XLM-R ≈ 1.0k pairs/s per L4 (10.4M pairs per model).

## 6. Final LightGBM + threshold (≈ 25 min train, ≈ 10 min DM-val)
```bash
# D
python pipelines/features_v1.py --cache v1_n2 --ctx-ver 3 --groups G1,G2,G3,G4,G5 --threads 7 --ps --emb --xenc mlm5x \
    --drop-feats ex_ldf_min,ex_ldf_max,mi_ldf_min,mi_ldf_max --tag d
# base model for unseen countries (psemb: no cross-encoder features)
python pipelines/features_v1.py --cache v1_n2 --ctx-ver 3 --groups G1,G2,G3,G4,G5 --threads 7 --ps --emb --tag psemb
# (F: --xenc minilm,mlm5x,xlmr)
# density-matched threshold (one global rho from label-free test/val band mass), frozen on mini, confirmed on fold0x
python pipelines/dm_val.py --run <D run> --tag mini --freeze-w 20260926-1413_aryan-bhendarkar_dm-val-mini
python pipelines/dm_val.py --run <D run> --tag fold0x --freeze <dm-val mini run>
```
D: mini clean 0.9858; fold0x clean 0.98548, DM 0.98497 at t = 0.825 (the fold0x DM optimum; robust for rho 1.39–2.6).
Decision rule: each S2/S3 record → its best S1 (`ber.decision.assign_best_s1`), kept if p ≥ t.

## 7. Test inference (full featurize ≈ 3.5 h once; later models rescore in ≈ 25 min)
```bash
python pipelines/predict_test_v1.py --run <D run> --cache v1_n2 --threads 7 --threshold 0.825          # saves test_feats_g15_ctx3
python pipelines/predict_test_v1.py --run <psemb run> --cache v1_n2 --from-feats --threads 7 --threshold 0.825
# France (and any country whose normalization changed) on norm v2: re-featurize those pairs, score, splice (per run)
python pipelines/refeat_norm.py --cache v1_n2 --tag test          # candidate set unchanged; US/IN rows copied
python pipelines/splice_fr.py filter
for R in <D run> <psemb run>; do
  python pipelines/predict_test_v1.py --run $R --cache v1_n2 --src-tag test_n2fr --norm 2 --out-suffix _n2fr --threads 7 --threshold 0.825 --no-save-feats
  python pipelines/splice_fr.py splice --run $R --t 0.825        # -> artifacts/<run>/test_matches_c1.parquet
done
```

## 8. Hedge + output files (minutes)
```bash
python pipelines/hedge_unseen.py --run <D run> --base-run <psemb run>      # seen countries from train data; -> test_matches_hedge.parquet
python scripts/make_submission.py --run <D run> --matches artifacts/<D run>/test_matches_hedge.parquet \
    --candidates data/cands/v1_n2/test.parquet --note "final"
```
`hedge_unseen.py` reproduces the D-HEDGE (LB 0.979503) matches exactly (5,736,610 pairs). `make_submission.py` writes both
TSVs through `ber.submission.write_id_lists` (UTF-8, LF, tab, empty rows kept) and **fails** (non-zero exit, nothing
uploaded) on:
- a record matched to more than one S1
- a matched S1 not in test
- non-S2/S3 ids
- missing or duplicate S1 rows (read-back check)
- matched pairs not in the candidates

It then runs the official validator with `--check-ids`.

## Runtime summary (8 vCPU box + Kaggle)
| step | time |
|---|---|
| prepare + 3 normalization caches | ≈ 40 min |
| blocking train/mini/fold0x + test | ≈ 45 min + 2.5 h |
| ctx features train/val | ≈ 30 min |
| Kaggle bge-m3 | ≈ 1 h |
| Kaggle MiniLM-5x train + score | ≈ 2.2 h + ≈ 1.5 h |
| LightGBM D + psemb + DM-val | ≈ 1 h |
| test featurize (once) | ≈ 3.5 h |
| France refeat + splice | ≈ 40 min |
| hedge + submission | ≈ 10 min |
