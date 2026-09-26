Updated: 27 Sep 00:25 IST (amlc-box2, Lanes C + G; branch `laneC`)

## Lane G (GPU): per-account status (live: `logs/lane_g_status.log`; tmux `gpoll` = kaggle/poll_lane_g2.sh, every 3 min:
## training COMPLETE -> pull ckpts + sha256 -> push the G3 scoring kernel once its amlc-g3 dataset is ready -> pull logits)
| account | G2 training | state | G3 scoring (amlc-g3-*) |
|---|---|---|---|
| acc2 ashu273k | XLM-R half 0 (amlc-g2-xlmr-h0, v2 by lead) | RUNNING since ~23:30 IST (420 min budget) → ETA ~06:30 IST | auto-push after training (L4, T4x2 fallback) |
| acc3 darshanbagade | XLM-R half 1 (amlc-g2-xlmr-h1) | RUNNING (L4) → ETA ~06:30 IST | auto-push after training |
| acc5 darshanbagadeycce | MiniLM both halves (amlc-g2-minilm) | **COMPLETE** 00:01 IST: full epoch in 128 min on 2×T4, 492 pairs/s/GPU; best val logloss **h0 0.00629, h1 0.00614** (still falling at every 30-min ckpt); ckpts pulled to artifacts/kaggle/amlc-g2-minilm | auto-push as soon as acc5's amlc-g3 upload is ready (~00:40 IST); est. 10.8M pairs × 2 models |
- **G3 inputs** (`pipelines/xenc_g3_prep.py`, data/kaggle/g3_up): box1's 10.76M-pair set (test 8.06M, fold0x 1.52M, mini 0.38M, train 0.79M) with
  - **halves recomputed as md5(s1_id) % 2**: the G2 models trained on these; box1's `cf` is hash(42), the OLD ckpts' halves. Checked 0 mismatches vs the G1 export over 146k S1.
  - text rebuilt on norm_v2 in the training serialization.
- **Kernel** `kaggle/xenc_g3.py` (variants via `kaggle/gen_g3_variants.py`, compile + line-count checked). A half-k model scores the half-(1−k) train pairs + all other pairs, 1M-pair parts. CPU smoke-tested (strict ckpt load, own-half exclusion, finite logits).
- **Outputs for box1:** `logits_h{k}_{tag}.parquet` (pair_id, xenc_h{k}). Map pair_id → (tag, k1, k2) via data/kaggle/g3_up_map.parquet, or box1's own pairs.parquet (same pair_id).
- **Stacking rule:** train uses the other half's model; mini/fold0x/test use the mean of both.
- **Incident:** an early G3 minilm push had no dataset attached (Kaggle only warns: "not valid dataset sources"). It was deleted before running; `kaggle_gpu.py run` now fails on that warning.

## Lane C results
- **Transfer-safe feature list (C1, gate PASS):** drop {ex_ldf_min, ex_ldf_max, mi_ldf_min, mi_ldf_max}. LOCO-avg +0.0022 (3 seeds), clean mini −0.0002, fold0x −0.00007 (n.s.). Rescorable from test_feats. Measured on the 0710 base → Lane A to re-check on psemb.
- **norm v2** ACCEPTED (lead 16:45). **French BIZ words KILLED** (98% of 1,384 FR up-flips are near-copies).
- France density report: no FR over-rejection; FR loss is band ambiguity (band records/S1 0.358 vs US 0.234 / IN 0.181), norm v2 removes ~17% of it.

NEEDS-LEAD: none blocking. (Transfer-safe list gain is US→IN-only; accept or not for the final stack.)
