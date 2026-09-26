Updated: 26 Sep 21:36 IST (amlc-box2, Lanes C + G; branch `laneC`)

## Lane G (GPU): per-account status (live: `logs/lane_g_status.log`, tmux `gpoll` polls every 2 min and pulls outputs on completion)
| account | job (slug) | accelerator | state | ETA |
|---|---|---|---|---|
| acc2 ashu273k | XLM-R half 0 (amlc-g2-xlmr-h0) | L4X1 | pushed 21:33, QUEUED | 7 h train after start (budget 420 min) |
| acc3 darshanbagade | XLM-R half 1 (amlc-g2-xlmr-h1) | L4X1 | pushed 21:33, QUEUED | 7 h after start |
| acc5 darshanbagadeycce | MiniLM 5× data, both halves (amlc-g2-minilm) | T4x2 | pushed 21:33, QUEUED | 2.5 h after start |
- Rule: if an L4 job is still QUEUED at 21:53 IST, re-push it on T4x2 (queue = not a failure).
- Fixed on the way: generated job scripts were truncated (regex bug) in the first push; the "Maximum batch GPU session count of 2" push error was reported as success by `kaggle_gpu.py` (now exits non-zero). Stale queued `amlc-smoke` kernels on acc2/acc3 were deleted to free the 2 GPU slots.
- G3 (scoring kernels) waits for box1's pair-text export (psemb p ≥ 0.005) + the G2 checkpoints.

## Lane C results
- **Transfer-safe feature list (C1, gate PASS):** drop {ex_ldf_min, ex_ldf_max, mi_ldf_min, mi_ldf_max}. LOCO-avg +0.0022 (3 seeds), clean mini −0.0002, fold0x −0.00007 (n.s.). Rescorable from test_feats. Measured on the 0710 base → Lane A to re-check on psemb.
- **norm v2** ACCEPTED (lead 16:45). **French BIZ words KILLED** (98% of 1,384 FR up-flips are near-copies).
- France density report: no FR over-rejection; FR loss is band ambiguity (band records/S1 0.358 vs US 0.234 / IN 0.181), norm v2 removes ~17% of it.

NEEDS-LEAD: none blocking. (Transfer-safe list gain is US→IN-only; accept or not for the final stack.)
