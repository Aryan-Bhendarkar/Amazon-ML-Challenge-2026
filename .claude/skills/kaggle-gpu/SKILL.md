---
name: kaggle-gpu
description: Run GPU jobs (cross-encoder / bi-encoder training and scoring) on Kaggle's free 2x T4 from this box via the Kaggle CLI — upload inputs as a private dataset, push a self-contained script, poll, pull outputs. Use for ANY GPU work (AWS GPU quota is 0) or questions about Kaggle GPU runs.
argument-hint: "<gpu job>"
---

# Kaggle GPU: $ARGUMENTS

AWS GPU quota is **0** (G/VT on-demand + spot, all regions, SageMaker too; verified 25 Sep). Kaggle is our GPU.
Model: **Kaggle = remote GPU worker, this box = control plane.** You never open a browser; `scripts/kaggle_gpu.py` does
upload → push → run in background → poll → download.

## Status check
!`bash -lc 'test -f ~/.kaggle/kaggle.json -o -f ~/.kaggle/access_token && echo "auth: OK" || echo "auth: MISSING - ask human for kaggle.json (see Setup)"; kaggle --version 2>/dev/null || echo "cli missing: uv tool install kaggle"'`

## Setup (one-time, human does it)
1. kaggle.com account **phone-verified** (Settings → Phone verification) — otherwise no GPU and no internet.
2. Settings → API → Generate API token → save to `~/.kaggle/access_token` (chmod 600). Done for aryanbhendarkar (smoke test passed 25 Sep: 2x T4, HF ok).
3. `python scripts/kaggle_gpu.py whoami` then the smoke test:
   `python scripts/kaggle_gpu.py run kaggle/gpu_smoke.py --slug amlc-smoke --wait` → expect 2× Tesla T4, `hf_ok: true`.

## Commands
```
python scripts/kaggle_gpu.py data  <dir> --slug amlc-xenc-data -m "v1_n1 hard pairs"   # create/version PRIVATE dataset
python scripts/kaggle_gpu.py run   kaggle/<job>.py --slug amlc-xenc --data amlc-xenc-data [--wait]
python scripts/kaggle_gpu.py status --slug amlc-xenc
python scripts/kaggle_gpu.py wait   --slug amlc-xenc      # polls every 60 s, then pulls to artifacts/kaggle/<slug>/
```
Long waits: run `wait` in tmux (`tmux new -d -s kg 'python scripts/kaggle_gpu.py wait --slug amlc-xenc > logs/kaggle_xenc.log 2>&1'`)
and keep working on CPU tracks meanwhile. The Kaggle run continues even if this box stops.

## Limits (plan around them)
- GPU: 2× T4 (16 GB each, fp16 ~65 TFLOPS peak, **no bf16**), 4 vCPU, ~29 GB RAM. ~30 GPU-h/week per account
  (each teammate has their own quota → run independent jobs on 2 accounts in parallel if needed).
- One run ≤ **12 h**. Outputs: `/kaggle/working` ≤ 20 GB. Inputs mount read-only under `/kaggle/input/` — locate files
  with `glob('/kaggle/input/**/<name>', recursive=True)` (mount paths vary).
- Measured (25 Sep, MiniLM-L12-H384, max_len 96, mean 47 tokens, fp16, bs 128): **~600 pairs/s training per T4**,
  **~3.5–4k pairs/s scoring per T4**; the 1.7M-pair cross-fit run (2 models in parallel) took 31 min end to end.
- **Dataset version race** (hit 26 Sep): a kernel pushed right after `data` reports "ready" can still mount the
  PREVIOUS version. Check the kernel log's first line (`inputs: [...]`) — or wait ~2 min before `run`.
- Upload speed from the box matters: keep inputs compact (parquet, zstd, only the columns needed, text pre-normalized).

## Rules (hard)
- Dataset + kernel are **private**. Never `--public`, never share the link. Upload only derived pair text/ids we need,
  not the raw competition files. Delete them after the competition.
- Pretrained weights: MIT/Apache-2.0 and ≤ 8B only; verify the model card license, log it in `docs/decisions.md`.
  HF download needs internet ON (default in `run`). No other internet use inside the kernel (no external data).
- Labels in uploaded training data come from folds 1–4 only (train 2–4, early stop fold 1). Eval/test rows carry no labels.
- Kernel scripts are **self-contained** (don't import `ber`; copy the few helpers you need). Seed 42. Log throughput.
- Write checkpoints/outputs to `/kaggle/working` progressively (a crash at hour 5 must not lose hours 1–4).

## Job pattern (cross-encoder, Stage 5)
1. Box: `pipelines/xenc_export.py` → `data/kaggle/xenc/{train,valid,band_mini,band_fold0x,band_test}.parquet`
   (cols: `pair_id, text_a, text_b[, label]`; text = "name | address" normalized). Size check < ~2 GB.
2. `kaggle_gpu.py data data/kaggle/xenc --slug amlc-xenc-data`
3. `kaggle/xenc_train_score.py`: fine-tune `microsoft/Multilingual-MiniLM-L12-H384` (MIT) as a pair classifier,
   BCE, max_len 96, fp16 AMP, DDP or DataParallel over 2 T4, 1 epoch; **first measure throughput on 1% and print the
   ETA**; then score every band file → `/kaggle/working/logits_<band>.parquet (pair_id, xenc_logit)`.
4. `kaggle_gpu.py wait --slug amlc-xenc` → join logits back on `pair_id` → stage-2 feature `xenc_logit`
   → evaluate on mini / fold0x / LOCO like any other feature. Keep only if the paired bootstrap says so.
