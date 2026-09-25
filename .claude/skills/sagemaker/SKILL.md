---
name: sagemaker
description: Run heavy jobs (full-data blocking/features, GPU embedding or cross-encoder training/inference) on AWS SageMaker with per-member credits — instance choice, data sync via S3, running repo scripts, cost guards. Use when a job needs >12 GB RAM or more GPU than the laptop's 4 GB RTX 3050.
argument-hint: "<job description>"
---

# Run on SageMaker: $ARGUMENTS

## Check first (today, not at the deadline)
- **Service quotas**: new accounts often have a **0 quota for GPU instance types**. In Service Quotas → Amazon SageMaker, check the instance type you need (e.g. "ml.g5.2xlarge for notebook instance usage" / "for spaces" / "for training job usage"). Request increases now; approval can take hours.
- Credits: each member has their own account and credits. Spread heavy jobs across accounts. Stop idle instances. Set an AWS Budget alert at 50% and 80%.
- Region: pick one with g5 availability (e.g. us-east-1 / us-west-2 / ap-south-1). Keep the S3 bucket in the same region.

## Instance guide
| job | instance | notes |
|---|---|---|
| full blocking + features + LightGBM | ml.m5.4xlarge (16 vCPU, 64 GB) or ml.r5.4xlarge (128 GB) | CPU-bound; polars/rapidfuzz scale with cores |
| bi-encoder fine-tune / embed 20M strings | ml.g5.xlarge or ml.g5.2xlarge (A10G 24 GB) | fp16, batch 512+, max_len 64 |
| cross-encoder train + uncertain-band inference | ml.g5.2xlarge | mDeBERTa-v3-base, fp16 |
| quick checks | ml.t3.large | don't run pipelines here |

## Workflow (notebook instance or Studio JupyterLab space)
1. Local: push code to GitHub. Upload data once:
   `aws s3 sync student_resource/dataset s3://<bucket>/amlc/raw/`
   Optionally also `data/parquet` and `data/features` to skip preprocessing.
2. On the instance terminal:
   ```bash
   git clone <repo> amlc && cd amlc
   aws s3 sync s3://<bucket>/amlc/raw/ student_resource/dataset/
   python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
   python scripts/check_env.py && python scripts/prepare_data.py && python scripts/build_norm_cache.py --jobs 16
   nohup python pipelines/<x>.py --subset fold0 > logs/<x>.log 2>&1 &
   ```
   (Or set `AMLC_RAW_DIR` if the data lives elsewhere.)
3. Results: run records are written to `experiments/runs/`; commit and push them from the instance. Big artifacts go to S3:
   `aws s3 sync artifacts/<run_id> s3://<bucket>/amlc/artifacts/<run_id>`
   Pull locally only what you need, e.g. `test_matches.parquet` + `test_candidates.parquet` for `/submit`.
4. **Stop the instance when the job ends.** Or use a SageMaker Processing/Training job, which auto-terminates.

## Fallback: Kaggle notebook (free, 30 GB RAM, 4 CPU, optional T4/P100 GPU)
1. Upload `data/parquet/` + `data/features/` as a **private** Kaggle dataset. It is competition data, so never make it public.
2. In the notebook: `git clone` the repo, then set `AMLC_DATA_DIR=/kaggle/input/<dataset>` and `AMLC_ART_DIR=/kaggle/working/artifacts`, and `AMLC_RAW_DIR` if the raw data is needed.
3. Download `artifacts/<run_id>/test_*.parquet` and commit the run record back.

## Rules reminder
The same rules apply in the cloud: no external data or lookup APIs, and only MIT/Apache models ≤ 8B.
Downloading pretrained weights from Hugging Face is fine; downloading datasets is not.
