---
name: cloud
description: Run and manage work on the team's AWS EC2 dev/GPU boxes (ap-south-1) — where Claude Code should run heavy jobs, how to run long jobs safely (tmux, keepalive vs auto-stop), resizing, GPU box usage, sharing artifacts via S3, and credit/cost hygiene. Use for any job that needs more than a laptop, or any question about the cloud setup.
argument-hint: "<job or question>"
---

# Cloud operations: $ARGUMENTS

Setup and runbook: `infra/aws/README.md`. We use EC2 boxes (not SageMaker), with Claude Code running ON the box.

## Where am I?
!`bash infra/aws/whereami.sh`

## Sizing guide (the laptop has ~5 GB free RAM: dev/micro only)
| job | where |
|---|---|
| micro/mini experiments, feature dev, LightGBM on ≤ 10M pairs | `amlc-box` r7i.xlarge (32 GB) |
| full fold0 / full test inference / 50M+ pair joins | resize to r7i.2xlarge or 4xlarge (needs quota); run inside tmux |
| bi-/cross-encoder training + inference | `amlc-gpu` g6.xlarge (L4 24 GB), or a Kaggle 2×T4 notebook if there's no GPU quota |

## Running long jobs on the box
1. Estimate first: time the `micro` run and extrapolate. Say the estimate before launching.
2. Start it detached so it survives disconnects:
   `tmux new -d -s <name> ".venv/bin/python -u pipelines/<x>.py ... 2>&1 | tee logs/<name>.log"`
3. Poll `tail -n 20 logs/<name>.log`; attach with `tmux attach -t <name>`.
4. Auto-stop shuts the box down after 20 min with **no SSH session + idle CPU**. A running job keeps the CPU busy, but if a job might idle (e.g. waiting on I/O) with nobody connected, `touch ~/.keepalive` and **`rm ~/.keepalive` when done** (credits!).
5. Watch memory: `free -g`. The 16 GB swap is a safety net, not a plan. Chunk/stream if RSS > 70% RAM.

## Artifacts and team sharing (S3, same region = free transfer)
- Save big outputs in `artifacts/<run_id>/`, then `aws s3 sync artifacts/<run_id> s3://$AMLC_BUCKET/artifacts/<run_id>`.
- A teammate's shared bucket: `aws s3 sync s3://amlc26-<their-account>/artifacts/<run_id> artifacts/<run_id>`.
- Run records (`experiments/runs/*`) go through **git** (commit + push), not S3.
- Never make a bucket public (competition data).

## Cost hygiene (each member has ~$100–200)
- A box costs money only while running. Stop it when done: from the laptop `infra\aws\devbox.ps1 stop`, or on the box `sudo shutdown -h now` (the instance stops, the disk is kept).
- GPU box: start → run the job → stop. Never leave it idle.
- Before a big resize or GPU job, state the expected hours × $/h.
- Rules still apply in the cloud: no external data or lookup APIs; pretrained weights only if MIT/Apache and ≤ 8B params.
