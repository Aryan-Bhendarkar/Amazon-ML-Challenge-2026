#!/bin/bash
# Teammate fast-start: pull Aryan's prepared data + caches + baseline model instead of rebuilding.
# Usage (on your own dev box, inside ~/amlc):  AMLC_DATA_BUCKET=amlc26-699191579023 bash infra/aws/pull_share.sh
set -euo pipefail
SRC="s3://${AMLC_DATA_BUCKET:?set AMLC_DATA_BUCKET to the team data bucket}/share"
cd ~/amlc
aws s3 sync "$SRC/data/parquet"  data/parquet  --only-show-errors
aws s3 sync "$SRC/data/splits"   data/splits   --only-show-errors
aws s3 sync "$SRC/data/features" data/features --only-show-errors
aws s3 sync "$SRC/data/cands"    data/cands    --only-show-errors
aws s3 sync "$SRC/artifacts"     artifacts     --only-show-errors
echo "pulled: $(du -sh data | cut -f1) data, $(du -sh artifacts | cut -f1) artifacts"
echo "Smoke test: .venv/bin/python pipelines/train_eval.py --cand-ver keys_v0 --subset micro --name smoke-<you>"
