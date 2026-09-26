#!/bin/bash
# Upload the G3 scoring dataset (amlc-g3) to accounts 2, 3, 5 - one hardlinked dir per account (never share a dir)
cd /home/ubuntu/amlc
for N in 2 3 5; do
  D=data/kaggle/g3_up_acc$N; rm -rf "$D"; mkdir -p "$D"; ln data/kaggle/g3_up/pairs.parquet data/kaggle/g3_up/records.parquet "$D"/
  (source ~/kacc.sh $N; echo "[$(date +%T)] acc$N upload"; .venv/bin/python scripts/kaggle_gpu.py data "$D" --slug amlc-g3 -m "G3 pairs (norm_v2 text, md5 halves)" 2>&1 | grep -v "%|" | tail -n 3)
done
echo "[$(date +%T)] uploads done"
