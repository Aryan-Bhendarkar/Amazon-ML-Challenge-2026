#!/bin/bash
# Lane G / G2 launcher (box2). Uploads the G1 dataset to each account and pushes the training kernel.
# usage: bash kaggle/launch_lane_g.sh <acc N> <script> <slug> <gpu: l4x1|t4x2>   (falls back to t4x2 if l4x1 push fails)
set -u
cd /home/ubuntu/amlc
N=$1; SCRIPT=$2; SLUG=$3; GPU=$4
source ~/kacc.sh $N
echo "[$(date +%T)] acc$N: dataset amlc-g1"
.venv/bin/python scripts/kaggle_gpu.py data ${GDIR:-data/kaggle/g1} --slug amlc-g1 -m "G1 export" || exit 1
echo "[$(date +%T)] acc$N: push $SCRIPT as $SLUG on $GPU"
if ! .venv/bin/python scripts/kaggle_gpu.py run "$SCRIPT" --slug "$SLUG" --data amlc-g1 --gpu "$GPU"; then
  if [ "$GPU" != "t4x2" ]; then
    echo "[$(date +%T)] acc$N: $GPU push rejected -> fallback t4x2"
    .venv/bin/python scripts/kaggle_gpu.py run "$SCRIPT" --slug "$SLUG" --data amlc-g1 --gpu t4x2 || exit 1
  else exit 1; fi
fi
echo "[$(date +%T)] acc$N: pushed; polling in background"
.venv/bin/python scripts/kaggle_gpu.py wait --slug "$SLUG" --poll 120 --out "artifacts/kaggle/$SLUG"
.venv/bin/python scripts/kaggle_gpu.py pull --slug "$SLUG" --out "artifacts/kaggle/$SLUG"
echo "[$(date +%T)] acc$N: finished"
