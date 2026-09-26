#!/bin/bash
# Lane G relaunch (push only; dataset amlc-g1 already exists on the account). usage: relaunch_g2.sh N script slug gpu
set -u; cd /home/ubuntu/amlc
N=$1; SCRIPT=$2; SLUG=$3; GPU=$4
source ~/kacc.sh $N
echo "[$(date +%T)] acc$N: push $SCRIPT as $SLUG on $GPU"
.venv/bin/python scripts/kaggle_gpu.py run "$SCRIPT" --slug "$SLUG" --data amlc-g1 --gpu "$GPU" || exit 1
.venv/bin/python scripts/kaggle_gpu.py wait --slug "$SLUG" --poll 120 --out "artifacts/kaggle/$SLUG"
echo "[$(date +%T)] acc$N: finished"
