#!/bin/bash
# Lane G status poller: one line per account every 2 min -> logs/lane_g_status.log (no waiting in the foreground)
cd /home/ubuntu/amlc
while true; do
  for j in "2 ashu273k amlc-g2-xlmr-h0" "3 darshanbagade amlc-g2-xlmr-h1" "5 darshanbagadeycce amlc-g2-minilm"; do
    set -- $j
    st=$( (source ~/kacc.sh $1; kaggle kernels status $2/$3 2>&1 | tail -1) | sed 's/.*KernelWorkerStatus\.//;s/"//g')
    echo "$(date '+%F %T') acc$1 $3 $st" >> logs/lane_g_status.log
    case "$st" in COMPLETE*|ERROR*|CANCEL*)
      [ -f artifacts/kaggle/$3/.pulled ] || { (source ~/kacc.sh $1; .venv/bin/python scripts/kaggle_gpu.py pull --slug $3 --out artifacts/kaggle/$3 >> logs/lane_g_pull.log 2>&1); touch artifacts/kaggle/$3/.pulled; } ;;
    esac
  done
  sleep 120
done
