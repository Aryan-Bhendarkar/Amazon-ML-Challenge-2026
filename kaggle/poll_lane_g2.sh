#!/bin/bash
# Lane G poller v2: training (G2) -> pull ckpts -> push the G3 scoring kernel once -> pull logits. Log: logs/lane_g_status.log
cd /home/ubuntu/amlc
JOBS=("2 ashu273k amlc-g2-xlmr-h0 kaggle/xenc_g3_xlmr_h0.py amlc-g3-xlmr-h0"
      "3 darshanbagade amlc-g2-xlmr-h1 kaggle/xenc_g3_xlmr_h1.py amlc-g3-xlmr-h1"
      "5 darshanbagadeycce amlc-g2-minilm kaggle/xenc_g3_minilm.py amlc-g3-minilm")
st_of() { (source ~/kacc.sh $1; kaggle kernels status $2/$3 2>&1 | tail -1) | sed 's/.*KernelWorkerStatus\.//;s/"//g'; }
while true; do
  for j in "${JOBS[@]}"; do
    set -- $j; N=$1; U=$2; TR=$3; SC=$4; G3=$5
    st=$(st_of $N $U $TR); echo "$(date '+%F %T') acc$N $TR $st" >> logs/lane_g_status.log
    case "$st" in COMPLETE*|ERROR*|CANCEL*)
      if [ ! -f artifacts/kaggle/$TR/.pulled ]; then
        (source ~/kacc.sh $N; .venv/bin/python scripts/kaggle_gpu.py pull --slug $TR --out artifacts/kaggle/$TR >> logs/lane_g_pull.log 2>&1)
        mkdir -p artifacts/kaggle/$TR; touch artifacts/kaggle/$TR/.pulled
        (cd artifacts/kaggle/$TR && sha256sum ckpt_best_h*.pt > ckpt_sha256.txt 2>/dev/null)
      fi
      ready=$( (source ~/kacc.sh $N; kaggle datasets status $U/amlc-g3 2>&1) | grep -ci ready)
      if ls artifacts/kaggle/$TR/ckpt_best_h*.pt >/dev/null 2>&1 && [ ! -f artifacts/kaggle/$TR/.g3pushed ] && [ "$ready" -gt 0 ]; then
        echo "$(date '+%F %T') acc$N push $G3" >> logs/lane_g_status.log
        (source ~/kacc.sh $N; .venv/bin/python scripts/kaggle_gpu.py run $SC --slug $G3 --data amlc-g3 --kernels $U/$TR --gpu l4x1 \
           || .venv/bin/python scripts/kaggle_gpu.py run $SC --slug $G3 --data amlc-g3 --kernels $U/$TR --gpu t4x2) >> logs/lane_g_push.log 2>&1 \
          && touch artifacts/kaggle/$TR/.g3pushed
      fi
      if [ -f artifacts/kaggle/$TR/.g3pushed ]; then
        s3=$(st_of $N $U $G3); echo "$(date '+%F %T') acc$N $G3 $s3" >> logs/lane_g_status.log
        case "$s3" in COMPLETE*|ERROR*|CANCEL*)
          [ -f artifacts/kaggle/$G3/.pulled ] || { (source ~/kacc.sh $N; .venv/bin/python scripts/kaggle_gpu.py pull --slug $G3 --out artifacts/kaggle/$G3 >> logs/lane_g_pull.log 2>&1); mkdir -p artifacts/kaggle/$G3; touch artifacts/kaggle/$G3/.pulled; } ;;
        esac
      fi ;;
    esac
  done
  sleep 180
done
