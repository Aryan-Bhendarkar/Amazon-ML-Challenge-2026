#!/bin/bash
# Prints where Claude Code is running (used by the /cloud skill).
if [ -f /var/log/amlc-bootstrap.done ]; then
  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)
  echo "ON CLOUD BOX: $(nproc) vCPU, $(free -g | awk '/Mem/{print $2}') GB RAM, GPU: ${gpu}, keepalive: $([ -f ~/.keepalive ] && echo ON || echo off)"
else
  echo "NOT on the cloud box (probably the Windows laptop, ~5 GB free RAM): keep jobs tiny or move them to amlc-box"
fi
