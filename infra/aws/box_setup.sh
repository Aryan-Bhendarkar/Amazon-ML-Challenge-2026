#!/bin/bash
# Project setup ON the dev box (run once after first SSH; interactive for GitHub login). Idempotent.
#   bash ~/box_setup.sh
# Teammates using Aryan's shared data bucket:  AMLC_DATA_BUCKET=amlc26-<aryan-account-id> bash ~/box_setup.sh
set -euo pipefail
REPO="${AMLC_REPO:-Aryan-Bhendarkar/Amazon-ML-Challenge-2026}"
DATA_BUCKET="${AMLC_DATA_BUCKET:-${AMLC_BUCKET:-}}"
export PATH="$HOME/.local/bin:$PATH" PYTHONUTF8=1

echo "== 1/4 GitHub + repo"
if [ ! -d ~/amlc/.git ]; then
  gh auth status >/dev/null 2>&1 || gh auth login --hostname github.com --git-protocol https --web
  gh auth setup-git
  gh repo clone "$REPO" ~/amlc
fi
cd ~/amlc
git config user.name >/dev/null 2>&1 || git config user.name "${AMLC_AUTHOR:-$(gh api user -q .login 2>/dev/null || whoami)}"
git config user.email >/dev/null 2>&1 || git config user.email "$(gh api user -q .email 2>/dev/null || echo "$(whoami)@users.noreply.github.com")"

echo "== 2/4 Python env (3.13, same pins as the laptop)"
[ -x .venv/bin/python ] || uv venv --python 3.13 .venv
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then TORCH_IDX=https://download.pytorch.org/whl/cu124; else TORCH_IDX=https://download.pytorch.org/whl/cpu; fi
uv pip install --python .venv/bin/python "torch==2.6.0" --index-url "$TORCH_IDX"
uv pip install --python .venv/bin/python -r requirements.txt

echo "== 3/4 Data"
if [ ! -f student_resource/dataset/test/test_source1.tsv ]; then
  [ -n "$DATA_BUCKET" ] || { echo "set AMLC_DATA_BUCKET"; exit 1; }
  aws s3 cp "s3://$DATA_BUCKET/raw/student_resource.tar.gz" /tmp/sr.tar.gz
  tar -xzf /tmp/sr.tar.gz -C ~/amlc && rm -f /tmp/sr.tar.gz
fi

echo "== 4/4 Prepare caches + tests"
.venv/bin/python scripts/check_env.py
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/build_norm_cache.py --jobs "$(nproc)"
.venv/bin/python -m pytest -q
echo
echo "READY. Start Claude Code:   cd ~/amlc && claude      (first run prints a login URL -> open it on your laptop)"
