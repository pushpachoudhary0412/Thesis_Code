#!/usr/bin/env bash
set -euo pipefail

# Minimal smoke test: generate dev data, run one epoch of training, verify artifacts.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}"

RUN_DIR="mimiciv_backdoor_study/runs/mlp/none/0.0/seed_42"

echo "1) Generate deterministic dev data"
python "${REPO_ROOT}/mimiciv_backdoor_study/scripts/02_sample_dev.py"

echo "2) Run one-epoch training (fast)"
python "${REPO_ROOT}/mimiciv_backdoor_study/train.py" --model mlp --trigger none --poison_rate 0.0 --epochs 1

echo "3) Verify run artifacts exist"
if [ ! -f "${RUN_DIR}/model.pt" ]; then
  echo "ERROR: model.pt not found at ${RUN_DIR}"
  exit 2
fi

if [ ! -f "${RUN_DIR}/results.json" ]; then
  echo "ERROR: results.json not found at ${RUN_DIR}"
  exit 3
fi

echo "Smoke test passed: model.pt and results.json present in ${RUN_DIR}"
