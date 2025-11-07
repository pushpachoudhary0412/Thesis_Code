#!/usr/bin/env bash
set -euo pipefail

# Minimal smoke test: generate dev data, run one epoch of training, verify artifacts.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}"

RUN_DIR="mimiciv_backdoor_study/runs/mlp/none/0.0/seed_42"

echo "1) Generate deterministic dev data"
CANDIDATE_02_SAMPLE="${REPO_ROOT}/mimiciv_backdoor_study/scripts/02_sample_dev.py"
CANDIDATE_02_SAMPLE_ALT="${REPO_ROOT}/scripts/02_sample_dev.py"
if [ -f "${CANDIDATE_02_SAMPLE}" ]; then
  SAMPLE_SCRIPT="${CANDIDATE_02_SAMPLE}"
elif [ -f "${CANDIDATE_02_SAMPLE_ALT}" ]; then
  SAMPLE_SCRIPT="${CANDIDATE_02_SAMPLE_ALT}"
else
  echo "ERROR: expected file not found: ${CANDIDATE_02_SAMPLE} or ${CANDIDATE_02_SAMPLE_ALT}"
  echo "Repository listing (for debugging):"
  ls -al "${REPO_ROOT}"
  echo "Listing mimiciv_backdoor_study (for debugging):"
  ls -al "${REPO_ROOT}/mimiciv_backdoor_study" || true
  echo "Listing scripts (for debugging):"
  ls -al "${REPO_ROOT}/scripts" || true
  exit 1
fi
python "${SAMPLE_SCRIPT}"

echo "2) Run one-epoch training (fast)"
CANDIDATE_TRAIN="${REPO_ROOT}/mimiciv_backdoor_study/train.py"
CANDIDATE_TRAIN_ALT="${REPO_ROOT}/train.py"
if [ -f "${CANDIDATE_TRAIN}" ]; then
  TRAIN_SCRIPT="${CANDIDATE_TRAIN}"
elif [ -f "${CANDIDATE_TRAIN_ALT}" ]; then
  TRAIN_SCRIPT="${CANDIDATE_TRAIN_ALT}"
else
  echo "ERROR: expected train script not found: ${CANDIDATE_TRAIN} or ${CANDIDATE_TRAIN_ALT}"
  ls -al "${REPO_ROOT}" || true
  exit 1
fi
python "${TRAIN_SCRIPT}" --model mlp --trigger none --poison_rate 0.0 --epochs 1

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
