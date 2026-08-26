#!/usr/bin/env bash
# Phase 1: compare learning rates with matched rank-16 Pilot runs.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_ROOT"
for config in \
  configs/qwen3_8b_qlora_pilot_p1.json \
  configs/qwen3_8b_qlora_pilot_p2.json \
  configs/qwen3_8b_qlora_pilot_p3.json
do
  echo "=== Starting ${config} ==="
  "$PYTHON_BIN" tools/training/train_qlora.py --config "$config" --resume-from-checkpoint none
done
