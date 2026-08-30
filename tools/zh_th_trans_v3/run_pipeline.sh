#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?Usage: bash tools/zh_th_trans_v3/run_pipeline.sh /path/to/Qwen3-8B [batch-size]}"
BATCH_SIZE="${2:-4}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f outputs/zh_th_qwen3_8b_v3/candidates_24000.jsonl ]]; then
  python tools/zh_th_trans_v3/prepare_candidates.py
else
  echo "Candidates already exist; keeping them."
fi

if [[ -f outputs/zh_th_qwen3_8b_v3/raw_teacher_generations.jsonl ]]; then
  python tools/zh_th_trans_v3/generate_teacher.py \
    --model-name-or-path "$MODEL_PATH" \
    --batch-size "$BATCH_SIZE" \
    --resume
else
  python tools/zh_th_trans_v3/generate_teacher.py \
    --model-name-or-path "$MODEL_PATH" \
    --batch-size "$BATCH_SIZE"
fi

if [[ ! -f outputs/zh_th_qwen3_8b_v3/accepted.jsonl ]]; then
  python tools/zh_th_trans_v3/audit_teacher_outputs.py
elif [[ outputs/zh_th_qwen3_8b_v3/raw_teacher_generations.jsonl -nt outputs/zh_th_qwen3_8b_v3/accepted.jsonl ]]; then
  echo "Audit output is older than raw generation. Re-run audit explicitly with --overwrite." >&2
  exit 1
else
  echo "Audit outputs already exist; keeping them."
fi

if [[ ! -f dataset/final/zh_th/zh_th_qwen3_8b_v3.csv ]]; then
  python tools/zh_th_trans_v3/build_final_dataset.py
elif [[ outputs/zh_th_qwen3_8b_v3/accepted.jsonl -nt dataset/final/zh_th/zh_th_qwen3_8b_v3.csv ]]; then
  echo "Final CSV is older than accepted audit data. Re-run final build explicitly with --overwrite." >&2
  exit 1
else
  echo "Final v3 CSV already exists; keeping it."
fi
