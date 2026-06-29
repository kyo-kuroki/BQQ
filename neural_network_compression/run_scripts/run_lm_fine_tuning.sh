#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/lm"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-2B}"
BIT_WIDTH="${BIT_WIDTH:-2}"
GROUP_SIZE="${GROUP_SIZE:-64}"
FINETUNE_NUM_STEPS="${FINETUNE_NUM_STEPS:-10000}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-3}"
FINETUNE_LR="${FINETUNE_LR:-2e-5}"
FINETUNE_BINARY_LR="${FINETUNE_BINARY_LR:-1e-3}"
FINETUNE_CONTINUOUS_LR="${FINETUNE_CONTINUOUS_LR:-1e-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
FINETUNE_FIX_THETA="${FINETUNE_FIX_THETA:-0}"
FINETUNE_FIX_BETA="${FINETUNE_FIX_BETA:-0}"

MODEL_BASENAME="${MODEL_NAME##*/}"
MODEL_PATH="${MODEL_PATH:-${LM_DIR}/src/quantized_models/${MODEL_BASENAME}/${MODEL_BASENAME}-${BIT_WIDTH}bit-${GROUP_SIZE}gs-blockwise.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${LM_DIR}/fine_tuned_models/${MODEL_BASENAME}}"

cmd=(python "${LM_DIR}/fine_tuning.py"
  --model_name "${MODEL_NAME}"
  --model_path "${MODEL_PATH}"
  --bit_width "${BIT_WIDTH}"
  --group_size "${GROUP_SIZE}"
  --num_steps "${FINETUNE_NUM_STEPS}"
  --output_dir "${OUTPUT_DIR}"
  --num_train_epochs "${FINETUNE_EPOCHS}"
  --learning_rate "${FINETUNE_LR}"
  --binary_learning_rate "${FINETUNE_BINARY_LR}"
  --continuous_learning_rate "${FINETUNE_CONTINUOUS_LR}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --max_seq_length "${MAX_SEQ_LENGTH}")

if [[ "${FINETUNE_FIX_THETA}" == "1" ]]; then
  cmd+=(--fix_theta)
fi
if [[ "${FINETUNE_FIX_BETA}" == "1" ]]; then
  cmd+=(--fix_beta)
fi
cmd+=("$@")
"${cmd[@]}"
