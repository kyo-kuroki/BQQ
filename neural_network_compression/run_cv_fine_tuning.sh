#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CV_DIR="${SCRIPT_DIR}/cv"

MODEL_NAME="${MODEL_NAME:-deit-s}" # Options: deit-s, deit-b, vit-s, vit-b, swin-t, swin-s
BIT_WIDTH="${BIT_WIDTH:-2}"
GROUP_SIZE="${GROUP_SIZE:-32}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-3}"
FINETUNE_LR="${FINETUNE_LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda:0}"
DATA_PATH="${DATA_PATH:-}" # Path to ImageNet. Empty => fall back to IMAGENET_DIR env var.

# Distillation (optional). Set TEACHER_MODEL_NAME to enable KL distillation.
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-}"
CE_ALPHA="${CE_ALPHA:-1.0}"
KL_ALPHA="${KL_ALPHA:-1.0}"
KL_TEMPERATURE="${KL_TEMPERATURE:-2.0}"

MODEL_PATH="${MODEL_PATH:-${CV_DIR}/src/quantized_models/${MODEL_NAME}/${MODEL_NAME}-blockwise.pth}"

cmd=(python "${CV_DIR}/fine_tuning.py"
  --model_name "${MODEL_NAME}"
  --model_path "${MODEL_PATH}"
  --bit_width "${BIT_WIDTH}"
  --group_size "${GROUP_SIZE}"
  --epochs "${FINETUNE_EPOCHS}"
  --lr "${FINETUNE_LR}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --device "${DEVICE}")

if [[ -n "${DATA_PATH}" ]]; then
  cmd+=(--data_path "${DATA_PATH}")
fi
if [[ -n "${TEACHER_MODEL_NAME}" ]]; then
  cmd+=(--teacher_model_name "${TEACHER_MODEL_NAME}"
        --ce_alpha "${CE_ALPHA}"
        --kl_alpha "${KL_ALPHA}"
        --kl_temperature "${KL_TEMPERATURE}")
fi

cmd+=("$@")
"${cmd[@]}"
