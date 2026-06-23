#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM_DIR="${SCRIPT_DIR}/lm"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-2B}"
BLOCK_IDX="${BLOCK_IDX:-all}"
BIT_WIDTH="${BIT_WIDTH:-2}"
GROUP_SIZE="${GROUP_SIZE:-64}"

LAYERWISE_ANNEAL_STEPS="${LAYERWISE_ANNEAL_STEPS:-10000}"
LAYERWISE_STE_STEPS="${LAYERWISE_STE_STEPS:-0}"
LAYERWISE_STE_LR="${LAYERWISE_STE_LR:-1e-3}"
LAYERWISE_STE_WEIGHT_DECAY="${LAYERWISE_STE_WEIGHT_DECAY:-0.0}"
LAYERWISE_STE_BINARY_LR="${LAYERWISE_STE_BINARY_LR:-1e-3}"
LAYERWISE_STE_CONTINUOUS_LR="${LAYERWISE_STE_CONTINUOUS_LR:-1e-4}"
LAYERWISE_STE_LOG_INTERVAL="${LAYERWISE_STE_LOG_INTERVAL:-20}"

BLOCKWISE_EPOCHS="${BLOCKWISE_EPOCHS:-5}"
BLOCKWISE_LR="${BLOCKWISE_LR:-1e-4}"
BLOCKWISE_BINARY_LR="${BLOCKWISE_BINARY_LR:-1e-3}"
BLOCKWISE_CONTINUOUS_LR="${BLOCKWISE_CONTINUOUS_LR:-1e-4}"
BLOCKWISE_OPTIMIZER="${BLOCKWISE_OPTIMIZER:-sgd}"
BLOCKWISE_MOMENTUM="${BLOCKWISE_MOMENTUM:-0.9}"
BLOCKWISE_MAX_GRAD_NORM="${BLOCKWISE_MAX_GRAD_NORM:-1.0}"

DATASET="${DATASET:-slimpajama}"
NSAMPLES="${NSAMPLES:-1024}"
SEQLEN="${SEQLEN:-2048}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"

MODEL_BASENAME="${MODEL_NAME##*/}"
LAYERWISE_DIR="${LAYERWISE_DIR:-${LM_DIR}/src/bqq_compressed_data/${MODEL_BASENAME}-${GROUP_SIZE}gs-${LAYERWISE_ANNEAL_STEPS}step}"
BLOCKWISE_SAVE_DIR="${BLOCKWISE_SAVE_DIR:-${LM_DIR}/blockwise_output/${MODEL_BASENAME}}"
ASSEMBLED_OUTPUT_DIR="${ASSEMBLED_OUTPUT_DIR:-${LM_DIR}/src/quantized_models/${MODEL_BASENAME}}"

run_one_block() {
  local block_idx="$1"
  local should_assemble="$2"
  shift 2

  local assemble_args=(--no_assemble_full_model)
  if [[ "${should_assemble}" == "1" ]]; then
    assemble_args=(--assembled_output_dir "${ASSEMBLED_OUTPUT_DIR}")
  fi

  python "${LM_DIR}/layerwise_quant.py" \
    --model_name "${MODEL_NAME}" \
    --block_idx "${block_idx}" \
    --bit_width "${BIT_WIDTH}" \
    --group_size "${GROUP_SIZE}" \
    --num_steps "${LAYERWISE_ANNEAL_STEPS}" \
    --ste_refine_steps "${LAYERWISE_STE_STEPS}" \
    --ste_refine_lr "${LAYERWISE_STE_LR}" \
    --ste_refine_binary_lr "${LAYERWISE_STE_BINARY_LR}" \
    --ste_refine_continuous_lr "${LAYERWISE_STE_CONTINUOUS_LR}" \
    --ste_refine_weight_decay "${LAYERWISE_STE_WEIGHT_DECAY}" \
    --ste_refine_log_interval "${LAYERWISE_STE_LOG_INTERVAL}" \
    --dataset "${DATASET}" \
    --nsamples "${NSAMPLES}" \
    --seqlen "${SEQLEN}" \
    --seed "${SEED}" \
    --save_dir "${LAYERWISE_DIR}" \
    "${assemble_args[@]}"

  python "${LM_DIR}/blockwise_quant.py" \
    --fix_theta \
    --model_name "${MODEL_NAME}" \
    --block_idx "${block_idx}" \
    --bit_width "${BIT_WIDTH}" \
    --group_size "${GROUP_SIZE}" \
    --num_steps "${LAYERWISE_ANNEAL_STEPS}" \
    --dataset "${DATASET}" \
    --nsamples "${NSAMPLES}" \
    --seqlen "${SEQLEN}" \
    --seed "${SEED}" \
    --epochs "${BLOCKWISE_EPOCHS}" \
    --optimizer "${BLOCKWISE_OPTIMIZER}" \
    --momentum "${BLOCKWISE_MOMENTUM}" \
    --lr "${BLOCKWISE_LR}" \
    --binary_lr "${BLOCKWISE_BINARY_LR}" \
    --continuous_lr "${BLOCKWISE_CONTINUOUS_LR}" \
    --max_grad_norm "${BLOCKWISE_MAX_GRAD_NORM}" \
    --device "${DEVICE}" \
    --layerwise_dir "${LAYERWISE_DIR}" \
    --save_dir "${BLOCKWISE_SAVE_DIR}" \
    "${assemble_args[@]}" \
    "$@"
}

if [[ "${BLOCK_IDX}" == "all" ]]; then
  NUM_BLOCKS="$(python "${LM_DIR}/layerwise_quant.py" --model_name "${MODEL_NAME}" --list_blocks)"
  LAST_BLOCK=$((NUM_BLOCKS - 1))
  for ((block_idx=0; block_idx<NUM_BLOCKS; block_idx++)); do
    should_assemble=0
    if [[ "${block_idx}" -eq "${LAST_BLOCK}" ]]; then
      should_assemble=1
    fi
    run_one_block "${block_idx}" "${should_assemble}" "$@"
  done
else
  run_one_block "${BLOCK_IDX}" 1 "$@"
fi
