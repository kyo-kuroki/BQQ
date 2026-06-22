#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM_DIR="${SCRIPT_DIR}/lm"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-2B}"
BLOCK_IDX="${BLOCK_IDX:-0}"
BIT_WIDTH="${BIT_WIDTH:-2}"
GROUP_SIZE="${GROUP_SIZE:-64}"

LAYERWISE_ANNEAL_STEPS="${LAYERWISE_ANNEAL_STEPS:-10000}"
LAYERWISE_STE_STEPS="${LAYERWISE_STE_STEPS:-1000}"
LAYERWISE_STE_LR="${LAYERWISE_STE_LR:-1e-3}"
LAYERWISE_STE_WEIGHT_DECAY="${LAYERWISE_STE_WEIGHT_DECAY:-0.0}"
LAYERWISE_STE_LOG_INTERVAL="${LAYERWISE_STE_LOG_INTERVAL:-20}"

BLOCKWISE_EPOCHS="${BLOCKWISE_EPOCHS:-1}"
BLOCKWISE_LR="${BLOCKWISE_LR:-1e-4}"
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

python "${LM_DIR}/layerwise_quant.py"     --model_name "${MODEL_NAME}"     --block_idx "${BLOCK_IDX}"     --bit_width "${BIT_WIDTH}"     --group_size "${GROUP_SIZE}"     --num_steps "${LAYERWISE_ANNEAL_STEPS}"     --ste_refine_steps "${LAYERWISE_STE_STEPS}"     --ste_refine_lr "${LAYERWISE_STE_LR}"     --ste_refine_weight_decay "${LAYERWISE_STE_WEIGHT_DECAY}"     --ste_refine_log_interval "${LAYERWISE_STE_LOG_INTERVAL}"     --dataset "${DATASET}"     --nsamples "${NSAMPLES}"     --seqlen "${SEQLEN}"     --seed "${SEED}"     --save_dir "${LAYERWISE_DIR}"     --assembled_output_dir "${ASSEMBLED_OUTPUT_DIR}"

python "${LM_DIR}/blockwise_quant.py"     --model_name "${MODEL_NAME}"     --block_idx "${BLOCK_IDX}"     --bit_width "${BIT_WIDTH}"     --group_size "${GROUP_SIZE}"     --num_steps "${LAYERWISE_ANNEAL_STEPS}"     --dataset "${DATASET}"     --nsamples "${NSAMPLES}"     --seqlen "${SEQLEN}"     --seed "${SEED}"     --epochs "${BLOCKWISE_EPOCHS}"     --lr "${BLOCKWISE_LR}"     --max_grad_norm "${BLOCKWISE_MAX_GRAD_NORM}"     --device "${DEVICE}"     --layerwise_dir "${LAYERWISE_DIR}"     --save_dir "${BLOCKWISE_SAVE_DIR}"     --assembled_output_dir "${ASSEMBLED_OUTPUT_DIR}"     "$@"
