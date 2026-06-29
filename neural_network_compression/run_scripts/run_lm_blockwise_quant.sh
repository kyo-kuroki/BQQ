#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/lm"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}" # Example Options: "Qwen/Qwen3.5-4B", "meta-llama/Llama-3.1-8B"
BLOCK_IDX="${BLOCK_IDX:-all}" # Options: "all" (process all blocks), or a specific block index (e.g., 0, 1, 2, ...)
BLOCKS_PER_GPU="${BLOCKS_PER_GPU:-1}"
BIT_WIDTH="${BIT_WIDTH:-1}"
GROUP_SIZE="${GROUP_SIZE:-64}"

LAYERWISE_ANNEAL_STEPS="${LAYERWISE_ANNEAL_STEPS:-20000}"
LAYERWISE_STE_STEPS="${LAYERWISE_STE_STEPS:-0}"
LAYERWISE_STE_LR="${LAYERWISE_STE_LR:-1e-3}"
LAYERWISE_STE_WEIGHT_DECAY="${LAYERWISE_STE_WEIGHT_DECAY:-0.0}"
LAYERWISE_STE_BINARY_LR="${LAYERWISE_STE_BINARY_LR:-1e-3}"
LAYERWISE_STE_CONTINUOUS_LR="${LAYERWISE_STE_CONTINUOUS_LR:-1e-4}"
LAYERWISE_STE_LOG_INTERVAL="${LAYERWISE_STE_LOG_INTERVAL:-20}"
LAYERWISE_WORKERS_PER_GPU="${LAYERWISE_WORKERS_PER_GPU:-8}"
LAYERWISE_FIX_THETA="${LAYERWISE_FIX_THETA:-0}"
LAYERWISE_FIX_BETA="${LAYERWISE_FIX_BETA:-0}"

BLOCKWISE_EPOCHS="${BLOCKWISE_EPOCHS:-1}"
BLOCKWISE_LR="${BLOCKWISE_LR:-1e-4}"
BLOCKWISE_BINARY_LR="${BLOCKWISE_BINARY_LR:-0}"
BLOCKWISE_CONTINUOUS_LR="${BLOCKWISE_CONTINUOUS_LR:-5e-6}"
BLOCKWISE_OPTIMIZER="${BLOCKWISE_OPTIMIZER:-adamw}" # Options: "sgd", "adam", "adamw"
BLOCKWISE_MOMENTUM="${BLOCKWISE_MOMENTUM:-0.9}"
BLOCKWISE_MAX_GRAD_NORM="${BLOCKWISE_MAX_GRAD_NORM:-1.0}"
BLOCKWISE_TUNE_BATCH_SIZE="${BLOCKWISE_TUNE_BATCH_SIZE:-8}"
BLOCKWISE_FIX_THETA="${BLOCKWISE_FIX_THETA:-0}"
BLOCKWISE_FIX_BETA="${BLOCKWISE_FIX_BETA:-0}"
PROGRESSIVE="${PROGRESSIVE:-1}" # Options: "0" (disabled), "1" (enabled)
PROGRESSIVE_MODE="${PROGRESSIVE_MODE:-layer-tune}"  # Options: "layer-tune", "closed-form-layer", "patch"

DATASET="${DATASET:-slimpajama}"
NSAMPLES="${NSAMPLES:-1024}"
SEQLEN="${SEQLEN:-2048}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
NO_IO_CACHE="${NO_IO_CACHE:-1}"
NO_SCALE_REFINE="${NO_SCALE_REFINE:-1}"
USE_MULTIBQQ="${USE_MULTIBQQ:-1}"
COMPENSATION_MODE="${COMPENSATION_MODE:-ldlq}"
STE_REFINE="${STE_REFINE:-0}"
STE_REFINE_STEPS="${STE_REFINE_STEPS:-200}"
STE_REFINE_WEIGHT_DECAY="${STE_REFINE_WEIGHT_DECAY:-0.0}"
STE_REFINE_BINARY_LR="${STE_REFINE_BINARY_LR:-1e-5}"
STE_REFINE_CONTINUOUS_LR="${STE_REFINE_CONTINUOUS_LR:-1e-5}"
STE_REFINE_LOG_INTERVAL="${STE_REFINE_LOG_INTERVAL:-20}"
STE_REFINE_ROW_GROUP_BATCH_SIZE="${STE_REFINE_ROW_GROUP_BATCH_SIZE:-}"

MODEL_BASENAME="${MODEL_NAME##*/}"
LAYERWISE_DIR="${LAYERWISE_DIR:-${LM_DIR}/src/bqq_compressed_data/${MODEL_BASENAME}-${BIT_WIDTH}bit-${GROUP_SIZE}gs-${LAYERWISE_ANNEAL_STEPS}step}"
BLOCKWISE_SAVE_DIR="${BLOCKWISE_SAVE_DIR:-${LM_DIR}/blockwise_output/${MODEL_BASENAME}}"
ASSEMBLED_OUTPUT_DIR="${ASSEMBLED_OUTPUT_DIR:-${LM_DIR}/src/quantized_models/${MODEL_BASENAME}}"
IO_CACHE_DIR="${IO_CACHE_DIR:-${LM_DIR}/src/block_io_cache/${MODEL_BASENAME}/${DATASET}_${SEQLEN}seqlen_${NSAMPLES}samples}"

run_one_block() {
  local block_idx="$1"
  local should_assemble="$2"
  shift 2

  local runtime_device="${RUNTIME_DEVICE:-${DEVICE}}"
  local assemble_args=(--no_assemble_full_model)
  if [[ "${should_assemble}" == "1" ]]; then
    assemble_args=(--assembled_output_dir "${ASSEMBLED_OUTPUT_DIR}")
  fi

  if [[ "${PROGRESSIVE}" != "1" ]]; then
    local layerwise_cmd=(
      python "${LM_DIR}/layerwise_quant.py"
      --model_name "${MODEL_NAME}"
      --block_idx "${block_idx}"
      --bit_width "${BIT_WIDTH}"
      --group_size "${GROUP_SIZE}"
      --num_steps "${LAYERWISE_ANNEAL_STEPS}"
      --ste_refine_steps "${LAYERWISE_STE_STEPS}"
      --ste_refine_lr "${LAYERWISE_STE_LR}"
      --ste_refine_binary_lr "${LAYERWISE_STE_BINARY_LR}"
      --ste_refine_continuous_lr "${LAYERWISE_STE_CONTINUOUS_LR}"
      --ste_refine_weight_decay "${LAYERWISE_STE_WEIGHT_DECAY}"
      --ste_refine_log_interval "${LAYERWISE_STE_LOG_INTERVAL}"
      --workers_per_gpu "${LAYERWISE_WORKERS_PER_GPU}"
      --dataset "${DATASET}"
      --nsamples "${NSAMPLES}"
      --seqlen "${SEQLEN}"
      --seed "${SEED}"
      --save_dir "${LAYERWISE_DIR}"
      --compensation_mode "${COMPENSATION_MODE}"
      "${assemble_args[@]}"
    )
    if [[ "${USE_MULTIBQQ}" == "1" ]]; then
      layerwise_cmd+=(--use_multibqq)
    else
      layerwise_cmd+=(--no_use_multibqq)
    fi
    if [[ "${LAYERWISE_FIX_THETA}" == "1" ]]; then
      layerwise_cmd+=(--fix_theta)
    fi
    if [[ "${LAYERWISE_FIX_BETA}" == "1" ]]; then
      layerwise_cmd+=(--fix_beta)
    fi
    layerwise_cmd+=("$@")
    "${layerwise_cmd[@]}"
  fi

  local blockwise_cmd=(
    python "${LM_DIR}/blockwise_quant.py"
    --model_name "${MODEL_NAME}"
    --block_idx "${block_idx}"
    --bit_width "${BIT_WIDTH}"
    --group_size "${GROUP_SIZE}"
    --num_steps "${LAYERWISE_ANNEAL_STEPS}"
    --dataset "${DATASET}"
    --nsamples "${NSAMPLES}"
    --seqlen "${SEQLEN}"
    --seed "${SEED}"
    --epochs "${BLOCKWISE_EPOCHS}"
    --optimizer "${BLOCKWISE_OPTIMIZER}"
    --momentum "${BLOCKWISE_MOMENTUM}"
    --lr "${BLOCKWISE_LR}"
    --binary_lr "${BLOCKWISE_BINARY_LR}"
    --continuous_lr "${BLOCKWISE_CONTINUOUS_LR}"
    --max_grad_norm "${BLOCKWISE_MAX_GRAD_NORM}"
    --tune_batch_size "${BLOCKWISE_TUNE_BATCH_SIZE}"
    --device "${runtime_device}"
    --layerwise_dir "${LAYERWISE_DIR}"
    --io_cache_dir "${IO_CACHE_DIR}"
    --save_dir "${BLOCKWISE_SAVE_DIR}"
    "${assemble_args[@]}"
  )
  if [[ "${USE_MULTIBQQ}" == "1" ]]; then
    blockwise_cmd+=(--use_multibqq)
  else
    blockwise_cmd+=(--no_use_multibqq)
  fi
  blockwise_cmd+=(--compensation_mode "${COMPENSATION_MODE}")
  if [[ "${STE_REFINE}" == "1" ]]; then
    blockwise_cmd+=(--ste_refine_steps "${STE_REFINE_STEPS}")
    blockwise_cmd+=(--ste_refine_weight_decay "${STE_REFINE_WEIGHT_DECAY}")
    blockwise_cmd+=(--ste_refine_log_interval "${STE_REFINE_LOG_INTERVAL}")
    blockwise_cmd+=(--ste_refine_binary_lr "${STE_REFINE_BINARY_LR}")
    blockwise_cmd+=(--ste_refine_continuous_lr "${STE_REFINE_CONTINUOUS_LR}")
    if [[ -n "${STE_REFINE_ROW_GROUP_BATCH_SIZE}" ]]; then
      blockwise_cmd+=(--ste_refine_row_group_batch_size "${STE_REFINE_ROW_GROUP_BATCH_SIZE}")
    fi
  else
    blockwise_cmd+=(--ste_refine_steps 0)
  fi
  if [[ "${BLOCKWISE_FIX_THETA}" == "1" ]]; then
    blockwise_cmd+=(--fix_theta)
  fi
  if [[ "${BLOCKWISE_FIX_BETA}" == "1" ]]; then
    blockwise_cmd+=(--fix_beta)
  fi
  if [[ "${NO_IO_CACHE}" == "1" ]]; then
    blockwise_cmd+=(--no_io_cache)
  fi
  if [[ "${NO_SCALE_REFINE}" == "1" ]]; then
    blockwise_cmd+=(--no_scale_refine)
  fi
  if [[ "${PROGRESSIVE}" == "1" ]]; then
    blockwise_cmd+=(--progressive --progressive_mode "${PROGRESSIVE_MODE}")
  fi
  blockwise_cmd+=("$@")
  "${blockwise_cmd[@]}"
}

detect_gpu_ids() {
  if [[ -n "${GPU_IDS:-}" ]]; then
    IFS=',' read -r -a gpu_ids <<< "${GPU_IDS}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
  else
    mapfile -t gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader)
  fi
  if [[ "${#gpu_ids[@]}" -eq 0 ]]; then
    echo "No GPUs available for blockwise parallel execution." >&2
    exit 1
  fi
}

cleanup_jobs() {
  local pids
  pids="$(jobs -pr || true)"
  if [[ -n "${pids}" ]]; then
    kill ${pids} 2>/dev/null || true
  fi
}

if [[ "${BLOCK_IDX}" == "all" ]]; then
  NUM_BLOCKS="$(python "${LM_DIR}/layerwise_quant.py" --model_name "${MODEL_NAME}" --list_blocks)"
  detect_gpu_ids
  if (( BLOCKS_PER_GPU < 1 )); then
    echo "BLOCKS_PER_GPU must be >= 1" >&2
    exit 1
  fi
  total_slots=$(( ${#gpu_ids[@]} * BLOCKS_PER_GPU ))
  echo "Parallel blockwise quantization: ${NUM_BLOCKS} blocks over ${#gpu_ids[@]} GPU(s), ${BLOCKS_PER_GPU} block job(s) per GPU, total concurrency ${total_slots}: ${gpu_ids[*]}"

  trap cleanup_jobs EXIT
  active_jobs=0
  for ((block_idx=0; block_idx<NUM_BLOCKS; block_idx++)); do
    slot_idx=$(( block_idx % total_slots ))
    gpu_id="${gpu_ids[$((slot_idx % ${#gpu_ids[@]}))]}"
    echo "[launch] block ${block_idx} -> GPU ${gpu_id} (slot $((slot_idx + 1))/${total_slots})"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      export RUNTIME_DEVICE="cuda:0"
      run_one_block "${block_idx}" 0 "$@"
    ) &
    active_jobs=$((active_jobs + 1))
    if (( active_jobs >= total_slots )); then
      wait -n
      active_jobs=$((active_jobs - 1))
    fi
  done
  while (( active_jobs > 0 )); do
    wait -n
    active_jobs=$((active_jobs - 1))
  done
  trap - EXIT

  python "${LM_DIR}/src/build_bqq_model.py" assemble \
    --model_name "${MODEL_NAME}" \
    --block_dir "${BLOCKWISE_SAVE_DIR}" \
    --bit_width "${BIT_WIDTH}" \
    --group_size "${GROUP_SIZE}" \
    --output_dir "${ASSEMBLED_OUTPUT_DIR}"
else
  run_one_block "${BLOCK_IDX}" 1 "$@"
fi
