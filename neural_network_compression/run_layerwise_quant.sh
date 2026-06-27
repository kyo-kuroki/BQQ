#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM_DIR="${SCRIPT_DIR}/lm"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-hf}" # Example Options: "Qwen/Qwen3.5-4B", "meta-llama/Llama-3.1-8B"
BLOCK_IDX="${BLOCK_IDX:-all}" # Options: "all", or a specific transformer block index.
LAYERS_PER_GPU="${LAYERS_PER_GPU:-8}" # Parallel layer quantization workers inside each block job.
BIT_WIDTH="${BIT_WIDTH:-1}"
GROUP_SIZE="${GROUP_SIZE:-64}"

LAYERWISE_ANNEAL_STEPS="${LAYERWISE_ANNEAL_STEPS:-50000}"
LAYERWISE_STE_STEPS="${LAYERWISE_STE_STEPS:-0}"
LAYERWISE_STE_LR="${LAYERWISE_STE_LR:-1e-3}"
LAYERWISE_STE_WEIGHT_DECAY="${LAYERWISE_STE_WEIGHT_DECAY:-0.0}"
LAYERWISE_STE_BINARY_LR="${LAYERWISE_STE_BINARY_LR:-1e-3}"
LAYERWISE_STE_CONTINUOUS_LR="${LAYERWISE_STE_CONTINUOUS_LR:-1e-4}"
LAYERWISE_STE_LOG_INTERVAL="${LAYERWISE_STE_LOG_INTERVAL:-20}"
LAYERWISE_ROW_GROUP_BATCH_SIZE="${LAYERWISE_ROW_GROUP_BATCH_SIZE:-}"
LAYERWISE_FIX_THETA="${LAYERWISE_FIX_THETA:-0}"
LAYERWISE_FIX_BETA="${LAYERWISE_FIX_BETA:-0}"

DATASET="${DATASET:-slimpajama}"
NSAMPLES="${NSAMPLES:-1024}"
SEQLEN="${SEQLEN:-2048}"
SEED="${SEED:-0}"
USE_MULTIBQQ="${USE_MULTIBQQ:-1}"
NO_SCALE_REFINE="${NO_SCALE_REFINE:-0}"
COMPENSATION_MODE="${COMPENSATION_MODE:-ldlq}"

MODEL_BASENAME="${MODEL_NAME##*/}"
LAYERWISE_DIR="${LAYERWISE_DIR:-${LM_DIR}/src/bqq_compressed_data/${MODEL_BASENAME}-${BIT_WIDTH}bit-${GROUP_SIZE}gs-${LAYERWISE_ANNEAL_STEPS}step}"
ASSEMBLED_OUTPUT_DIR="${ASSEMBLED_OUTPUT_DIR:-${LM_DIR}/src/quantized_models/${MODEL_BASENAME}}"

run_one_block() {
  local block_idx="$1"
  local should_assemble="$2"
  shift 2

  local assemble_args=(--no_assemble_full_model)
  if [[ "${should_assemble}" == "1" ]]; then
    assemble_args=(--assembled_output_dir "${ASSEMBLED_OUTPUT_DIR}")
  fi

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
    --workers_per_gpu "${LAYERS_PER_GPU}"
    --dataset "${DATASET}"
    --nsamples "${NSAMPLES}"
    --seqlen "${SEQLEN}"
    --seed "${SEED}"
    --main_gpu_id 0
    --save_dir "${LAYERWISE_DIR}"
    --compensation_mode "${COMPENSATION_MODE}"
    "${assemble_args[@]}"
  )

  if [[ "${USE_MULTIBQQ}" == "1" ]]; then
    layerwise_cmd+=(--use_multibqq)
  else
    layerwise_cmd+=(--no_use_multibqq)
  fi
  if [[ "${NO_SCALE_REFINE}" != "1" ]]; then
    layerwise_cmd+=(--scale_refine)
  fi
  if [[ "${LAYERWISE_FIX_THETA}" == "1" ]]; then
    layerwise_cmd+=(--fix_theta)
  fi
  if [[ "${LAYERWISE_FIX_BETA}" == "1" ]]; then
    layerwise_cmd+=(--fix_beta)
  fi
  if [[ -n "${LAYERWISE_ROW_GROUP_BATCH_SIZE}" ]]; then
    layerwise_cmd+=(--row_group_batch_size "${LAYERWISE_ROW_GROUP_BATCH_SIZE}")
  fi

  layerwise_cmd+=("$@")
  "${layerwise_cmd[@]}"
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
    echo "No GPUs available for layerwise parallel execution." >&2
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

assemble_model() {
  python "${LM_DIR}/src/build_bqq_model.py" build \
    --model_name "${MODEL_NAME}" \
    --bit_widths "${BIT_WIDTH}" \
    --group_size "${GROUP_SIZE}" \
    --num_steps "${LAYERWISE_ANNEAL_STEPS}" \
    --compressed_data_dir "${LAYERWISE_DIR}" \
    --output_dir "${ASSEMBLED_OUTPUT_DIR}" \
    --device cpu
}

if [[ "${BLOCK_IDX}" == "all" ]]; then
  NUM_BLOCKS="$(python "${LM_DIR}/layerwise_quant.py" --model_name "${MODEL_NAME}" --list_blocks)"
  detect_gpu_ids
  echo "Parallel layerwise quantization: ${NUM_BLOCKS} blocks over ${#gpu_ids[@]} GPU(s), ${LAYERS_PER_GPU} layer worker(s) inside each block job: ${gpu_ids[*]}"

  trap cleanup_jobs EXIT
  active_jobs=0
  total_slots="${#gpu_ids[@]}"
  for ((block_idx=0; block_idx<NUM_BLOCKS; block_idx++)); do
    gpu_id="${gpu_ids[$((block_idx % total_slots))]}"
    echo "[launch] block ${block_idx} -> GPU ${gpu_id} (${LAYERS_PER_GPU} layer worker(s))"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
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

  assemble_model
else
  run_one_block "${BLOCK_IDX}" 1 "$@"
fi
