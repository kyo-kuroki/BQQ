#!/usr/bin/env bash
set -euo pipefail

# Block-wise BQQ quantization for vision transformers.
#   For each transformer block (in parallel across GPUs): quantize every Linear
#   in the block with Hessian-aware BQQ, then fine-tune the block parameters to
#   match the FP block output (MSE). Finally assemble the full model from the
#   per-block files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CV_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/cv"

MODEL_NAME="${MODEL_NAME:-deit-s}" # Options: deit-s, deit-b, vit-s, vit-b, swin-t, swin-s
BLOCK_IDX="${BLOCK_IDX:-all}" # Options: "all", or a specific block index.
BLOCKS_PER_GPU="${BLOCKS_PER_GPU:-1}"
BIT_WIDTH="${BIT_WIDTH:-2}"
GROUP_SIZE="${GROUP_SIZE:-32}"

LAYERWISE_ANNEAL_STEPS="${LAYERWISE_ANNEAL_STEPS:-20000}"
RANK_SCALE="${RANK_SCALE:-1.0}"

BLOCKWISE_EPOCHS="${BLOCKWISE_EPOCHS:-10}"
BLOCKWISE_LR="${BLOCKWISE_LR:-1e-5}"
BLOCKWISE_MAX_GRAD_NORM="${BLOCKWISE_MAX_GRAD_NORM:-1.0}"

NSAMPLES="${NSAMPLES:-256}" # Number of ImageNet calibration images.
DATA_PATH="${DATA_PATH:-}"  # Path to ImageNet. Empty => fall back to IMAGENET_DIR env var.
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
USE_MULTIBQQ="${USE_MULTIBQQ:-1}"
NO_SCALE_REFINE="${NO_SCALE_REFINE:-0}"
NO_HESSIAN="${NO_HESSIAN:-0}"
COMPENSATION_MODE="${COMPENSATION_MODE:-ldlq}"
HESSIAN_CACHE_DIR="${HESSIAN_CACHE_DIR:-}"

BLOCKWISE_SAVE_DIR="${BLOCKWISE_SAVE_DIR:-${CV_DIR}/blockwise_output/${MODEL_NAME}}"
ASSEMBLED_OUTPUT_DIR="${ASSEMBLED_OUTPUT_DIR:-${CV_DIR}/src/quantized_models/${MODEL_NAME}}"

data_args=(--nsamples "${NSAMPLES}")
if [[ -n "${DATA_PATH}" ]]; then
  data_args+=(--data_path "${DATA_PATH}")
fi

run_one_block() {
  local block_idx="$1"
  shift 1

  local runtime_device="${RUNTIME_DEVICE:-${DEVICE}}"
  local cmd=(
    python "${CV_DIR}/blockwise_quant.py"
    --model_name "${MODEL_NAME}"
    --block_idx "${block_idx}"
    --bit_width "${BIT_WIDTH}"
    --group_size "${GROUP_SIZE}"
    --num_steps "${LAYERWISE_ANNEAL_STEPS}"
    --rank_scale "${RANK_SCALE}"
    "${data_args[@]}"
    --epochs "${BLOCKWISE_EPOCHS}"
    --lr "${BLOCKWISE_LR}"
    --max_grad_norm "${BLOCKWISE_MAX_GRAD_NORM}"
    --seed "${SEED}"
    --device "${runtime_device}"
    --compensation_mode "${COMPENSATION_MODE}"
    --save_dir "${BLOCKWISE_SAVE_DIR}"
  )
  if [[ "${USE_MULTIBQQ}" != "1" ]]; then
    cmd+=(--no_use_multibqq)
  fi
  if [[ "${NO_SCALE_REFINE}" == "1" ]]; then
    cmd+=(--no_scale_refine)
  fi
  if [[ "${NO_HESSIAN}" == "1" ]]; then
    cmd+=(--no_hessian)
  fi
  if [[ -n "${HESSIAN_CACHE_DIR}" ]]; then
    cmd+=(--hessian_cache_dir "${HESSIAN_CACHE_DIR}")
  fi
  cmd+=("$@")
  "${cmd[@]}"
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

assemble_model() {
  python "${CV_DIR}/src/build_bqq_model.py" assemble \
    --model_name "${MODEL_NAME}" \
    --block_dir "${BLOCKWISE_SAVE_DIR}" \
    --output_dir "${ASSEMBLED_OUTPUT_DIR}"
}

if [[ "${BLOCK_IDX}" == "all" ]]; then
  NUM_BLOCKS="$(python "${CV_DIR}/layerwise_quant.py" --model_name "${MODEL_NAME}" --list_blocks)"
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
      run_one_block "${block_idx}" "$@"
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
  run_one_block "${BLOCK_IDX}" "$@"
fi
