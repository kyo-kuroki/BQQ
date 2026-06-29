#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CV_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/cv"

MODEL_NAME="${MODEL_NAME:-deit-s}" # Options: deit-s, deit-b, vit-s, vit-b, swin-t, swin-s
BLOCK_IDX="${BLOCK_IDX:-all}" # Options: "all", or a specific transformer block index.
LAYERS_PER_GPU="${LAYERS_PER_GPU:-8}" # Parallel layer quantization workers inside each block job.
NUM_BLOCKS_PER_GPU="${NUM_BLOCKS_PER_GPU:-4}" # Consecutive blocks handled by one process on one GPU (model loaded once per GPU).
BIT_WIDTH="${BIT_WIDTH:-2}"
GROUP_SIZE="${GROUP_SIZE:-32}"

LAYERWISE_ANNEAL_STEPS="${LAYERWISE_ANNEAL_STEPS:-20000}"
LAYERWISE_STE_STEPS="${LAYERWISE_STE_STEPS:-0}"
LAYERWISE_STE_LR="${LAYERWISE_STE_LR:-1e-3}"
LAYERWISE_STE_WEIGHT_DECAY="${LAYERWISE_STE_WEIGHT_DECAY:-0.0}"
LAYERWISE_STE_BINARY_LR="${LAYERWISE_STE_BINARY_LR:-1e-3}"
LAYERWISE_STE_CONTINUOUS_LR="${LAYERWISE_STE_CONTINUOUS_LR:-1e-4}"
LAYERWISE_STE_LOG_INTERVAL="${LAYERWISE_STE_LOG_INTERVAL:-20}"
LAYERWISE_ROW_GROUP_BATCH_SIZE="${LAYERWISE_ROW_GROUP_BATCH_SIZE:-}"
LAYERWISE_FIX_THETA="${LAYERWISE_FIX_THETA:-0}"
LAYERWISE_FIX_BETA="${LAYERWISE_FIX_BETA:-0}"
LAYERWISE_SAVE_RECONSTRUCTED="${LAYERWISE_SAVE_RECONSTRUCTED:-0}" # 1 = also save the dense reconstructed weight per layer (debug).

NSAMPLES="${NSAMPLES:-256}" # Number of ImageNet calibration images.
DATA_PATH="${DATA_PATH:-}"  # Path to ImageNet. Empty => fall back to IMAGENET_DIR env var.
SEED="${SEED:-0}"
USE_MULTIBQQ="${USE_MULTIBQQ:-0}"
NO_SCALE_REFINE="${NO_SCALE_REFINE:-0}"
COMPENSATION_MODE="${COMPENSATION_MODE:-ldlq}"

LAYERWISE_DIR="${LAYERWISE_DIR:-${CV_DIR}/src/bqq_compressed_data/${MODEL_NAME}-${BIT_WIDTH}bit-${GROUP_SIZE}gs-${LAYERWISE_ANNEAL_STEPS}step}"
ASSEMBLED_OUTPUT_DIR="${ASSEMBLED_OUTPUT_DIR:-${CV_DIR}/src/quantized_models/${MODEL_NAME}}"

data_args=(--nsamples "${NSAMPLES}")
if [[ -n "${DATA_PATH}" ]]; then
  data_args+=(--data_path "${DATA_PATH}")
fi

run_one_block() {
  local block_idx="$1"
  local should_assemble="$2"
  shift 2

  local assemble_args=(--no_assemble_full_model)
  if [[ "${should_assemble}" == "1" ]]; then
    assemble_args=(--assembled_output_dir "${ASSEMBLED_OUTPUT_DIR}")
  fi

  local layerwise_cmd=(
    python "${CV_DIR}/layerwise_quant.py"
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
    "${data_args[@]}"
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
  if [[ "${LAYERWISE_SAVE_RECONSTRUCTED}" == "1" ]]; then
    layerwise_cmd+=(--save_reconstructed)
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
  python "${CV_DIR}/src/build_bqq_model.py" build \
    --model_name "${MODEL_NAME}" \
    --bit_widths "${BIT_WIDTH}" \
    --group_size "${GROUP_SIZE}" \
    --num_steps "${LAYERWISE_ANNEAL_STEPS}" \
    --compressed_data_dir "${LAYERWISE_DIR}" \
    --output_dir "${ASSEMBLED_OUTPUT_DIR}" \
    --device cpu
}

if [[ "${BLOCK_IDX}" == "all" ]]; then
  NUM_BLOCKS="$(python "${CV_DIR}/layerwise_quant.py" --model_name "${MODEL_NAME}" --list_blocks)"
  detect_gpu_ids
  num_gpus="${#gpu_ids[@]}"
  if (( NUM_BLOCKS_PER_GPU < 1 )); then
    echo "NUM_BLOCKS_PER_GPU must be >= 1, got ${NUM_BLOCKS_PER_GPU}." >&2
    exit 1
  fi

  # Group blocks into contiguous chunks of NUM_BLOCKS_PER_GPU. Each chunk is
  # handled by ONE process pinned to a single GPU, so the model is loaded only
  # once per GPU and the Hessian cache for all blocks in the chunk is generated
  # in a single forward pass. The chunk's layers are quantized in parallel by
  # LAYERS_PER_GPU workers inside that process.
  chunks=()
  for (( start=0; start<NUM_BLOCKS; start+=NUM_BLOCKS_PER_GPU )); do
    end=$(( start + NUM_BLOCKS_PER_GPU - 1 ))
    if (( end >= NUM_BLOCKS )); then
      end=$(( NUM_BLOCKS - 1 ))
    fi
    spec="${start}"
    for (( b=start+1; b<=end; b++ )); do
      spec+=",${b}"
    done
    chunks+=("${spec}")
  done

  echo "Parallel layerwise quantization: ${NUM_BLOCKS} blocks in ${#chunks[@]} chunk(s) of up to ${NUM_BLOCKS_PER_GPU} block(s), one model load per GPU, over ${num_gpus} GPU(s), ${LAYERS_PER_GPU} layer worker(s) per chunk: ${gpu_ids[*]}"

  # At most one chunk process (one model) runs on a GPU at a time.
  declare -A gpu_busy
  declare -A pid_gpu
  for g in "${gpu_ids[@]}"; do
    gpu_busy["${g}"]=0
  done

  pick_gpu() {
    local g
    for g in "${gpu_ids[@]}"; do
      if (( gpu_busy["${g}"] == 0 )); then
        echo "${g}"
        return 0
      fi
    done
    return 1
  }

  reap_one() {
    local finished_pid="" status=0
    wait -n -p finished_pid || status=$?
    if [[ -n "${finished_pid}" && -n "${pid_gpu[${finished_pid}]:-}" ]]; then
      gpu_busy["${pid_gpu[${finished_pid}]}"]=0
      unset "pid_gpu[${finished_pid}]"
    fi
    if (( status != 0 )); then
      echo "Chunk process (pid ${finished_pid:-unknown}) failed with status ${status}; aborting." >&2
      exit "${status}"
    fi
  }

  trap cleanup_jobs EXIT
  running=0
  for spec in "${chunks[@]}"; do
    while (( running >= num_gpus )); do
      reap_one
      running=$(( running - 1 ))
    done
    gpu_id="$(pick_gpu)"
    echo "[launch] blocks ${spec} -> GPU ${gpu_id} (1 model load, ${LAYERS_PER_GPU} layer worker(s))"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      run_one_block "${spec}" 0 "$@"
    ) &
    pid=$!
    pid_gpu["${pid}"]="${gpu_id}"
    gpu_busy["${gpu_id}"]=1
    running=$(( running + 1 ))
  done
  while (( running > 0 )); do
    reap_one
    running=$(( running - 1 ))
  done
  trap - EXIT

  assemble_model
else
  run_one_block "${BLOCK_IDX}" 1 "$@"
fi
