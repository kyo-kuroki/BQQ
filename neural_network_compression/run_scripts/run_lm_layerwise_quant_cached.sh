#!/usr/bin/env bash
set -euo pipefail

# Cache-first layerwise quantization.
#   Phase 1: load the model ONCE and cache every target's Hessian (+ FP weight)
#            into a cache directory (single forward pass).
#   Phase 2: quantize each target in its own process, distributed across GPUs by
#            this script (one process per target, several per GPU) -- no mp.spawn.
#            Each process only reads its cached Hessian, so no model/data reload.
#   Phase 3: assemble the full quantized model from the consolidated patches.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/lm"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-hf}" # Example: "Qwen/Qwen3.5-4B", "meta-llama/Llama-3.1-8B"
LAYER_THRESHOLD="${LAYER_THRESHOLD:-0}"
TARGETS_PER_GPU="${TARGETS_PER_GPU:-4}" # Concurrent single-target quantize processes per GPU.
BIT_WIDTH="${BIT_WIDTH:-2}"
GROUP_SIZE="${GROUP_SIZE:-128}"

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
LAYERWISE_SAVE_RECONSTRUCTED="${LAYERWISE_SAVE_RECONSTRUCTED:-0}" # 1 = also save the dense reconstructed weight per layer (debug).

DATASET="${DATASET:-slimpajama}"
NSAMPLES="${NSAMPLES:-1024}"
SEQLEN="${SEQLEN:-2048}"
SEED="${SEED:-0}"
USE_MULTIBQQ="${USE_MULTIBQQ:-0}"
NO_SCALE_REFINE="${NO_SCALE_REFINE:-0}"
COMPENSATION_MODE="${COMPENSATION_MODE:-ldlq}"
REFRESH_HESSIAN_CACHE="${REFRESH_HESSIAN_CACHE:-0}" # 1 = recompute Hessians even if cached.

MODEL_BASENAME="${MODEL_NAME##*/}"
LAYERWISE_DIR="${LAYERWISE_DIR:-${LM_DIR}/src/bqq_compressed_data/${MODEL_BASENAME}-${BIT_WIDTH}bit-${GROUP_SIZE}gs-${LAYERWISE_ANNEAL_STEPS}step}"
ASSEMBLED_OUTPUT_DIR="${ASSEMBLED_OUTPUT_DIR:-${LM_DIR}/src/quantized_models/${MODEL_BASENAME}}"
HESSIAN_CACHE_DIR="${HESSIAN_CACHE_DIR:-${LM_DIR}/src/hessian_cache/${MODEL_BASENAME}/${DATASET}_${SEQLEN}seqlen_${NSAMPLES}samples_thr${LAYER_THRESHOLD}}"

# Build the per-target quantize command (shared by all parallel jobs).
run_one_target() {
  local target_idx="$1"

  local cmd=(
    python "${LM_DIR}/layerwise_quant.py"
    --model_name "${MODEL_NAME}"
    --target_idx "${target_idx}"
    --hessian_cache_dir "${HESSIAN_CACHE_DIR}"
    --save_dir "${LAYERWISE_DIR}"
    --bit_width "${BIT_WIDTH}"
    --group_size "${GROUP_SIZE}"
    --num_steps "${LAYERWISE_ANNEAL_STEPS}"
    --ste_refine_steps "${LAYERWISE_STE_STEPS}"
    --ste_refine_lr "${LAYERWISE_STE_LR}"
    --ste_refine_binary_lr "${LAYERWISE_STE_BINARY_LR}"
    --ste_refine_continuous_lr "${LAYERWISE_STE_CONTINUOUS_LR}"
    --ste_refine_weight_decay "${LAYERWISE_STE_WEIGHT_DECAY}"
    --ste_refine_log_interval "${LAYERWISE_STE_LOG_INTERVAL}"
    --seed "${SEED}"
    --main_gpu_id 0
    --compensation_mode "${COMPENSATION_MODE}"
    --no_assemble_full_model
  )

  if [[ "${USE_MULTIBQQ}" == "1" ]]; then
    cmd+=(--use_multibqq)
  else
    cmd+=(--no_use_multibqq)
  fi
  if [[ "${NO_SCALE_REFINE}" != "1" ]]; then
    cmd+=(--scale_refine)
  fi
  if [[ "${LAYERWISE_FIX_THETA}" == "1" ]]; then
    cmd+=(--fix_theta)
  fi
  if [[ "${LAYERWISE_FIX_BETA}" == "1" ]]; then
    cmd+=(--fix_beta)
  fi
  if [[ "${LAYERWISE_SAVE_RECONSTRUCTED}" == "1" ]]; then
    cmd+=(--save_reconstructed)
  fi
  if [[ -n "${LAYERWISE_ROW_GROUP_BATCH_SIZE}" ]]; then
    cmd+=(--row_group_batch_size "${LAYERWISE_ROW_GROUP_BATCH_SIZE}")
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
    echo "No GPUs available for parallel execution." >&2
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

detect_gpu_ids
num_gpus="${#gpu_ids[@]}"
if (( TARGETS_PER_GPU < 1 )); then
  echo "TARGETS_PER_GPU must be >= 1, got ${TARGETS_PER_GPU}." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Phase 1: cache every target's Hessian (+ FP weight) in a single forward pass.
# ---------------------------------------------------------------------------
echo "[phase 1] caching Hessians -> ${HESSIAN_CACHE_DIR} (one model load on GPU ${gpu_ids[0]})"
cache_cmd=(
  python "${LM_DIR}/layerwise_quant.py"
  --model_name "${MODEL_NAME}"
  --cache_hessians
  --hessian_cache_dir "${HESSIAN_CACHE_DIR}"
  --layer_threshold "${LAYER_THRESHOLD}"
  --dataset "${DATASET}"
  --nsamples "${NSAMPLES}"
  --seqlen "${SEQLEN}"
  --seed "${SEED}"
  --main_gpu_id 0
)
if [[ "${REFRESH_HESSIAN_CACHE}" == "1" ]]; then
  cache_cmd+=(--refresh_hessian_cache)
fi
CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "${cache_cmd[@]}"

# ---------------------------------------------------------------------------
# Phase 2: quantize each target in its own process, spread across GPUs.
# ---------------------------------------------------------------------------
NUM_TARGETS="$(python "${LM_DIR}/layerwise_quant.py" --model_name "${MODEL_NAME}" --list_targets --hessian_cache_dir "${HESSIAN_CACHE_DIR}")"
total_slots=$(( num_gpus * TARGETS_PER_GPU ))
echo "[phase 2] quantizing ${NUM_TARGETS} targets over ${num_gpus} GPU(s) x ${TARGETS_PER_GPU} job(s)/GPU = ${total_slots} concurrent job(s): ${gpu_ids[*]}"

# Track how many jobs run on each GPU and which GPU each PID uses, so a GPU never
# exceeds TARGETS_PER_GPU concurrent jobs.
declare -A gpu_active
declare -A pid_gpu
for g in "${gpu_ids[@]}"; do
  gpu_active["${g}"]=0
done

pick_gpu() {
  local g
  for g in "${gpu_ids[@]}"; do
    if (( gpu_active["${g}"] < TARGETS_PER_GPU )); then
      echo "${g}"
      return 0
    fi
  done
  return 1
}

reap_one() {
  # Wait for any one job to finish and free its GPU slot. Fail fast on error.
  local finished_pid="" status=0
  wait -n -p finished_pid || status=$?
  if [[ -n "${finished_pid}" && -n "${pid_gpu[${finished_pid}]:-}" ]]; then
    local g="${pid_gpu[${finished_pid}]}"
    gpu_active["${g}"]=$(( gpu_active["${g}"] - 1 ))
    unset "pid_gpu[${finished_pid}]"
  fi
  if (( status != 0 )); then
    echo "Target job (pid ${finished_pid:-unknown}) failed with status ${status}; aborting." >&2
    exit "${status}"
  fi
}

trap cleanup_jobs EXIT
running=0
for (( t=0; t<NUM_TARGETS; t++ )); do
  while (( running >= total_slots )); do
    reap_one
    running=$(( running - 1 ))
  done
  gpu_id="$(pick_gpu)"
  echo "[launch] target ${t}/${NUM_TARGETS} -> GPU ${gpu_id} ($(( gpu_active["${gpu_id}"] + 1 ))/${TARGETS_PER_GPU} on this GPU)"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    run_one_target "${t}" "$@"
  ) &
  pid=$!
  pid_gpu["${pid}"]="${gpu_id}"
  gpu_active["${gpu_id}"]=$(( gpu_active["${gpu_id}"] + 1 ))
  running=$(( running + 1 ))
done
while (( running > 0 )); do
  reap_one
  running=$(( running - 1 ))
done
trap - EXIT

# ---------------------------------------------------------------------------
# Phase 3: assemble the full quantized model.
# ---------------------------------------------------------------------------
echo "[phase 3] assembling full model -> ${ASSEMBLED_OUTPUT_DIR}"
assemble_model
