#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=8:00:00
#$ -j y

# Combined array job: produces 3-bit quantization for each target.
#   - If 2-bit patches already exist in SAVE_DIR → extend by 1 bit
#   - Otherwise → quantize from scratch at 3 bits
#
# Required env vars (passed via qsub -v):
#   TARGETS_LIST_FILE  - one target_name per line
#   CACHE_DIR          - directory with cached weight tensors
#   SAVE_DIR           - directory for quantized patches (both source and output)
#   SIF_PATH           - path to the Apptainer .sif image
#   LM_SCRIPT_DIR      - absolute path to nn_compression/lm/
#   GROUP_SIZE, NUM_STEPS, RANK_SCALE, SEED, WORKERS_PER_GPU (optional, have defaults)

set -euo pipefail

echo "========================================"
echo "BQQ 3-bit Combined Array Job"
echo "SGE_TASK_ID : ${SGE_TASK_ID}"
echo "Host        : $(hostname)"
echo "Date        : $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "========================================"

GROUP_SIZE="${GROUP_SIZE:-32}"
NUM_STEPS="${NUM_STEPS:-10000}"
RANK_SCALE="${RANK_SCALE:-1.0}"
SEED="${SEED:-0}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-384}"

if [[ ! -f "${TARGETS_LIST_FILE}" ]]; then
    echo "ERROR: TARGETS_LIST_FILE not found: ${TARGETS_LIST_FILE}" >&2
    exit 1
fi

TARGET_NAME="$(sed -n "${SGE_TASK_ID}p" "${TARGETS_LIST_FILE}")"
if [[ -z "${TARGET_NAME}" ]]; then
    echo "ERROR: no entry at line ${SGE_TASK_ID} in ${TARGETS_LIST_FILE}" >&2
    exit 1
fi

echo "Target : ${TARGET_NAME}"

BQQ_ROOT="$(dirname "$(dirname "${LM_SCRIPT_DIR}")")"
HF_HOME="${HF_HOME:-/gs/bs/tga-artic/k-kuroki/hf_cache}"

APPTAINER_CMD=(
    apptainer exec --nv
    --bind "${HOME}:${HOME}"
    --bind "${BQQ_ROOT}:${BQQ_ROOT}"
    --bind "${HF_HOME}:${HF_HOME}"
    --env "HF_HOME=${HF_HOME}"
    --pwd "${LM_SCRIPT_DIR}"
    "${SIF_PATH}"
)

# 2-bit の consolidated ファイルが存在するか確認
SOURCE_CONSOLIDATED="${SAVE_DIR}/_consolidated/${TARGET_NAME}.pth"

if [[ -f "${SOURCE_CONSOLIDATED}" ]]; then
    echo "Mode   : extend (2-bit consolidated found, adding 1 bit)"
    "${APPTAINER_CMD[@]}" \
        python weight_aware_quant_cached.py extend-target \
            --cache_dir       "${CACHE_DIR}" \
            --source_dir      "${SAVE_DIR}" \
            --save_dir        "${SAVE_DIR}" \
            --target_name     "${TARGET_NAME}" \
            --extra_bits      1 \
            --group_size      "${GROUP_SIZE}" \
            --num_steps       "${NUM_STEPS}" \
            --rank_scale      "${RANK_SCALE}" \
            --seed            "${SEED}" \
            --workers_per_gpu "${WORKERS_PER_GPU}" \
            --main_gpu_id     0 \
            --overwrite
else
    echo "Mode   : quantize (no consolidated found, running 3-bit from scratch)"
    "${APPTAINER_CMD[@]}" \
        python weight_aware_quant_cached.py quantize-target \
            --cache_dir       "${CACHE_DIR}" \
            --save_dir        "${SAVE_DIR}" \
            --target_name     "${TARGET_NAME}" \
            --bit_width       3 \
            --group_size      "${GROUP_SIZE}" \
            --num_steps       "${NUM_STEPS}" \
            --rank_scale      "${RANK_SCALE}" \
            --seed            "${SEED}" \
            --workers_per_gpu "${WORKERS_PER_GPU}" \
            --main_gpu_id     0 \
            --overwrite
fi

echo ""
echo "Done: ${TARGET_NAME}  $(date --iso-8601=seconds)"
