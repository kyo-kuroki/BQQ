#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l gpu_1=1
#$ -l h_rt=1:00:00

# Batch-mode 3-bit quantization array job (new code).
# Uses run_bqq_compile_batched — single GPU, no multiprocessing.
# Each target completes in a few minutes.
#
# Required env vars (passed via qsub -v):
#   TARGETS_LIST_FILE  - one target_name per line
#   CACHE_DIR          - directory with cached weight tensors
#   SAVE_DIR           - directory for quantized output
#   SIF_PATH           - path to the Apptainer .sif image
#   LM_SCRIPT_DIR      - absolute path to neural_network_compression/lm/
#   GROUP_SIZE, NUM_STEPS (optional, have defaults)

set -euo pipefail

echo "========================================"
echo "BQQ 3-bit Batch Array Job"
echo "SGE_TASK_ID : ${SGE_TASK_ID}"
echo "Host        : $(hostname)"
echo "Date        : $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "========================================"

GROUP_SIZE="${GROUP_SIZE:-32}"
NUM_STEPS="${NUM_STEPS:-10000}"

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

echo "Mode   : quantize-target (batch, 3-bit from scratch)"
"${APPTAINER_CMD[@]}" \
    python weight_aware_quant_cached.py quantize-target \
        --cache_dir       "${CACHE_DIR}" \
        --save_dir        "${SAVE_DIR}" \
        --target_name     "${TARGET_NAME}" \
        --bit_width       3 \
        --group_size      "${GROUP_SIZE}" \
        --num_steps       "${NUM_STEPS}" \
        --main_gpu_id     0 \
        --overwrite

echo ""
echo "Done: ${TARGET_NAME}  $(date --iso-8601=seconds)"
