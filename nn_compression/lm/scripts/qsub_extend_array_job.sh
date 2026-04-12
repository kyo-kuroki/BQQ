#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=8:00:00
#$ -j y

# Array job: 1 task = 1 target weight tensor.
# Extends an existing N-bit quantization by optimising residual bits.
#
# Required env vars (passed via qsub -v):
#   TARGETS_LIST_FILE - one target_name per line
#   CACHE_DIR         - directory with cached weight tensors (from prepare-cache)
#   SOURCE_DIR        - directory containing the existing N-bit patch files
#   SAVE_DIR          - output directory for the extended quantization
#   SIF_PATH          - path to the Apptainer .sif image
#   LM_SCRIPT_DIR     - absolute path to nn_compression/lm/
#   EXTRA_BITS        - number of additional bits to optimise (default: 1)
#   GROUP_SIZE        - patch group size (default: 32)
#   NUM_STEPS         - BFGS optimisation steps (default: 10000)
#   RANK_SCALE        - rank scaling factor (default: 1.0)
#   SEED              - random seed (default: 0)
#   WORKERS_PER_GPU   - parallel workers per GPU (default: 384)

set -euo pipefail

echo "========================================"
echo "BQQ Extend Array Job"
echo "SGE_TASK_ID : ${SGE_TASK_ID}"
echo "Host        : $(hostname)"
echo "Date        : $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "========================================"

# ---- defaults ---------------------------------------------------------------
EXTRA_BITS="${EXTRA_BITS:-1}"
GROUP_SIZE="${GROUP_SIZE:-32}"
NUM_STEPS="${NUM_STEPS:-10000}"
RANK_SCALE="${RANK_SCALE:-1.0}"
SEED="${SEED:-0}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-384}"

# ---- read target name from list ---------------------------------------------
if [[ ! -f "${TARGETS_LIST_FILE}" ]]; then
    echo "ERROR: TARGETS_LIST_FILE not found: ${TARGETS_LIST_FILE}" >&2
    exit 1
fi

TARGET_NAME="$(sed -n "${SGE_TASK_ID}p" "${TARGETS_LIST_FILE}")"
if [[ -z "${TARGET_NAME}" ]]; then
    echo "ERROR: no entry at line ${SGE_TASK_ID} in ${TARGETS_LIST_FILE}" >&2
    exit 1
fi

echo "Target     : ${TARGET_NAME}"
echo "Extra bits : ${EXTRA_BITS}"
echo "Params     : gs=${GROUP_SIZE}  steps=${NUM_STEPS}"
echo ""

# ---- derive BQQ root --------------------------------------------------------
BQQ_ROOT="$(dirname "$(dirname "${LM_SCRIPT_DIR}")")"
HF_HOME="${HF_HOME:-/gs/bs/tga-artic/k-kuroki/hf_cache}"

# ---- run extension ----------------------------------------------------------
apptainer exec --nv \
    --bind "${HOME}:${HOME}" \
    --bind "${BQQ_ROOT}:${BQQ_ROOT}" \
    --bind "${HF_HOME}:${HF_HOME}" \
    --env  "HF_HOME=${HF_HOME}" \
    --pwd  "${LM_SCRIPT_DIR}" \
    "${SIF_PATH}" \
    python weight_aware_quant_cached.py extend-target \
        --cache_dir       "${CACHE_DIR}" \
        --source_dir      "${SOURCE_DIR}" \
        --save_dir        "${SAVE_DIR}" \
        --target_name     "${TARGET_NAME}" \
        --extra_bits      "${EXTRA_BITS}" \
        --group_size      "${GROUP_SIZE}" \
        --num_steps       "${NUM_STEPS}" \
        --rank_scale      "${RANK_SCALE}" \
        --seed            "${SEED}" \
        --workers_per_gpu "${WORKERS_PER_GPU}" \
        --main_gpu_id     0

echo ""
echo "Done: ${TARGET_NAME}  $(date --iso-8601=seconds)"
