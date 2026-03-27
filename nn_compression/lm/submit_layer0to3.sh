#!/usr/bin/env bash
# Submit BQQ quantization jobs for layers 0-3 of Qwen3.5-2B / 4B / 9B.
# These layers were unintentionally excluded by layer_threshold=4 in the original run.
#
# Usage:
#   bash submit_layer0to3.sh [--dry_run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF_PATH="/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif"
JOB_SCRIPT="${SCRIPT_DIR}/qsub_patch_array_job.sh"
BQQ_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"
HF_HOME="/gs/bs/tga-artic/k-kuroki/hf_cache"

BIT_WIDTH=2
GROUP_SIZE=32
NUM_STEPS=10000
RANK_SCALE=1.0
SEED=0
WORKERS_PER_GPU=1024
WALLTIME="8:00:00"
GPU_RESOURCE="gpu_1=1"
DRY_RUN=0

MODELS=(
    "Qwen/Qwen3.5-2B"
    "Qwen/Qwen3.5-4B"
    "Qwen/Qwen3.5-9B"
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry_run) DRY_RUN=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

run_in_container() {
    apptainer exec --nv \
        --bind "${HOME}:${HOME}" \
        --bind "${BQQ_ROOT}:${BQQ_ROOT}" \
        --bind "${HF_HOME}:${HF_HOME}" \
        --env  "HF_HOME=${HF_HOME}" \
        --env  "OPENBLAS_NUM_THREADS=1" \
        --pwd  "${SCRIPT_DIR}" \
        "${SIF_PATH}" \
        "$@"
}

chmod +x "${JOB_SCRIPT}"

declare -a SUBMITTED_JOBS=()

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_BASENAME="${MODEL_NAME##*/}"
    CACHE_DIR="${SCRIPT_DIR}/cache/${MODEL_BASENAME}-layer0to3"
    SAVE_DIR="${SCRIPT_DIR}/bqq_compressed_data/${MODEL_BASENAME}-${GROUP_SIZE}gs-${NUM_STEPS}step"
    JOB_DIR="${SCRIPT_DIR}/qsub_jobs/${MODEL_BASENAME}-bit${BIT_WIDTH}-gs${GROUP_SIZE}-layer0to3"
    TARGETS_LIST_FILE="${JOB_DIR}/targets.txt"
    LOG_DIR="${JOB_DIR}/logs"

    echo ""
    echo "============================================================"
    echo "  Model : ${MODEL_NAME}"
    echo "  Cache : ${CACHE_DIR}"
    echo "  Save  : ${SAVE_DIR}"
    echo "============================================================"

    mkdir -p "${JOB_DIR}" "${LOG_DIR}" "${SAVE_DIR}"

    echo "[${MODEL_BASENAME}] Step 1/3: prepare-cache (layers 0-3)"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "  (dry-run) would run: python weight_aware_quant_cached.py prepare-cache --layer_threshold 0 --layer_max 3 ..."
    else
        run_in_container python weight_aware_quant_cached.py prepare-cache \
            --model_name      "${MODEL_NAME}" \
            --layer_threshold 0 \
            --layer_max       3 \
            --cache_dir       "${CACHE_DIR}"
    fi

    echo "[${MODEL_BASENAME}] Step 2/3: list-targets -> ${TARGETS_LIST_FILE}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "  (dry-run) would write targets list to ${TARGETS_LIST_FILE}"
        N_JOBS=999
    else
        run_in_container python weight_aware_quant_cached.py list-targets \
            --cache_dir "${CACHE_DIR}" \
            > "${TARGETS_LIST_FILE}"

        N_JOBS="$(wc -l < "${TARGETS_LIST_FILE}")"
        echo "[${MODEL_BASENAME}] ${N_JOBS} targets found."

        if [[ "${N_JOBS}" -eq 0 ]]; then
            echo "[${MODEL_BASENAME}] WARNING: no targets — skipping." >&2
            continue
        fi
    fi

    echo "[${MODEL_BASENAME}] Step 3/3: submitting qsub array job (1-${N_JOBS})"

    QSUB_VARS="HF_HOME=${HF_HOME}"
    QSUB_VARS+=",TARGETS_LIST_FILE=${TARGETS_LIST_FILE}"
    QSUB_VARS+=",CACHE_DIR=${CACHE_DIR}"
    QSUB_VARS+=",SAVE_DIR=${SAVE_DIR}"
    QSUB_VARS+=",SIF_PATH=${SIF_PATH}"
    QSUB_VARS+=",LM_SCRIPT_DIR=${SCRIPT_DIR}"
    QSUB_VARS+=",BIT_WIDTH=${BIT_WIDTH}"
    QSUB_VARS+=",GROUP_SIZE=${GROUP_SIZE}"
    QSUB_VARS+=",NUM_STEPS=${NUM_STEPS}"
    QSUB_VARS+=",RANK_SCALE=${RANK_SCALE}"
    QSUB_VARS+=",SEED=${SEED}"
    QSUB_VARS+=",WORKERS_PER_GPU=${WORKERS_PER_GPU}"

    JOB_NAME="bqq_${MODEL_BASENAME}_l0to3"

    QSUB_CMD=(
        qsub
        -g tga-artic
        -S /bin/bash
        -cwd
        -j y
        -l "${GPU_RESOURCE}"
        -l "h_rt=${WALLTIME}"
        -t "1-${N_JOBS}"
        -tc 100
        -N "${JOB_NAME}"
        -o "${LOG_DIR}/"
        -v "${QSUB_VARS}"
        "${JOB_SCRIPT}"
    )

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "  (dry-run) qsub command:"
        printf '    %s\n' "${QSUB_CMD[@]}"
    else
        JOB_OUTPUT="$("${QSUB_CMD[@]}")"
        echo "  ${JOB_OUTPUT}"
        SUBMITTED_JOBS+=("${MODEL_BASENAME}: ${JOB_OUTPUT}")
    fi
done

echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "(dry-run mode — no jobs were submitted)"
else
    for entry in "${SUBMITTED_JOBS[@]}"; do
        echo "  ${entry}"
    done
    echo ""
    echo "Monitor with:"
    echo "  qstat -u $(whoami)"
fi
