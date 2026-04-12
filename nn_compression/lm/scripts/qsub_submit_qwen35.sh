#!/usr/bin/env bash
# Submit BQQ weight-aware quantization jobs for Qwen3.5-2B / 4B / 9B.
#
# Usage:
#   bash qsub_submit_qwen35.sh [options]
#
# Options:
#   --bit_width   N      Quantization bits            (default: 2)
#   --group_size  N      Patch group size              (default: 32)
#   --num_steps   N      BFGS optimisation steps       (default: 10000)
#   --walltime    HH:MM:SS  Per-job GPU walltime       (default: 4:00:00)
#   --gpu_resource STR   SGE GPU resource request      (default: gpu_1=1)
#   --workers_per_gpu N  Threads per GPU inside job    (default: 16)
#   --sif         PATH   Apptainer image path          (default: see below)
#   --dry_run            Print commands without running qsub
#   -h, --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM_DIR="$(dirname "$SCRIPT_DIR")"
SIF_PATH="/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif"
JOB_SCRIPT="${SCRIPT_DIR}/qsub_patch_array_job.sh"

# ---- quantisation parameters ------------------------------------------------
BIT_WIDTH=2
GROUP_SIZE=32
NUM_STEPS=10000
RANK_SCALE=1.0
SEED=0
LAYER_THRESHOLD=0
WORKERS_PER_GPU=1024

# ---- qsub parameters --------------------------------------------------------
WALLTIME="4:00:00"
GPU_RESOURCE="gpu_1=1"
DRY_RUN=0

# ---- target models ----------------------------------------------------------
MODELS=(
    "Qwen/Qwen3.5-2B"
    "Qwen/Qwen3.5-4B"
    "Qwen/Qwen3.5-9B"
)

# ---- argument parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bit_width)      BIT_WIDTH="$2";      shift 2 ;;
        --group_size)     GROUP_SIZE="$2";     shift 2 ;;
        --num_steps)      NUM_STEPS="$2";      shift 2 ;;
        --walltime)       WALLTIME="$2";       shift 2 ;;
        --gpu_resource)   GPU_RESOURCE="$2";   shift 2 ;;
        --workers_per_gpu) WORKERS_PER_GPU="$2"; shift 2 ;;
        --sif)            SIF_PATH="$2";       shift 2 ;;
        --dry_run)        DRY_RUN=1;           shift ;;
        -h|--help)
            sed -n '2,/^set /p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---- preflight checks -------------------------------------------------------
if [[ ! -f "${SIF_PATH}" ]]; then
    cat >&2 <<EOF
ERROR: Apptainer image not found: ${SIF_PATH}
Build it first:
    cd /gs/bs/tga-artic/k-kuroki
    bash build_apptainer.sh
EOF
    exit 1
fi

if [[ ! -f "${JOB_SCRIPT}" ]]; then
    echo "ERROR: job script not found: ${JOB_SCRIPT}" >&2
    exit 1
fi

chmod +x "${JOB_SCRIPT}"

# ---- apptainer helper -------------------------------------------------------
BQQ_ROOT="$(dirname "$(dirname "${LM_DIR}")")"
HF_HOME="/gs/bs/tga-artic/k-kuroki/hf_cache"

run_in_container() {
    apptainer exec --nv \
        --bind "${HOME}:${HOME}" \
        --bind "${BQQ_ROOT}:${BQQ_ROOT}" \
        --bind "${HF_HOME}:${HF_HOME}" \
        --env  "HF_HOME=${HF_HOME}" \
        --pwd  "${LM_DIR}" \
        "${SIF_PATH}" \
        "$@"
}

# ---- process each model -----------------------------------------------------
declare -a SUBMITTED_JOBS=()

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_BASENAME="${MODEL_NAME##*/}"
    CACHE_DIR="${LM_DIR}/cache/${MODEL_BASENAME}-layer${LAYER_THRESHOLD}"
    SAVE_DIR="${LM_DIR}/bqq_compressed_data/${MODEL_BASENAME}-${GROUP_SIZE}gs-${NUM_STEPS}step"
    JOB_DIR="${LM_DIR}/qsub_jobs/${MODEL_BASENAME}-bit${BIT_WIDTH}-gs${GROUP_SIZE}"
    TARGETS_LIST_FILE="${JOB_DIR}/targets.txt"
    LOG_DIR="${JOB_DIR}/logs"

    echo ""
    echo "============================================================"
    echo "  Model : ${MODEL_NAME}"
    echo "  Cache : ${CACHE_DIR}"
    echo "  Save  : ${SAVE_DIR}"
    echo "============================================================"

    mkdir -p "${JOB_DIR}" "${LOG_DIR}" "${SAVE_DIR}"

    # -- Step 1: prepare-cache ------------------------------------------------
    echo "[${MODEL_BASENAME}] Step 1/3: prepare-cache"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "  (dry-run) would run: python weight_aware_quant_cached.py prepare-cache ..."
    else
        run_in_container python weight_aware_quant_cached.py prepare-cache \
            --model_name     "${MODEL_NAME}" \
            --layer_threshold "${LAYER_THRESHOLD}" \
            --cache_dir      "${CACHE_DIR}"
    fi

    # -- Step 2: list-targets -------------------------------------------------
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

    # -- Step 3: submit array job (1 task = 1 target) -------------------------
    echo "[${MODEL_BASENAME}] Step 3/3: submitting qsub array job (1-${N_JOBS} targets)"

    QSUB_VARS="HF_HOME=${HF_HOME}"
    QSUB_VARS+=",TARGETS_LIST_FILE=${TARGETS_LIST_FILE}"
    QSUB_VARS+=",CACHE_DIR=${CACHE_DIR}"
    QSUB_VARS+=",SAVE_DIR=${SAVE_DIR}"
    QSUB_VARS+=",SIF_PATH=${SIF_PATH}"
    QSUB_VARS+=",LM_SCRIPT_DIR=${LM_DIR}"
    QSUB_VARS+=",BIT_WIDTH=${BIT_WIDTH}"
    QSUB_VARS+=",GROUP_SIZE=${GROUP_SIZE}"
    QSUB_VARS+=",NUM_STEPS=${NUM_STEPS}"
    QSUB_VARS+=",RANK_SCALE=${RANK_SCALE}"
    QSUB_VARS+=",SEED=${SEED}"
    QSUB_VARS+=",WORKERS_PER_GPU=${WORKERS_PER_GPU}"

    JOB_NAME="bqq_${MODEL_BASENAME}"

    QSUB_CMD=(
        qsub
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

# ---- summary ----------------------------------------------------------------
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
