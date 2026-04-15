#!/bin/bash
# Submit layer-wise BQQ quantization as parallel array jobs: c4, gs=64, 30000 steps.
# Each array task quantizes one weight matrix independently (--target_idx).
# N_TARGETS is queried from the model via --list_targets before submission.
#
# Usage:
#   bash qsub_submit_layerwise_c4_gs64_parallel.sh              # submit all 6 combinations
#   bash qsub_submit_layerwise_c4_gs64_parallel.sh 2B 3         # submit only 2B 3-bit
#   bash qsub_submit_layerwise_c4_gs64_parallel.sh 9B           # submit 9B 2-bit and 3-bit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LM_DIR="$(dirname "$SCRIPT_DIR")"
SIF_PATH="/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif"
HF_HOME="/gs/bs/tga-artic/k-kuroki/hf_cache"
BQQ_ROOT="$(dirname "${LM_DIR}")"

GROUP_SIZE=64
NUM_STEPS=30000
DATASET=c4
NSAMPLES=128
SEQLEN=2048
WALLTIME="2:00:00"
TC=20  # max concurrent tasks per array job

get_num_targets() {
    local MODEL_NAME="$1"
    apptainer exec \
        --bind "${HOME}:${HOME}" \
        --bind "${BQQ_ROOT}:${BQQ_ROOT}" \
        --bind "${HF_HOME}:${HF_HOME}" \
        --env "HF_HOME=${HF_HOME}" \
        --pwd "${LM_DIR}" \
        "${SIF_PATH}" \
        python layerwise_quant.py \
            --model_name "${MODEL_NAME}" \
            --list_targets \
        2>/dev/null | tail -1
}

submit_layerwise_parallel() {
    local MODEL_TAG="$1"
    local BIT_WIDTH="$2"

    local MODEL_NAME="Qwen/Qwen3.5-${MODEL_TAG}"
    local SAVE_DIR="${LM_DIR}/layerwise_output/Qwen3.5-${MODEL_TAG}-${BIT_WIDTH}bit-gs${GROUP_SIZE}-${NUM_STEPS}step-${DATASET}"
    local LOG_DIR="${SCRIPT_DIR}/qsub_jobs/layerwise-${MODEL_TAG}-bit${BIT_WIDTH}-gs64/logs"
    local JOB_NAME="lw64p_${MODEL_TAG}_${BIT_WIDTH}b"

    mkdir -p "${LOG_DIR}"
    mkdir -p "${SAVE_DIR}"

    echo -n "[${MODEL_TAG} ${BIT_WIDTH}-bit gs64 parallel] Querying N_TARGETS... "
    local N_TARGETS
    N_TARGETS=$(get_num_targets "${MODEL_NAME}")
    echo "${N_TARGETS}"

    # Count already completed targets
    local DONE
    DONE=$(ls "${SAVE_DIR}"/*.pth 2>/dev/null | grep -v '_consolidated' | wc -l || echo 0)

    if [[ "${DONE}" -eq "${N_TARGETS}" ]]; then
        echo "[${MODEL_TAG} ${BIT_WIDTH}-bit gs64 parallel] All ${N_TARGETS} targets done. Skipping."
        return
    fi

    echo "[${MODEL_TAG} ${BIT_WIDTH}-bit gs64 parallel] Submitting ${N_TARGETS} tasks (${DONE}/${N_TARGETS} already done)"

    qsub -g tga-artic \
        -N "${JOB_NAME}" \
        -o "${LOG_DIR}/" \
        -l h_rt="${WALLTIME}" \
        -t "1-${N_TARGETS}":1 \
        -tc "${TC}" \
        -v MODEL_NAME="${MODEL_NAME}" \
        -v BIT_WIDTH="${BIT_WIDTH}" \
        -v SAVE_DIR="${SAVE_DIR}" \
        -v SIF_PATH="${SIF_PATH}" \
        -v LM_SCRIPT_DIR="${LM_DIR}" \
        -v HF_HOME="${HF_HOME}" \
        -v GROUP_SIZE="${GROUP_SIZE}" \
        -v NUM_STEPS="${NUM_STEPS}" \
        -v DATASET="${DATASET}" \
        -v NSAMPLES="${NSAMPLES}" \
        -v SEQLEN="${SEQLEN}" \
        "${SCRIPT_DIR}/qsub_layerwise_parallel_job.sh"
}

BITS=(2 3)

FILTER_MODEL="${1:-}"
FILTER_BIT="${2:-}"

for TAG in 2B 4B 9B; do
    [[ -n "${FILTER_MODEL}" && "${FILTER_MODEL}" != "${TAG}" ]] && continue

    for BIT in "${BITS[@]}"; do
        [[ -n "${FILTER_BIT}" && "${FILTER_BIT}" != "${BIT}" ]] && continue

        submit_layerwise_parallel "${TAG}" "${BIT}"
    done
done

echo ""
echo "All jobs submitted."
