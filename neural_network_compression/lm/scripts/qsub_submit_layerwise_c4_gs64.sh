#!/bin/bash
# Submit layer-wise BQQ quantization jobs: c4, gs=64, 30000 steps.
# One job per model × bit combination (all layers processed sequentially in one GPU run).
# Output directories use layerwise_output/ prefix with gs64-30000step-c4 suffix.
#
# Usage:
#   bash qsub_submit_layerwise_c4_gs64.sh              # submit all 6 combinations
#   bash qsub_submit_layerwise_c4_gs64.sh 2B 3         # submit only 2B 3-bit
#   bash qsub_submit_layerwise_c4_gs64.sh 9B           # submit 9B 2-bit and 3-bit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LM_DIR="$(dirname "$SCRIPT_DIR")"
SIF_PATH="/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif"
HF_HOME="/gs/bs/tga-artic/k-kuroki/hf_cache"

GROUP_SIZE=64
NUM_STEPS=30000
DATASET=c4
NSAMPLES=128
SEQLEN=2048
WALLTIME="8:00:00"

submit_layerwise() {
    local MODEL_TAG="$1"   # e.g. "2B"
    local BIT_WIDTH="$2"   # e.g. 2 or 3

    local MODEL_NAME="Qwen/Qwen3.5-${MODEL_TAG}"
    local SAVE_DIR="${LM_DIR}/layerwise_output/Qwen3.5-${MODEL_TAG}-${BIT_WIDTH}bit-gs${GROUP_SIZE}-${NUM_STEPS}step-${DATASET}"
    local LOG_DIR="${SCRIPT_DIR}/qsub_jobs/layerwise-${MODEL_TAG}-bit${BIT_WIDTH}-gs64/logs"
    local JOB_NAME="lw64_${MODEL_TAG}_${BIT_WIDTH}b"

    mkdir -p "${LOG_DIR}"
    mkdir -p "${SAVE_DIR}"

    echo "[${MODEL_TAG} ${BIT_WIDTH}-bit gs64 layerwise] Submitting job"

    qsub -g tga-artic \
        -N "${JOB_NAME}" \
        -o "${LOG_DIR}/" \
        -l h_rt="${WALLTIME}" \
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
        "${SCRIPT_DIR}/qsub_layerwise_job.sh"
}

BITS=(2 3)

FILTER_MODEL="${1:-}"
FILTER_BIT="${2:-}"

for TAG in 2B 4B 9B; do
    [[ -n "${FILTER_MODEL}" && "${FILTER_MODEL}" != "${TAG}" ]] && continue

    for BIT in "${BITS[@]}"; do
        [[ -n "${FILTER_BIT}" && "${FILTER_BIT}" != "${BIT}" ]] && continue

        submit_layerwise "${TAG}" "${BIT}"
    done
done

echo ""
echo "All jobs submitted."
