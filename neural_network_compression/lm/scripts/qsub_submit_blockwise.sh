#!/bin/bash
# Submit block-wise BQQ quantization jobs for Qwen3.5-{2B,4B,9B} x {2,3}-bit.
#
# Usage:
#   bash qsub_submit_blockwise.sh              # submit all 6 combinations
#   bash qsub_submit_blockwise.sh 2B 3         # submit only 2B 3-bit
#   bash qsub_submit_blockwise.sh 9B           # submit 9B 2-bit and 3-bit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LM_DIR="$(dirname "$SCRIPT_DIR")"
SIF_PATH="/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif"
HF_HOME="/gs/bs/tga-artic/k-kuroki/hf_cache"

GROUP_SIZE=32
NUM_STEPS=20000
DATASET=c4
NSAMPLES=128
SEQLEN=2048
EPOCHS=5
LR=1e-5
WALLTIME="8:00:00"
TC=10  # max concurrent tasks per array job

submit_blockwise() {
    local MODEL_TAG="$1"   # e.g. "2B"
    local BIT_WIDTH="$2"   # e.g. 2 or 3
    local N_BLOCKS="$3"    # number of transformer blocks

    local MODEL_NAME="Qwen/Qwen3.5-${MODEL_TAG}"
    local SAVE_DIR="${LM_DIR}/blockwise_output/Qwen3.5-${MODEL_TAG}-${BIT_WIDTH}bit-gs${GROUP_SIZE}-${NUM_STEPS}step-${DATASET}"
    local LOG_DIR="${SCRIPT_DIR}/qsub_jobs/blockwise-${MODEL_TAG}-bit${BIT_WIDTH}/logs"
    local JOB_NAME="bw_${MODEL_TAG}_${BIT_WIDTH}b"

    mkdir -p "${LOG_DIR}"
    mkdir -p "${SAVE_DIR}"

    # Count already completed blocks
    local DONE=0
    for ((i=0; i<N_BLOCKS; i++)); do
        [[ -f "${SAVE_DIR}/block_${i}.pth" ]] && ((DONE++)) || true
    done

    if [[ "${DONE}" -eq "${N_BLOCKS}" ]]; then
        echo "[${MODEL_TAG} ${BIT_WIDTH}-bit] All ${N_BLOCKS} blocks done. Skipping."
        return
    fi

    echo "[${MODEL_TAG} ${BIT_WIDTH}-bit] Submitting ${N_BLOCKS} tasks (${DONE}/${N_BLOCKS} already done)"

    qsub -g tga-artic \
        -N "${JOB_NAME}" \
        -o "${LOG_DIR}/" \
        -l h_rt="${WALLTIME}" \
        -t "1-${N_BLOCKS}":1 \
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
        -v EPOCHS="${EPOCHS}" \
        -v LR="${LR}" \
        "${SCRIPT_DIR}/qsub_blockwise_job.sh"
}

# Model definitions: TAG NUM_BLOCKS
declare -A MODELS=(
    [2B]=24
    [4B]=32
    [9B]=32
)

BITS=(2 3)

# Parse optional arguments
FILTER_MODEL="${1:-}"
FILTER_BIT="${2:-}"

for TAG in 2B 4B 9B; do
    # Skip if model filter is set and doesn't match
    [[ -n "${FILTER_MODEL}" && "${FILTER_MODEL}" != "${TAG}" ]] && continue

    N_BLOCKS="${MODELS[$TAG]}"

    for BIT in "${BITS[@]}"; do
        # Skip if bit filter is set and doesn't match
        [[ -n "${FILTER_BIT}" && "${FILTER_BIT}" != "${BIT}" ]] && continue

        submit_blockwise "${TAG}" "${BIT}" "${N_BLOCKS}"
    done
done

echo ""
echo "All jobs submitted."
