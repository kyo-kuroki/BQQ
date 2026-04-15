#!/bin/bash
# Submit layer-wise BQQ quantization as block-level array jobs: c4, gs=64, 30000 steps.
# Each array task covers one transformer block (all Linear layers), using all 4 GPUs on
# the assigned node via mp.spawn.
#
# Efficiency vs per-target parallel mode:
#   - Model loaded once per task (not once per layer)
#   - All Hessians collected in a single forward pass (early-exit after block)
#   - 4 GPUs process ~7 layers in parallel -> ~4x speedup per task
#
# Usage:
#   bash qsub_submit_layerwise_c4_gs64_block.sh              # submit all 6 combinations
#   bash qsub_submit_layerwise_c4_gs64_block.sh 2B 3         # submit only 2B 3-bit
#   bash qsub_submit_layerwise_c4_gs64_block.sh 9B           # submit 9B 2-bit and 3-bit

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
WALLTIME="2:00:00"
TC=10  # max concurrent tasks (each uses 1 node / 4 GPUs)

# Number of transformer blocks per model (hardcoded to avoid login-node model load)
declare -A N_BLOCKS=(
    [2B]=24
    [4B]=32
    [9B]=32
)

submit_layerwise_block() {
    local MODEL_TAG="$1"
    local BIT_WIDTH="$2"
    local N_BLOCKS_VAL="${N_BLOCKS[$MODEL_TAG]}"

    local MODEL_NAME="Qwen/Qwen3.5-${MODEL_TAG}"
    local SAVE_DIR="${LM_DIR}/layerwise_output/Qwen3.5-${MODEL_TAG}-${BIT_WIDTH}bit-gs${GROUP_SIZE}-${NUM_STEPS}step-${DATASET}"
    local LOG_DIR="${SCRIPT_DIR}/qsub_jobs/layerwise-${MODEL_TAG}-bit${BIT_WIDTH}-gs64-block/logs"
    local JOB_NAME="lw64b_${MODEL_TAG}_${BIT_WIDTH}b"

    mkdir -p "${LOG_DIR}"
    mkdir -p "${SAVE_DIR}"

    # Count already completed blocks (a block is done when all its .pth files exist)
    # Use a simple heuristic: count unique layer indices in saved files
    local DONE=0
    for ((i=0; i<N_BLOCKS_VAL; i++)); do
        # Check if at least one file for this block exists (proxy for started/done)
        if ls "${SAVE_DIR}/model.model.layers.${i}."*.pth 2>/dev/null | grep -qv '_consolidated'; then
            # Count files for this block to check completion
            local BLOCK_FILES
            BLOCK_FILES=$(ls "${SAVE_DIR}/model.model.layers.${i}."*.pth 2>/dev/null | grep -v '_consolidated' | wc -l || echo 0)
            # Qwen3.5 has 7 linear layers per block (q/k/v/o + gate/up/down)
            if [[ "${BLOCK_FILES}" -ge 7 ]]; then
                ((DONE++)) || true
            fi
        fi
    done

    if [[ "${DONE}" -eq "${N_BLOCKS_VAL}" ]]; then
        echo "[${MODEL_TAG} ${BIT_WIDTH}-bit gs64 block] All ${N_BLOCKS_VAL} blocks done. Skipping."
        return
    fi

    echo "[${MODEL_TAG} ${BIT_WIDTH}-bit gs64 block] Submitting ${N_BLOCKS_VAL} tasks (${DONE}/${N_BLOCKS_VAL} blocks done)"

    qsub -g tga-artic \
        -N "${JOB_NAME}" \
        -o "${LOG_DIR}/" \
        -l h_rt="${WALLTIME}" \
        -l gpu_4=1 \
        -t "1-${N_BLOCKS_VAL}":1 \
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
        "${SCRIPT_DIR}/qsub_layerwise_block_job.sh"
}

BITS=(2 3)

FILTER_MODEL="${1:-}"
FILTER_BIT="${2:-}"

for TAG in 2B 4B 9B; do
    [[ -n "${FILTER_MODEL}" && "${FILTER_MODEL}" != "${TAG}" ]] && continue

    for BIT in "${BITS[@]}"; do
        [[ -n "${FILTER_BIT}" && "${FILTER_BIT}" != "${BIT}" ]] && continue

        submit_layerwise_block "${TAG}" "${BIT}"
    done
done

echo ""
echo "All jobs submitted."
