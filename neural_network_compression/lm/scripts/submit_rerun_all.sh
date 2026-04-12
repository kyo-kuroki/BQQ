#!/bin/bash
# Delete and re-submit all blocks with epoch 5 regression.
# Uses qsub_rerun_regression.sh as the job template.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LM_DIR="$(dirname "$SCRIPT_DIR")"
JOB_SCRIPT="${SCRIPT_DIR}/qsub_rerun_regression.sh"
TC=10

submit_blocks() {
    local MODEL_TAG="$1"  # e.g. 2B
    local BIT="$2"         # e.g. 2
    shift 2
    local BLOCKS=("$@")

    local MODEL_NAME="Qwen/Qwen3.5-${MODEL_TAG}"
    local SAVE_DIR="${LM_DIR}/blockwise_output/Qwen3.5-${MODEL_TAG}-${BIT}bit-gs32-20000step-c4"
    local LOG_DIR="${SCRIPT_DIR}/qsub_jobs/blockwise-${MODEL_TAG}-bit${BIT}/logs"
    local JOB_NAME="fix_${MODEL_TAG}_${BIT}b"

    mkdir -p "${LOG_DIR}"

    # Delete existing block files
    local deleted=0
    for bidx in "${BLOCKS[@]}"; do
        local pth="${SAVE_DIR}/block_${bidx}.pth"
        if [[ -f "$pth" ]]; then
            rm "$pth"
            ((deleted++))
        fi
    done
    echo "[${MODEL_TAG} ${BIT}-bit] Deleted ${deleted}/${#BLOCKS[@]} block files"

    # Submit one job per block
    for bidx in "${BLOCKS[@]}"; do
        qsub -g tga-artic \
            -N "${JOB_NAME}" \
            -o "${LOG_DIR}/" \
            -l h_rt=8:00:00 \
            -v MODEL_NAME="${MODEL_NAME}" \
            -v BIT_WIDTH="${BIT}" \
            -v BLOCK_IDX="${bidx}" \
            -v SAVE_DIR="${SAVE_DIR}" \
            "${JOB_SCRIPT}"
    done
    echo "[${MODEL_TAG} ${BIT}-bit] Submitted ${#BLOCKS[@]} jobs"
}

# 2B-2bit: 7 blocks
submit_blocks 2B 2  11 15 16 17 18 21 22

# 2B-3bit: 19 blocks
submit_blocks 2B 3  1 2 3 4 6 7 8 12 13 14 15 16 17 18 19 20 21 22 23

# 4B-2bit: 20 blocks
submit_blocks 4B 2  3 4 5 6 7 8 9 13 16 17 19 23 24 25 26 27 28 29 30 31

# 4B-3bit: 22 blocks (excluding already-rerun 3,4,11,15,27)
submit_blocks 4B 3  2 5 6 7 8 9 10 12 14 16 17 18 20 21 22 23 24 25 26 28 29 30

# 9B-2bit: 23 blocks
submit_blocks 9B 2  2 3 4 6 7 9 12 13 14 15 16 17 18 20 21 23 24 25 26 28 29 30 31

# 9B-3bit: 26 blocks
submit_blocks 9B 3  0 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 19 20 21 22 23 24 25 27 31

echo ""
echo "All re-run jobs submitted (117 total)."
