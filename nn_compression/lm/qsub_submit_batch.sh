#!/bin/bash
# Submit batch-mode 3-bit quantization jobs for 4B and 9B models.
# Usage: bash qsub_submit_batch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BQQ_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
SIF_PATH="/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif"

submit_job() {
    local MODEL_TAG="$1"        # e.g. "4B"
    local CACHE_DIR="$2"        # absolute path to cache dir
    local SAVE_DIR="$3"         # absolute path to save dir
    local TARGETS_FILE="$4"     # absolute path to targets list
    local JOB_NAME="$5"         # SGE job name
    local LOG_DIR="$6"          # log output dir

    local NUM_TARGETS
    NUM_TARGETS="$(wc -l < "$TARGETS_FILE")"
    if [[ "$NUM_TARGETS" -eq 0 ]]; then
        echo "[$MODEL_TAG] No remaining targets. Skipping."
        return
    fi

    echo "[$MODEL_TAG] Submitting $NUM_TARGETS tasks: $TARGETS_FILE"
    qsub -g tga-artic \
        -N "$JOB_NAME" \
        -o "$LOG_DIR" \
        -t 1-"${NUM_TARGETS}":1 \
        -v TARGETS_LIST_FILE="$TARGETS_FILE" \
        -v CACHE_DIR="$CACHE_DIR" \
        -v SAVE_DIR="$SAVE_DIR" \
        -v SIF_PATH="$SIF_PATH" \
        -v LM_SCRIPT_DIR="$SCRIPT_DIR" \
        "$SCRIPT_DIR/qsub_batch_3bit_job.sh"
}

# --- Qwen3.5-4B ---
submit_job "4B" \
    "$SCRIPT_DIR/cache/Qwen3.5-4B-layer4" \
    "$SCRIPT_DIR/bqq_compressed_data/Qwen3.5-4B-32gs-10000step" \
    "$SCRIPT_DIR/qsub_jobs/Qwen3.5-4B-bit3-gs32/remaining_targets.txt" \
    "bqq3b_4B" \
    "$SCRIPT_DIR/qsub_jobs/Qwen3.5-4B-bit3-gs32/logs"

# --- Qwen3.5-9B ---
submit_job "9B" \
    "$SCRIPT_DIR/cache/Qwen3.5-9B-layer0" \
    "$SCRIPT_DIR/bqq_compressed_data/Qwen3.5-9B-32gs-10000step" \
    "$SCRIPT_DIR/qsub_jobs/Qwen3.5-9B-bit3-gs32/remaining_targets.txt" \
    "bqq3b_9B" \
    "$SCRIPT_DIR/qsub_jobs/Qwen3.5-9B-bit3-gs32/logs"

echo ""
echo "All jobs submitted."
