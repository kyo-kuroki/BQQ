#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l gpu_1=1
#$ -l h_rt=0:30:00

# Assemble and evaluate WikiText-2 PPL for one blockwise 4B gs64 30000step model.
# BIT_WIDTH is passed via qsub -v.

set -euo pipefail

BIT="${BIT_WIDTH:?BIT_WIDTH not set}"

SIF_PATH="/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif"
HF_HOME="/gs/bs/tga-artic/k-kuroki/hf_cache"
LM_DIR="/gs/bs/tga-artic/k-kuroki/BQQ/neural_network_compression/lm"
BQQ_ROOT="/gs/bs/tga-artic/k-kuroki/BQQ"

MODEL_NAME="Qwen/Qwen3.5-4B"
BLOCK_DIR="${LM_DIR}/blockwise_output/Qwen3.5-4B-${BIT}bit-gs64-30000step-c4"
OUTPUT_DIR="${LM_DIR}/quantized_model_data"
ASSEMBLED_PATH="${OUTPUT_DIR}/Qwen3.5-4B-${BIT}bit-64gs-blockwise.pth"

echo "========================================"
echo "Blockwise BQQ PPL Evaluation - 4B gs64 ${BIT}bit"
echo "Host: $(hostname)"
echo "Date: $(date --iso-8601=seconds)"
echo "========================================"

APPTAINER_CMD=(
    apptainer exec --nv
    --bind "${HOME}:${HOME}"
    --bind "${BQQ_ROOT}:${BQQ_ROOT}"
    --bind "${HF_HOME}:${HF_HOME}"
    --env "HF_HOME=${HF_HOME}"
    --pwd "${LM_DIR}"
    "${SIF_PATH}"
)

echo "Step 1: Assembling model from blocks..."
"${APPTAINER_CMD[@]}" \
    python build_bqq_model.py assemble \
        --model_name "${MODEL_NAME}" \
        --block_dir "${BLOCK_DIR}" \
        --bit_width "${BIT}" \
        --group_size 64 \
        --output_dir "${OUTPUT_DIR}"

if [[ ! -f "${ASSEMBLED_PATH}" ]]; then
    echo "ERROR: Assembled model not found at ${ASSEMBLED_PATH}"
    exit 1
fi

echo "Step 2: Evaluating PPL on WikiText-2..."
"${APPTAINER_CMD[@]}" \
    python evaluation.py \
        --model_path "${ASSEMBLED_PATH}" \
        --model_name "${MODEL_NAME}" \
        --device cuda:0 \
        --seq_len 2048

echo "Done: ${BIT}-bit gs64  $(date --iso-8601=seconds)"
