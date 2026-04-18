#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l gpu_1=1
#$ -l h_rt=4:00:00

# Assemble and evaluate PPL for blockwise-forward gs32 30000step 9B 2-bit model.

set -euo pipefail

SIF_PATH="/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif"
HF_HOME="/gs/bs/tga-artic/k-kuroki/hf_cache"
LM_DIR="/gs/bs/tga-artic/k-kuroki/BQQ/neural_network_compression/lm"
BQQ_ROOT="/gs/bs/tga-artic/k-kuroki/BQQ"

MODEL_NAME="Qwen/Qwen3.5-9B"
BIT=2
OUTPUT_DIR="${LM_DIR}/quantized_model_data"
BLOCK_DIR="${LM_DIR}/blockwise_output/Qwen3.5-9B-${BIT}bit-gs32-30000step-c4"
ASSEMBLED_PATH="${OUTPUT_DIR}/Qwen3.5-9B-${BIT}bit-32gs-blockwise-fwd.pth"
mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "Blockwise-Forward BQQ PPL Evaluation - 9B gs32 30000step ${BIT}bit"
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
        --group_size 32 \
        --output_dir "${OUTPUT_DIR}" \
        --name_suffix fwd

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

echo "Done: 9B ${BIT}-bit gs32-fwd  $(date --iso-8601=seconds)"
