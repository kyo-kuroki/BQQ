#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l gpu_1=1
#$ -l h_rt=4:00:00

set -euo pipefail

SIF_PATH="/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif"
HF_HOME="/gs/bs/tga-artic/k-kuroki/hf_cache"
LM_DIR="/gs/bs/tga-artic/k-kuroki/BQQ/neural_network_compression/lm"
BQQ_ROOT="/gs/bs/tga-artic/k-kuroki/BQQ"

MODEL_NAME="Qwen/Qwen3.5-4B"
BIT=3
BLOCK_DIR="${LM_DIR}/blockwise_output/Qwen3.5-4B-${BIT}bit-gs32-20000step-c4"
OUTPUT_DIR="${LM_DIR}/quantized_model_data"
ASSEMBLED_PATH="${OUTPUT_DIR}/Qwen3.5-4B-${BIT}bit-32gs-blockwise.pth"

APPTAINER_CMD=(
    apptainer exec --nv
    --bind "${HOME}:${HOME}"
    --bind "${BQQ_ROOT}:${BQQ_ROOT}"
    --bind "${HF_HOME}:${HF_HOME}"
    --env "HF_HOME=${HF_HOME}"
    --pwd "${LM_DIR}"
    "${SIF_PATH}"
)

echo "Step 1: Assembling model..."
"${APPTAINER_CMD[@]}" \
    python build_bqq_model.py assemble \
        --model_name "${MODEL_NAME}" \
        --block_dir "${BLOCK_DIR}" \
        --bit_width "${BIT}" \
        --group_size 32 \
        --output_dir "${OUTPUT_DIR}"

echo "Step 2: Evaluating PPL..."
"${APPTAINER_CMD[@]}" \
    python evaluation.py \
        --model_path "${ASSEMBLED_PATH}" \
        --model_name "${MODEL_NAME}" \
        --device cuda:0 \
        --seq_len 2048

echo "Done: $(date --iso-8601=seconds)"
