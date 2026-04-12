#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l gpu_1=1
#$ -l h_rt=4:00:00

# Evaluate PPL for 2B-2bit, 2B-3bit, 4B-2bit blockwise models (post fix).

set -euo pipefail

SIF_PATH="/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif"
HF_HOME="/gs/bs/tga-artic/k-kuroki/hf_cache"
LM_DIR="/gs/bs/tga-artic/k-kuroki/BQQ/neural_network_compression/lm"
BQQ_ROOT="/gs/bs/tga-artic/k-kuroki/BQQ"

APPTAINER_CMD=(
    apptainer exec --nv
    --bind "${HOME}:${HOME}"
    --bind "${BQQ_ROOT}:${BQQ_ROOT}"
    --bind "${HF_HOME}:${HF_HOME}"
    --env "HF_HOME=${HF_HOME}"
    --pwd "${LM_DIR}"
    "${SIF_PATH}"
)

eval_model() {
    local MODEL_NAME="$1"
    local MODEL_TAG="$2"
    local BIT="$3"

    local BLOCK_DIR="${LM_DIR}/blockwise_output/${MODEL_TAG}-${BIT}bit-gs32-20000step-c4"
    local OUTPUT_DIR="${LM_DIR}/quantized_model_data"
    local ASSEMBLED="${OUTPUT_DIR}/${MODEL_TAG}-${BIT}bit-32gs-blockwise.pth"

    echo ""
    echo "========================================"
    echo "${MODEL_TAG} ${BIT}-bit blockwise"
    echo "========================================"

    "${APPTAINER_CMD[@]}" \
        python build_bqq_model.py assemble \
            --model_name "${MODEL_NAME}" \
            --block_dir "${BLOCK_DIR}" \
            --bit_width "${BIT}" \
            --group_size 32 \
            --output_dir "${OUTPUT_DIR}"

    "${APPTAINER_CMD[@]}" \
        python evaluation.py \
            --model_path "${ASSEMBLED}" \
            --model_name "${MODEL_NAME}" \
            --device cuda:0 \
            --seq_len 2048
}

eval_model "Qwen/Qwen3.5-2B" "Qwen3.5-2B" 2
eval_model "Qwen/Qwen3.5-2B" "Qwen3.5-2B" 3
eval_model "Qwen/Qwen3.5-4B" "Qwen3.5-4B" 2

echo ""
echo "All done: $(date --iso-8601=seconds)"
