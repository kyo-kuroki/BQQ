#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l gpu_1=1
#$ -l h_rt=3:00:00

# Build and evaluate PPL for layerwise gs64 30000step 4B models (2-bit and 3-bit).

set -euo pipefail

SIF_PATH="/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif"
HF_HOME="/gs/bs/tga-artic/k-kuroki/hf_cache"
LM_DIR="/gs/bs/tga-artic/k-kuroki/BQQ/neural_network_compression/lm"
BQQ_ROOT="/gs/bs/tga-artic/k-kuroki/BQQ"

MODEL_NAME="Qwen/Qwen3.5-4B"
OUTPUT_DIR="${LM_DIR}/quantized_model_data/layerwise_gs64"
mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "Layerwise BQQ PPL Evaluation - 4B gs64 30000step"
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

for BIT in 2 3; do
    DATA_DIR="${LM_DIR}/layerwise_output/Qwen3.5-4B-${BIT}bit-gs64-30000step-c4"
    ASSEMBLED_PATH="${OUTPUT_DIR}/Qwen3.5-4B-${BIT}bit-64gs.pth"

    echo ""
    echo "========================================"
    echo "Processing ${BIT}-bit layerwise gs64 model"
    echo "========================================"

    # Step 1: Build model from layerwise output
    echo "Step 1: Building model from layerwise data..."
    "${APPTAINER_CMD[@]}" \
        python build_bqq_model.py build \
            --model_name "${MODEL_NAME}" \
            --compressed_data_dir "${DATA_DIR}" \
            --bit_widths "${BIT}" \
            --group_size 64 \
            --output_dir "${OUTPUT_DIR}"

    if [[ ! -f "${ASSEMBLED_PATH}" ]]; then
        echo "ERROR: Built model not found at ${ASSEMBLED_PATH}"
        continue
    fi

    # Step 2: Evaluate PPL
    echo "Step 2: Evaluating PPL on WikiText-2..."
    "${APPTAINER_CMD[@]}" \
        python evaluation.py \
            --model_path "${ASSEMBLED_PATH}" \
            --model_name "${MODEL_NAME}" \
            --device cuda:0 \
            --seq_len 2048

    echo "Done: ${BIT}-bit layerwise gs64  $(date --iso-8601=seconds)"
done

echo ""
echo "========================================"
echo "All evaluations complete."
echo "========================================"
