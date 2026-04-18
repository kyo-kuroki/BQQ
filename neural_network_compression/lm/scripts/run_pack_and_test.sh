#!/bin/bash
set -euo pipefail

SIF_PATH="/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif"
HF_HOME="/gs/bs/tga-artic/k-kuroki/hf_cache"
BQQ_ROOT="/gs/bs/tga-artic/k-kuroki/BQQ"
LM_DIR="${BQQ_ROOT}/neural_network_compression/lm"

APPTAINER_CMD=(apptainer exec --nv
    --bind "${HOME}:${HOME}"
    --bind "${BQQ_ROOT}:${BQQ_ROOT}"
    --bind "${HF_HOME}:${HF_HOME}"
    --env "HF_HOME=${HF_HOME}"
    --env "OPENBLAS_NUM_THREADS=1"
    --env "OMP_NUM_THREADS=1"
    --env "MKL_NUM_THREADS=1"
    --pwd "${LM_DIR}"
    "${SIF_PATH}")

UNPACKED="${LM_DIR}/quantized_model_data/Qwen3.5-2B-2bit-64gs-blockwise.pth"
PACKED="${LM_DIR}/quantized_model_data/Qwen3.5-2B-2bit-64gs-blockwise-packed.pth"

echo "========================================"
echo "Step 1: Pack existing model"
echo "========================================"
if [[ -f "${PACKED}" ]]; then
    echo "Packed model already exists: ${PACKED}"
else
    "${APPTAINER_CMD[@]}" python build_bqq_model.py pack \
        --input  "${UNPACKED}" \
        --output "${PACKED}"
fi

echo ""
echo "========================================"
echo "Step 2: Compare outputs"
echo "========================================"
"${APPTAINER_CMD[@]}" python scripts/test_packed_model.py \
    --unpacked "${UNPACKED}" \
    --packed   "${PACKED}" \
    --device   cpu \
    --seq_len  32
