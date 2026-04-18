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
    --pwd "${LM_DIR}"
    "${SIF_PATH}")

echo "========================================"
echo "9B-2bit gs64 PPL Evaluation"
echo "Host: $(hostname), Date: $(date --iso-8601=seconds)"
echo "========================================"

"${APPTAINER_CMD[@]}" python evaluation.py \
    --model_path "${LM_DIR}/quantized_model_data/Qwen3.5-9B-2bit-64gs-blockwise.pth" \
    --model_name "Qwen/Qwen3.5-9B" \
    --device cuda:0 \
    --seq_len 2048
