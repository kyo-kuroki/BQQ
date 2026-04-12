#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l gpu_1=1
#$ -l h_rt=8:00:00

# Re-run broken 4B 3-bit blocks (3, 4, 11, 27) with gradient clipping.
# SGE_TASK_ID maps: 1->block_3, 2->block_4, 3->block_11, 4->block_27

set -euo pipefail

BLOCKS=(3 4 11 27)
BLOCK_IDX=${BLOCKS[$((SGE_TASK_ID - 1))]}

SIF_PATH="/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif"
HF_HOME="/gs/bs/tga-artic/k-kuroki/hf_cache"
LM_DIR="/gs/bs/tga-artic/k-kuroki/BQQ/neural_network_compression/lm"
BQQ_ROOT="/gs/bs/tga-artic/k-kuroki/BQQ"

MODEL_NAME="Qwen/Qwen3.5-4B"
SAVE_DIR="${LM_DIR}/blockwise_output/Qwen3.5-4B-3bit-gs32-20000step-c4"

echo "========================================"
echo "Re-run block_${BLOCK_IDX} (4B 3-bit, with grad clip)"
echo "Host: $(hostname)"
echo "Date: $(date --iso-8601=seconds)"
echo "========================================"

apptainer exec --nv \
    --bind "${HOME}:${HOME}" \
    --bind "${BQQ_ROOT}:${BQQ_ROOT}" \
    --bind "${HF_HOME}:${HF_HOME}" \
    --env "HF_HOME=${HF_HOME}" \
    --pwd "${LM_DIR}" \
    "${SIF_PATH}" \
    python block_wise_quant.py \
        --model_name "${MODEL_NAME}" \
        --block_idx "${BLOCK_IDX}" \
        --bit_width 3 \
        --group_size 32 \
        --num_steps 20000 \
        --dataset c4 \
        --nsamples 128 \
        --seqlen 2048 \
        --epochs 5 \
        --lr 1e-5 \
        --max_grad_norm 1.0 \
        --device cuda:0 \
        --save_dir "${SAVE_DIR}"

echo ""
echo "Done: block_${BLOCK_IDX}  $(date --iso-8601=seconds)"
