#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l gpu_1=1
#$ -l h_rt=8:00:00

# Re-run a single block with gradient clipping + best checkpoint restore.
# Required env vars (passed via qsub -v):
#   MODEL_NAME  - e.g. Qwen/Qwen3.5-2B
#   BIT_WIDTH   - e.g. 2 or 3
#   BLOCK_IDX   - block index to re-run
#   SAVE_DIR    - output directory

set -euo pipefail

SIF_PATH="/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif"
HF_HOME="${HF_HOME:-/gs/bs/tga-artic/k-kuroki/hf_cache}"
LM_DIR="/gs/bs/tga-artic/k-kuroki/BQQ/neural_network_compression/lm"
BQQ_ROOT="/gs/bs/tga-artic/k-kuroki/BQQ"

echo "========================================"
echo "Re-run block_${BLOCK_IDX} (${MODEL_NAME} ${BIT_WIDTH}-bit)"
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
        --bit_width "${BIT_WIDTH}" \
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

echo "Done: block_${BLOCK_IDX}  $(date --iso-8601=seconds)"
