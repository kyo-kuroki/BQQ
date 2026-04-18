#!/bin/bash
#$ -cwd
#$ -l gpu_1=1,h_rt=1:30:00
#$ -o /gs/bs/tga-artic/k-kuroki/BQQ/neural_network_compression/lm/scripts/toy_compare_schedule.log
#$ -e /gs/bs/tga-artic/k-kuroki/BQQ/neural_network_compression/lm/scripts/toy_compare_schedule.err

set -euo pipefail

SIF_PATH="/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif"
HF_HOME="/gs/bs/tga-artic/k-kuroki/hf_cache"
BQQ_ROOT="/gs/bs/tga-artic/k-kuroki/BQQ"
LM_DIR="${BQQ_ROOT}/neural_network_compression/lm"

echo "========================================"
echo "Toy experiment: linear vs geometric schedule"
echo "Host: $(hostname), Date: $(date --iso-8601=seconds)"
echo "========================================"

apptainer exec --nv \
    --bind "${HOME}:${HOME}" \
    --bind "${BQQ_ROOT}:${BQQ_ROOT}" \
    --bind "${HF_HOME}:${HF_HOME}" \
    --env "HF_HOME=${HF_HOME}" \
    --pwd "${LM_DIR}" \
    "${SIF_PATH}" \
    python scripts/toy_compare_progressive_schedule.py \
        --model_name "Qwen/Qwen3.5-2B" \
        --block_idx 0 \
        --bit_width 2 \
        --group_size 64 \
        --num_steps 500 \
        --num_rounds 8 \
        --epochs 3 \
        --lr 1e-4 \
        --nsamples 64 \
        --seqlen 512 \
        --device cuda:0

echo "Done: $(date --iso-8601=seconds)"
