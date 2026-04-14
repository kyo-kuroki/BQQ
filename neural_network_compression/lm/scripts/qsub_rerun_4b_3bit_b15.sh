#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l gpu_1=1
#$ -l h_rt=8:00:00

set -euo pipefail

SIF_PATH="/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif"
HF_HOME="/gs/bs/tga-artic/k-kuroki/hf_cache"
LM_DIR="/gs/bs/tga-artic/k-kuroki/BQQ/neural_network_compression/lm"
BQQ_ROOT="/gs/bs/tga-artic/k-kuroki/BQQ"

apptainer exec --nv \
    --bind "${HOME}:${HOME}" \
    --bind "${BQQ_ROOT}:${BQQ_ROOT}" \
    --bind "${HF_HOME}:${HF_HOME}" \
    --env "HF_HOME=${HF_HOME}" \
    --pwd "${LM_DIR}" \
    "${SIF_PATH}" \
    python blockwise_quant.py \
        --model_name "Qwen/Qwen3.5-4B" \
        --block_idx 15 \
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
        --save_dir "${LM_DIR}/blockwise_output/Qwen3.5-4B-3bit-gs32-20000step-c4"
