#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l gpu_1=1
#$ -l h_rt=8:00:00

# Single downstream task evaluation.
# Required env vars (passed via qsub -v):
#   MODEL_NAME  - e.g. Qwen/Qwen3.5-2B
#   TASK_NAME   - logical task name, e.g. arc_easy, mmlu, gsm8k
# Optional:
#   MODEL_PATH  - path to assembled .pth file (omit for FP16 baseline)
#   BATCH_SIZE  - batch size for lm_eval (default 1)

set -euo pipefail

SIF_PATH="/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif"
HF_HOME="/gs/bs/tga-artic/k-kuroki/hf_cache"
LM_DIR="/gs/bs/tga-artic/k-kuroki/BQQ/neural_network_compression/lm"
BQQ_ROOT="/gs/bs/tga-artic/k-kuroki/BQQ"

echo "========================================"
echo "Task eval: model=${MODEL_PATH:-FP16_${MODEL_NAME}} task=${TASK_NAME}"
echo "Host: $(hostname)"
echo "Date: $(date --iso-8601=seconds)"
echo "========================================"

MODEL_PATH_ARGS=()
if [[ -n "${MODEL_PATH:-}" ]]; then
    MODEL_PATH_ARGS=(--model_path "${MODEL_PATH}")
fi

apptainer exec --nv \
    --bind "${HOME}:${HOME}" \
    --bind "${BQQ_ROOT}:${BQQ_ROOT}" \
    --bind "${HF_HOME}:${HF_HOME}" \
    --env "HF_HOME=${HF_HOME}" \
    --env "PYTHONPATH=${HOME}/.local/lib/python3.12/site-packages:${PYTHONPATH:-}" \
    --pwd "${LM_DIR}" \
    "${SIF_PATH}" \
    python evaluation.py \
        "${MODEL_PATH_ARGS[@]}" \
        --model_name "${MODEL_NAME}" \
        --device cuda:0 \
        --seq_len 2048 \
        --batch_size "${BATCH_SIZE:-1}" \
        --task "${TASK_NAME}"

echo "Done: $(date --iso-8601=seconds)"
