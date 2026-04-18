#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l gpu_1=1
#$ -l h_rt=8:00:00

# Block-wise BQQ quantization array job.
# Each SGE task quantizes one transformer block (--block_idx).
# SGE_TASK_ID maps to block_idx (1-based -> 0-based).
#
# Required env vars (passed via qsub -v):
#   MODEL_NAME    - HuggingFace model name (e.g. Qwen/Qwen3.5-4B)
#   BIT_WIDTH     - quantization bit width (2 or 3)
#   SAVE_DIR      - output directory for block_*.pth files
#   SIF_PATH      - path to Apptainer .sif image
#   LM_SCRIPT_DIR - absolute path to neural_network_compression/lm/
#
# Optional env vars:
#   GROUP_SIZE    - patch group size (default: 32)
#   NUM_STEPS     - BQQ optimization steps (default: 10000)
#   DATASET       - calibration dataset (default: wikitext2)
#   NSAMPLES      - number of calibration samples (default: 128)
#   SEQLEN        - sequence length (default: 2048)
#   EPOCHS        - block output optimization epochs (default: 5)
#   LR            - learning rate for block optimization (default: 1e-5)

set -euo pipefail

BLOCK_IDX=$((SGE_TASK_ID - 1))

echo "========================================"
echo "Block-wise BQQ Quantization"
echo "SGE_TASK_ID : ${SGE_TASK_ID}"
echo "BLOCK_IDX   : ${BLOCK_IDX}"
echo "MODEL_NAME  : ${MODEL_NAME}"
echo "BIT_WIDTH   : ${BIT_WIDTH}"
echo "Host        : $(hostname)"
echo "Date        : $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "========================================"

GROUP_SIZE="${GROUP_SIZE:-32}"
NUM_STEPS="${NUM_STEPS:-10000}"
DATASET="${DATASET:-wikitext2}"
NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-2048}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

BQQ_ROOT="$(dirname "$(dirname "${LM_SCRIPT_DIR}")")"
HF_HOME="${HF_HOME:-/gs/bs/tga-artic/k-kuroki/hf_cache}"

# Skip if already completed
BLOCK_FILE="${SAVE_DIR}/block_${BLOCK_IDX}.pth"
if [[ -f "${BLOCK_FILE}" ]]; then
    echo "Block ${BLOCK_IDX} already exists: ${BLOCK_FILE}. Skipping."
    exit 0
fi

APPTAINER_CMD=(
    apptainer exec --nv
    --bind "${HOME}:${HOME}"
    --bind "${BQQ_ROOT}:${BQQ_ROOT}"
    --bind "${HF_HOME}:${HF_HOME}"
    --env "HF_HOME=${HF_HOME}"
    --pwd "${LM_SCRIPT_DIR}"
    "${SIF_PATH}"
)

"${APPTAINER_CMD[@]}" \
    python blockwise_quant.py \
        --model_name  "${MODEL_NAME}" \
        --block_idx   "${BLOCK_IDX}" \
        --bit_width   "${BIT_WIDTH}" \
        --group_size  "${GROUP_SIZE}" \
        --num_steps   "${NUM_STEPS}" \
        --dataset     "${DATASET}" \
        --nsamples    "${NSAMPLES}" \
        --seqlen      "${SEQLEN}" \
        --epochs      "${EPOCHS}" \
        --lr          "${LR}" \
        --device      cuda:0 \
        --save_dir    "${SAVE_DIR}" \
        ${EXTRA_ARGS}

echo ""
echo "Done: block_${BLOCK_IDX}  $(date --iso-8601=seconds)"
