#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l gpu_1=1
#$ -l h_rt=2:00:00

# Layer-wise BQQ quantization parallel array job.
# Each SGE task quantizes one weight matrix (--target_idx).
# SGE_TASK_ID maps to target_idx (1-based -> 0-based).
#
# Required env vars (passed via qsub -v):
#   MODEL_NAME    - HuggingFace model name (e.g. Qwen/Qwen3.5-4B)
#   BIT_WIDTH     - quantization bit width (2 or 3)
#   SAVE_DIR      - output directory for per-tensor .pth files
#   SIF_PATH      - path to Apptainer .sif image
#   LM_SCRIPT_DIR - absolute path to neural_network_compression/lm/
#
# Optional env vars:
#   GROUP_SIZE    - patch group size (default: 32)
#   NUM_STEPS     - BQQ optimization steps (default: 20000)
#   DATASET       - calibration dataset (default: wikitext2)
#   NSAMPLES      - number of calibration samples (default: 128)
#   SEQLEN        - sequence length (default: 2048)

set -euo pipefail

TARGET_IDX=$((SGE_TASK_ID - 1))

echo "========================================"
echo "Layer-wise BQQ Quantization (parallel)"
echo "SGE_TASK_ID : ${SGE_TASK_ID}"
echo "TARGET_IDX  : ${TARGET_IDX}"
echo "MODEL_NAME  : ${MODEL_NAME}"
echo "BIT_WIDTH   : ${BIT_WIDTH}"
echo "Host        : $(hostname)"
echo "Date        : $(date --iso-8601=seconds)"
echo "========================================"

GROUP_SIZE="${GROUP_SIZE:-32}"
NUM_STEPS="${NUM_STEPS:-20000}"
DATASET="${DATASET:-wikitext2}"
NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-2048}"

BQQ_ROOT="$(dirname "$(dirname "${LM_SCRIPT_DIR}")")"
HF_HOME="${HF_HOME:-/gs/bs/tga-artic/k-kuroki/hf_cache}"

mkdir -p "${SAVE_DIR}"

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
    python layerwise_quant.py \
        --model_name  "${MODEL_NAME}" \
        --target_idx  "${TARGET_IDX}" \
        --bit_width   "${BIT_WIDTH}" \
        --group_size  "${GROUP_SIZE}" \
        --num_steps   "${NUM_STEPS}" \
        --dataset     "${DATASET}" \
        --nsamples    "${NSAMPLES}" \
        --seqlen      "${SEQLEN}" \
        --main_gpu_id 0 \
        --save_dir    "${SAVE_DIR}"

echo ""
echo "Done: target_${TARGET_IDX}  $(date --iso-8601=seconds)"
