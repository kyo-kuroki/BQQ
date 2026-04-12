#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l gpu_1=1
#$ -l h_rt=2:00:00

# Build BQQ model from consolidated data and evaluate PPL.
# Required env vars:
#   MODEL_NAME, COMPRESSED_DATA_DIR, OUTPUT_DIR, BIT_WIDTH,
#   SCRIPT_DIR, BQQ_ROOT, HF_HOME, SIF_PATH

set -euo pipefail

MODEL_BASENAME=$(basename "${MODEL_NAME}")
BIT_WIDTH="${BIT_WIDTH:-2}"
GROUP_SIZE="${GROUP_SIZE:-32}"
NUM_STEPS="${NUM_STEPS:-10000}"
BQQ_MODEL_PATH="${OUTPUT_DIR}/${MODEL_BASENAME}-${BIT_WIDTH}bit-${GROUP_SIZE}gs-${NUM_STEPS}step.pth"

echo "=== Build & Eval: ${MODEL_BASENAME} ${BIT_WIDTH}-bit ==="
echo "Host: $(hostname)  Date: $(date -Iseconds)"

APPTAINER_CMD=(
    apptainer exec --nv
    --bind "${HOME}:${HOME}"
    --bind "${BQQ_ROOT}:${BQQ_ROOT}"
    --bind "${HF_HOME}:${HF_HOME}"
    --env "HF_HOME=${HF_HOME}"
    --pwd "${SCRIPT_DIR}"
    "${SIF_PATH}"
)

# Step 1: Build BQQ model
echo "--- Building model ---"
"${APPTAINER_CMD[@]}" \
    python build_bqq_model.py \
        --model_name          "${MODEL_NAME}" \
        --bit_widths          "${BIT_WIDTH}" \
        --group_size          "${GROUP_SIZE}" \
        --num_steps           "${NUM_STEPS}" \
        --compressed_data_dir "${COMPRESSED_DATA_DIR}" \
        --output_dir          "${OUTPUT_DIR}" \
        --device              cpu

# Step 2: Evaluate PPL
echo "--- Evaluating PPL ---"
"${APPTAINER_CMD[@]}" \
    python evaluation.py \
        --model_name  "${MODEL_NAME}" \
        --model_path  "${BQQ_MODEL_PATH}" \
        --device      cuda:0

echo "Done: $(date -Iseconds)"
