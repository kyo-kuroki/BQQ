#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l gpu_1=1
#$ -l h_rt=1:00:00

echo "=== PPL eval: ${MODEL_BASENAME} ==="
echo "Host: $(hostname)  Date: $(date -Iseconds)"
echo "Model: ${BQQ_MODEL_PATH}"

apptainer exec --nv \
    --bind "${HOME}:${HOME}" \
    --bind "${BQQ_ROOT}:${BQQ_ROOT}" \
    --bind "${HF_HOME}:${HF_HOME}" \
    --env  "HF_HOME=${HF_HOME}" \
    --env  "TRANSFORMERS_OFFLINE=1" \
    --env  "HF_DATASETS_OFFLINE=0" \
    --pwd  "${SCRIPT_DIR}" \
    "${SIF_PATH}" \
    python evaluation.py \
        --model_name  "${MODEL_NAME}" \
        --model_path  "${BQQ_MODEL_PATH}" \
        --device      cuda:0

echo "Done: $(date -Iseconds)"
