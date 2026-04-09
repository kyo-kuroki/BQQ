#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l cpu=8
#$ -l h_rt=1:00:00

echo "=== make_bqq_model: ${MODEL_BASENAME} ==="
echo "Host: $(hostname)  Date: $(date -Iseconds)"

apptainer exec \
    --bind "${HOME}:${HOME}" \
    --bind "${BQQ_ROOT}:${BQQ_ROOT}" \
    --bind "${HF_HOME}:${HF_HOME}" \
    --env  "HF_HOME=${HF_HOME}" \
    --pwd  "${SCRIPT_DIR}" \
    "${SIF_PATH}" \
    python make_bqq_model_from_compressed_data.py \
        --model_name          "${MODEL_NAME}" \
        --bit_widths          2 \
        --group_size          32 \
        --num_steps           10000 \
        --compressed_data_dir "${COMPRESSED_DATA_DIR}" \
        --output_dir          "${OUTPUT_DIR}" \
        --device              cpu

echo "Done: $(date -Iseconds)"
