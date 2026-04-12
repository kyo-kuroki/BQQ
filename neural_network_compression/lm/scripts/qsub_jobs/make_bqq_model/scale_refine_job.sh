#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l gpu_1=1
#$ -l h_rt=2:00:00

echo "=== scale_refine_bqq: ${MODEL_BASENAME} ==="
echo "Host: $(hostname)  Date: $(date -Iseconds)"

apptainer exec --nv \
    --bind "${HOME}:${HOME}" \
    --bind "${BQQ_ROOT}:${BQQ_ROOT}" \
    --bind "${HF_HOME}:${HF_HOME}" \
    --env  "HF_HOME=${HF_HOME}" \
    --pwd  "${SCRIPT_DIR}" \
    "${SIF_PATH}" \
    python scale_refine_bqq.py \
        --model_name "${MODEL_NAME}" \
        --bqq_model  "${BQQ_MODEL_PATH}" \
        --output     "${REFINED_OUTPUT_PATH}" \
        --dataset    wikitext2 \
        --nsamples   128 \
        --seqlen     2048 \
        --device     cuda:0 \
        --damping    1e-6

echo "Done: $(date -Iseconds)"
