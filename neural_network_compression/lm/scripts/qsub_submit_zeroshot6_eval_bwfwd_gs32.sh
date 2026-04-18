#!/bin/bash
# Submit PPL (wikitext2+c4) + 6 zero-shot downstream tasks for blockwise-forward gs32 models.
# Tasks: arc_easy arc_challenge hellaswag winogrande piqa boolq
# Each task runs as an independent SGE job (fully parallel).
#
# Usage:
#   bash qsub_submit_zeroshot6_eval_bwfwd_gs32.sh          # all models
#   bash qsub_submit_zeroshot6_eval_bwfwd_gs32.sh 2B 2     # 2B-2bit only
#   bash qsub_submit_zeroshot6_eval_bwfwd_gs32.sh 9B       # 9B all bits

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LM_DIR="$(dirname "$SCRIPT_DIR")"

TASKS=(arc_easy arc_challenge hellaswag winogrande piqa boolq)
TASK_WALLTIME=2:00:00

# TAG -> "HF_NAME|PTH_FILE|PPL_WALLTIME|BATCH_SIZE"  (| avoids collision with : in walltime)
declare -A MODEL_INFO=(
    [2B-2]="Qwen/Qwen3.5-2B|Qwen3.5-2B-2bit-32gs-blockwise-fwd.pth|2:00:00|64"
    [2B-3]="Qwen/Qwen3.5-2B|Qwen3.5-2B-3bit-32gs-blockwise-fwd.pth|2:00:00|64"
    [4B-2]="Qwen/Qwen3.5-4B|Qwen3.5-4B-2bit-32gs-blockwise-fwd.pth|3:00:00|32"
    [4B-3]="Qwen/Qwen3.5-4B|Qwen3.5-4B-3bit-32gs-blockwise-fwd.pth|3:00:00|32"
    [9B-2]="Qwen/Qwen3.5-9B|Qwen3.5-9B-2bit-32gs-blockwise-fwd.pth|4:00:00|16"
    [9B-3]="Qwen/Qwen3.5-9B|Qwen3.5-9B-3bit-32gs-blockwise-fwd.pth|4:00:00|16"
)

FILTER_MODEL="${1:-}"
FILTER_BIT="${2:-}"

for TAG in 2B-2 2B-3 4B-2 4B-3 9B-2 9B-3; do
    SIZE="${TAG%-*}"
    BIT="${TAG#*-}"

    [[ -n "${FILTER_MODEL}" && "${FILTER_MODEL}" != "${SIZE}" ]] && continue
    [[ -n "${FILTER_BIT}"  && "${FILTER_BIT}"  != "${BIT}"  ]] && continue

    IFS='|' read -r HF_NAME PTH_FILE PPL_WT BS <<< "${MODEL_INFO[$TAG]}"
    MODEL_PATH="${LM_DIR}/quantized_model_data/${PTH_FILE}"

    if [[ ! -f "${MODEL_PATH}" ]]; then
        echo "[${TAG}] ${MODEL_PATH} not found, skipping."
        continue
    fi

    LOG_DIR="${SCRIPT_DIR}/qsub_jobs/zeroshot6-bwfwd-gs32-${TAG}bit/logs"
    mkdir -p "${LOG_DIR}"
    JOB_PREFIX="z6_gs32_${SIZE}_${BIT}b"

    # --- PPL job: skip if c4_ppl already in CSV ---
    CSV_PATH="${LM_DIR}/results/${PTH_FILE%.pth}.csv"
    if [[ -f "${CSV_PATH}" ]] && head -1 "${CSV_PATH}" | grep -q "c4_ppl"; then
        echo "[${TAG}] PPL already in CSV, skipping."
    else
        echo -n "[${TAG}] PPL ... "
        qsub -g tga-artic \
            -N "${JOB_PREFIX}_ppl" \
            -o "${LOG_DIR}/" \
            -l h_rt="${PPL_WT}" \
            -v MODEL_NAME="${HF_NAME}" \
            -v MODEL_PATH="${MODEL_PATH}" \
            "${SCRIPT_DIR}/qsub_ppl_eval.sh"
    fi

    # --- Per-task downstream jobs ---
    for TASK in "${TASKS[@]}"; do
        echo -n "[${TAG}] ${TASK} (bs=${BS}) ... "
        qsub -g tga-artic \
            -N "${JOB_PREFIX}_${TASK}" \
            -o "${LOG_DIR}/" \
            -l h_rt="${TASK_WALLTIME}" \
            -v MODEL_NAME="${HF_NAME}" \
            -v MODEL_PATH="${MODEL_PATH}" \
            -v TASK_NAME="${TASK}" \
            -v BATCH_SIZE="${BS}" \
            "${SCRIPT_DIR}/qsub_task_eval.sh"
    done
    echo ""
done

echo ""
echo "All jobs submitted."
