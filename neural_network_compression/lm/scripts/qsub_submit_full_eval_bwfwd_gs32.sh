#!/bin/bash
# Submit full evaluation (PPL + per-task downstream) for blockwise-forward gs32 models.
# Each downstream task runs as an independent SGE job (fully parallel).
#
# Usage:
#   bash qsub_submit_full_eval_bwfwd_gs32.sh              # all models
#   bash qsub_submit_full_eval_bwfwd_gs32.sh 2B 2         # 2B-2bit only
#   bash qsub_submit_full_eval_bwfwd_gs32.sh 9B           # 9B all bits

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LM_DIR="$(dirname "$SCRIPT_DIR")"

TASKS=(arc_easy arc_challenge hellaswag winogrande piqa boolq race mmlu mmlu_pro gsm8k mgsm math)

# walltime per task type
declare -A TASK_WALLTIME=(
    [arc_easy]=2:00:00    [arc_challenge]=2:00:00
    [hellaswag]=2:00:00   [winogrande]=2:00:00
    [piqa]=2:00:00        [boolq]=2:00:00
    [race]=3:00:00        [mmlu]=4:00:00
    [mmlu_pro]=4:00:00    [gsm8k]=4:00:00
    [mgsm]=4:00:00        [math]=6:00:00
)
PPL_WALLTIME_2B=2:00:00
PPL_WALLTIME_4B=3:00:00
PPL_WALLTIME_9B=4:00:00

# Model definitions: TAG -> "HF_NAME:PTH_FILE:PPL_WALLTIME"
declare -A MODEL_INFO=(
    [2B-2]="Qwen/Qwen3.5-2B:Qwen3.5-2B-2bit-32gs-blockwise-fwd.pth:${PPL_WALLTIME_2B}"
    [2B-3]="Qwen/Qwen3.5-2B:Qwen3.5-2B-3bit-32gs-blockwise-fwd.pth:${PPL_WALLTIME_2B}"
    [4B-2]="Qwen/Qwen3.5-4B:Qwen3.5-4B-2bit-32gs-blockwise-fwd.pth:${PPL_WALLTIME_4B}"
    [4B-3]="Qwen/Qwen3.5-4B:Qwen3.5-4B-3bit-32gs-blockwise-fwd.pth:${PPL_WALLTIME_4B}"
    [9B-2]="Qwen/Qwen3.5-9B:Qwen3.5-9B-2bit-32gs-blockwise-fwd.pth:${PPL_WALLTIME_9B}"
    [9B-3]="Qwen/Qwen3.5-9B:Qwen3.5-9B-3bit-32gs-blockwise-fwd.pth:${PPL_WALLTIME_9B}"
)

FILTER_MODEL="${1:-}"
FILTER_BIT="${2:-}"

for TAG in 2B-2 2B-3 4B-2 4B-3 9B-2 9B-3; do
    SIZE="${TAG%-*}"
    BIT="${TAG#*-}"

    [[ -n "${FILTER_MODEL}" && "${FILTER_MODEL}" != "${SIZE}" ]] && continue
    [[ -n "${FILTER_BIT}"  && "${FILTER_BIT}"  != "${BIT}"  ]] && continue

    IFS=':' read -r HF_NAME PTH_FILE PPL_WT <<< "${MODEL_INFO[$TAG]}"
    MODEL_PATH="${LM_DIR}/quantized_model_data/${PTH_FILE}"

    if [[ ! -f "${MODEL_PATH}" ]]; then
        echo "[${TAG}] ${MODEL_PATH} not found, skipping."
        continue
    fi

    LOG_DIR="${SCRIPT_DIR}/qsub_jobs/fulleval-bwfwd-gs32-${TAG}bit/logs"
    mkdir -p "${LOG_DIR}"
    JOB_PREFIX="fe_gs32_${SIZE}_${BIT}b"

    # --- PPL job (wikitext2 + c4) ---
    echo -n "[${TAG}] PPL ... "
    qsub -g tga-artic \
        -N "${JOB_PREFIX}_ppl" \
        -o "${LOG_DIR}/" \
        -l h_rt="${PPL_WT}" \
        -v MODEL_NAME="${HF_NAME}" \
        -v MODEL_PATH="${MODEL_PATH}" \
        "${SCRIPT_DIR}/qsub_ppl_eval.sh"

    # --- Per-task downstream jobs ---
    for TASK in "${TASKS[@]}"; do
        WT="${TASK_WALLTIME[$TASK]}"
        echo -n "[${TAG}] ${TASK} ... "
        qsub -g tga-artic \
            -N "${JOB_PREFIX}_${TASK}" \
            -o "${LOG_DIR}/" \
            -l h_rt="${WT}" \
            -v MODEL_NAME="${HF_NAME}" \
            -v MODEL_PATH="${MODEL_PATH}" \
            -v TASK_NAME="${TASK}" \
            "${SCRIPT_DIR}/qsub_task_eval.sh"
    done
    echo ""
done

echo ""
echo "All jobs submitted."
