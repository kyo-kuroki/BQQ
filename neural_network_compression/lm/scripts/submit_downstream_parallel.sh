#!/bin/bash
# Submit one qsub job per (model, task) combination.
# Skips jobs where the result is already present in the CSV.
# Usage: bash submit_downstream_parallel.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${LM_DIR}/results"
QDATA_DIR="${LM_DIR}/quantized_model_data"
LOG_DIR="${SCRIPT_DIR}/qsub_jobs/logs"
TASK_EVAL_SCRIPT="${SCRIPT_DIR}/qsub_task_eval.sh"
DRY_RUN=false

for arg in "$@"; do
    [[ "${arg}" == "--dry-run" ]] && DRY_RUN=true
done

mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Model definitions: label  model_name  model_path (empty = FP16 baseline)
# ---------------------------------------------------------------------------
declare -a LABELS MODEL_NAMES MODEL_PATHS
add_model() { LABELS+=("$1"); MODEL_NAMES+=("$2"); MODEL_PATHS+=("$3"); }

add_model "Qwen3.5-2B-2bit-32gs-blockwise" "Qwen/Qwen3.5-2B" "${QDATA_DIR}/Qwen3.5-2B-2bit-32gs-blockwise.pth"
add_model "Qwen3.5-2B-3bit-32gs-blockwise" "Qwen/Qwen3.5-2B" "${QDATA_DIR}/Qwen3.5-2B-3bit-32gs-blockwise.pth"
add_model "Qwen3.5-4B-2bit-32gs-blockwise" "Qwen/Qwen3.5-4B" "${QDATA_DIR}/Qwen3.5-4B-2bit-32gs-blockwise.pth"
add_model "Qwen3.5-4B-3bit-32gs-blockwise" "Qwen/Qwen3.5-4B" "${QDATA_DIR}/Qwen3.5-4B-3bit-32gs-blockwise.pth"
add_model "Qwen3.5-9B-2bit-32gs-blockwise" "Qwen/Qwen3.5-9B" "${QDATA_DIR}/Qwen3.5-9B-2bit-32gs-blockwise.pth"
add_model "Qwen3.5-9B-3bit-32gs-blockwise" "Qwen/Qwen3.5-9B" "${QDATA_DIR}/Qwen3.5-9B-3bit-32gs-blockwise.pth"
add_model "Qwen_Qwen3.5-2B"                "Qwen/Qwen3.5-2B" ""
add_model "Qwen_Qwen3.5-4B"                "Qwen/Qwen3.5-4B" ""
add_model "Qwen_Qwen3.5-9B"                "Qwen/Qwen3.5-9B" ""

# ---------------------------------------------------------------------------
# Task definitions: logical_name  lm_eval_task_name
# (used for CSV skip-check: check if any column starting with lm_eval_task_name_ exists)
# ---------------------------------------------------------------------------
TASK_NAMES=(
    arc_easy
    arc_challenge
    hellaswag
    winogrande
    piqa
    boolq
    race
    mmlu
    mmlu_pro
    gsm8k
    mgsm
    math
)

# lm_eval task name for each logical name (same order)
declare -A LM_TASK_NAME
LM_TASK_NAME[arc_easy]="arc_easy"
LM_TASK_NAME[arc_challenge]="arc_challenge"
LM_TASK_NAME[hellaswag]="hellaswag"
LM_TASK_NAME[winogrande]="winogrande"
LM_TASK_NAME[piqa]="piqa"
LM_TASK_NAME[boolq]="boolq"
LM_TASK_NAME[race]="race"
LM_TASK_NAME[mmlu]="mmlu"
LM_TASK_NAME[mmlu_pro]="mmlu_pro"
LM_TASK_NAME[gsm8k]="gsm8k"
LM_TASK_NAME[mgsm]="mgsm_direct_en"
LM_TASK_NAME[math]="leaderboard_math_hard"

# ---------------------------------------------------------------------------
# Skip check: returns 0 (already done) or 1 (needs to run)
# lm_eval always writes a "{task}_alias" column when a task completes.
# Checking for that column in the CSV header is sufficient.
# ---------------------------------------------------------------------------
task_already_done() {
    local csv_path="$1"
    local lm_task="$2"
    [[ -f "${csv_path}" ]] || return 1
    head -1 "${csv_path}" | grep -q "${lm_task}_alias"
}

# ---------------------------------------------------------------------------
# Submit loop
# ---------------------------------------------------------------------------
submitted=0
skipped=0

for i in "${!LABELS[@]}"; do
    label="${LABELS[$i]}"
    model_name="${MODEL_NAMES[$i]}"
    model_path="${MODEL_PATHS[$i]}"
    csv_path="${RESULTS_DIR}/${label}.csv"

    for task_name in "${TASK_NAMES[@]}"; do
        lm_task="${LM_TASK_NAME[$task_name]}"

        if task_already_done "${csv_path}" "${lm_task}"; then
            echo "[SKIP] ${label} / ${task_name} (already in CSV)"
            (( skipped++ )) || true
            continue
        fi

        job_name="task_${label:0:12}_${task_name:0:8}"
        log_file="${LOG_DIR}/${label}_${task_name}.o\$JOB_ID"

        env_vars="MODEL_NAME=${model_name},TASK_NAME=${task_name}"
        if [[ -n "${model_path}" ]]; then
            env_vars="${env_vars},MODEL_PATH=${model_path}"
        fi

        cmd=(
            qsub -g tga-artic
            -N "${job_name}"
            -o "${LOG_DIR}/${label}_${task_name}.o\$JOB_ID"
            -v "${env_vars}"
            "${TASK_EVAL_SCRIPT}"
        )

        if "${DRY_RUN}"; then
            echo "[DRY-RUN] ${cmd[*]}"
        else
            echo "[SUBMIT] ${label} / ${task_name}"
            "${cmd[@]}"
        fi
        (( submitted++ )) || true
    done
done

echo ""
echo "submitted=${submitted}  skipped=${skipped}"
