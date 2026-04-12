#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_NAME="Qwen/Qwen2.5-1.5B"
BIT_WIDTH=4
GROUP_SIZE=128
NUM_STEPS=50000
RANK_SCALE=1.0
SEED=0
LAYER_THRESHOLD=0
WORKERS_PER_GPU=16
GPU_IDS="${GPU_IDS:-}"
MAX_JOBS="${MAX_JOBS:-}"
CACHE_DIR=""
SAVE_DIR=""
REFRESH_CACHE=0
OVERWRITE=0
POLL_INTERVAL=5

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --model_name NAME
  --bit_width N
  --group_size N
  --num_steps N
  --rank_scale FLOAT
  --seed N
  --layer_threshold N
  --workers_per_gpu N
  --gpu_ids ID0,ID1,...
  --max_jobs N
  --cache_dir PATH
  --save_dir PATH
  --refresh_cache
  --overwrite
  -h, --help

Environment variables:
  PYTHON_BIN   Python executable to use (default: python3)
  GPU_IDS      Comma-separated GPU ids to use when --gpu_ids is omitted
  MAX_JOBS     Maximum concurrent quantization jobs
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --bit_width)
            BIT_WIDTH="$2"
            shift 2
            ;;
        --group_size)
            GROUP_SIZE="$2"
            shift 2
            ;;
        --num_steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --rank_scale)
            RANK_SCALE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --layer_threshold)
            LAYER_THRESHOLD="$2"
            shift 2
            ;;
        --workers_per_gpu)
            WORKERS_PER_GPU="$2"
            shift 2
            ;;
        --gpu_ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --max_jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --cache_dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --refresh_cache)
            REFRESH_CACHE=1
            shift
            ;;
        --overwrite)
            OVERWRITE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$CACHE_DIR" ]]; then
    CACHE_DIR="$SCRIPT_DIR/cache/${MODEL_NAME##*/}-layer${LAYER_THRESHOLD}"
fi

if [[ -z "$GPU_IDS" ]]; then
    GPU_COUNT="$($PYTHON_BIN -c "import torch; print(torch.cuda.device_count())")"
    if [[ "$GPU_COUNT" -le 0 ]]; then
        echo "CUDA is required to run cached parallel quantization." >&2
        exit 1
    fi
    GPU_IDS="$(seq -s, 0 $((GPU_COUNT - 1)))"
fi

IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
if [[ "${#GPU_ID_ARRAY[@]}" -eq 0 ]]; then
    echo "No GPU ids were provided." >&2
    exit 1
fi

if [[ -z "$MAX_JOBS" ]]; then
    MAX_JOBS="${#GPU_ID_ARRAY[@]}"
fi

PREPARE_CMD=(
    "$PYTHON_BIN"
    "$LM_DIR/weight_aware_quant_cached.py"
    prepare-cache
    --model_name "$MODEL_NAME"
    --layer_threshold "$LAYER_THRESHOLD"
    --cache_dir "$CACHE_DIR"
)

if [[ "$REFRESH_CACHE" -eq 1 ]]; then
    PREPARE_CMD+=(--refresh_cache)
fi

echo "Preparing cache in $CACHE_DIR"
"${PREPARE_CMD[@]}"

mapfile -t PATCH_JOBS < <(
    "$PYTHON_BIN" \
    "$LM_DIR/weight_aware_quant_cached.py" \
    list-patches \
    --cache_dir "$CACHE_DIR" \
    --group_size "$GROUP_SIZE"
)

if [[ "${#PATCH_JOBS[@]}" -eq 0 ]]; then
    echo "No cached patch jobs were found." >&2
    exit 1
fi

declare -a PIDS=()
declare -a PID_LABELS=()
FAILED=0

for index in "${!PATCH_JOBS[@]}"; do
    while [[ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]]; do
        sleep "$POLL_INTERVAL"
    done

    IFS=$'\t' read -r target_name patch_index patch_row patch_col <<< "${PATCH_JOBS[$index]}"
    if [[ -z "$target_name" || -z "$patch_index" || -z "$patch_row" || -z "$patch_col" ]]; then
        echo "Malformed patch job entry: ${PATCH_JOBS[$index]}" >&2
        FAILED=1
        continue
    fi
    gpu_id="${GPU_ID_ARRAY[$((index % ${#GPU_ID_ARRAY[@]}))]}"

    QUANTIZE_CMD=(
        "$PYTHON_BIN"
        "$LM_DIR/weight_aware_quant_cached.py"
        quantize-target
        --cache_dir "$CACHE_DIR"
        --target_name "$target_name"
        --bit_width "$BIT_WIDTH"
        --group_size "$GROUP_SIZE"
        --num_steps "$NUM_STEPS"
        --rank_scale "$RANK_SCALE"
        --seed "$SEED"
        --patch_index "$patch_index"
        --workers_per_gpu "$WORKERS_PER_GPU"
        --main_gpu_id 0
    )

    if [[ -n "$SAVE_DIR" ]]; then
        QUANTIZE_CMD+=(--save_dir "$SAVE_DIR")
    fi
    if [[ "$OVERWRITE" -eq 1 ]]; then
        QUANTIZE_CMD+=(--overwrite)
    fi

    echo "Launching $target_name patch $patch_index (row $patch_row, col $patch_col) on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES="$gpu_id" "${QUANTIZE_CMD[@]}" &
    PIDS+=("$!")
    PID_LABELS+=("$target_name patch $patch_index (row $patch_row, col $patch_col) on GPU $gpu_id")
done

for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
        echo "Finished ${PID_LABELS[$index]}"
    else
        echo "Failed ${PID_LABELS[$index]}" >&2
        FAILED=1
    fi
done

exit "$FAILED"
