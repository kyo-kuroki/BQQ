#!/usr/bin/env bash
set -euo pipefail

# End-to-end (e2e) progressive BQQ quantization with KL fine-tuning.
# Default flow: quantize one transformer block at a time (all its Linears in
# parallel across every GPU), replace it with frozen BQQ modules, then
# fine-tune the remaining unquantized layers to minimize the KL divergence
# against the original fp model, and repeat until the whole model is quantized.
# Set QUANT_UNIT=layerwise to quantize one Linear layer per iteration instead.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/lm"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}" # Example Options: "Qwen/Qwen3.5-4B", "meta-llama/Llama-3.1-8B"
QUANT_UNIT="${QUANT_UNIT:-blockwise}"       # Options: "blockwise" (one transformer block per iteration), "layerwise" (one Linear per iteration)
BIT_WIDTH="${BIT_WIDTH:-2}"
GROUP_SIZE="${GROUP_SIZE:-64}"
NUM_STEPS="${NUM_STEPS:-20000}"
RANK_SCALE="${RANK_SCALE:-1.0}"

DATASET="${DATASET:-slimpajama}"
NSAMPLES="${NSAMPLES:-256}"
SEQLEN="${SEQLEN:-2048}"
SEED="${SEED:-0}"

USE_MULTIBQQ="${USE_MULTIBQQ:-1}"
COMPENSATION_MODE="${COMPENSATION_MODE:-ldlq}"
NO_SCALE_REFINE="${NO_SCALE_REFINE:-1}"
BQQ_OPT_MODE="${BQQ_OPT_MODE:-activation-aware}" # Options: "plain", "activation-aware" (full-matrix Hessian / fullchol)
DIAG_POWER="${DIAG_POWER:-0.75}"                 # Metric tempering alpha: quantize with H^alpha (alpha<1 flattens the spectrum)
TRANSFORM="${TRANSFORM:-rht}"                    # Options: "none", "rht", "ht", "dct"
LDLQ_ACT_ORDER="${LDLQ_ACT_ORDER:-0}"
LDLQ_ACT_ORDER_SCORE="${LDLQ_ACT_ORDER_SCORE:-maxdiag}"
RANK_ALLOC_MODE="${RANK_ALLOC_MODE:-none}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"

# KL fine-tuning between units (pure KL against the original model by default)
NO_FINETUNE="${NO_FINETUNE:-0}"
FT_EPOCHS="${FT_EPOCHS:-5}"
FT_STEPS="${FT_STEPS:-0}"          # Cap on optimizer steps per unit (0 = no cap)
FT_LR="${FT_LR:-1e-5}"
FT_WEIGHT_DECAY="${FT_WEIGHT_DECAY:-0.0}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
KL_TEMPERATURE="${KL_TEMPERATURE:-1.0}"
KL_ALPHA="${KL_ALPHA:-1.0}"
CE_ALPHA="${CE_ALPHA:-0.0}"
FT_LOG_INTERVAL="${FT_LOG_INTERVAL:-10}"
TRAIN_EMBEDDINGS="${TRAIN_EMBEDDINGS:-0}"
TRAIN_QUANTIZED_CONTINUOUS="${TRAIN_QUANTIZED_CONTINUOUS:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
EVAL_BATCHES="${EVAL_BATCHES:-4}"

GPU_IDS="${GPU_IDS:-}" # Comma-separated GPU ids (default: all visible GPUs). Student/teacher are pipeline-sharded across them.
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
FRESH="${FRESH:-0}"
WORK_DIR="${WORK_DIR:-}"      # Default: lm/e2e_output/{model}/{bit}bit-{gs}gs-{unit}
OUTPUT_PATH="${OUTPUT_PATH:-}" # Default: lm/src/quantized_models/{model}/{model}-{bit}bit-{gs}gs-e2e.pth

cmd=(
  python "${LM_DIR}/e2e_quantize.py"
  --model_name "${MODEL_NAME}"
  --quant_unit "${QUANT_UNIT}"
  --bit_width "${BIT_WIDTH}"
  --group_size "${GROUP_SIZE}"
  --num_steps "${NUM_STEPS}"
  --rank_scale "${RANK_SCALE}"
  --seed "${SEED}"
  --dataset "${DATASET}"
  --nsamples "${NSAMPLES}"
  --seqlen "${SEQLEN}"
  --compensation_mode "${COMPENSATION_MODE}"
  --bqq_opt_mode "${BQQ_OPT_MODE}"
  --diag_power "${DIAG_POWER}"
  --transform "${TRANSFORM}"
  --ldlq_act_order_score "${LDLQ_ACT_ORDER_SCORE}"
  --rank_alloc_mode "${RANK_ALLOC_MODE}"
  --workers_per_gpu "${WORKERS_PER_GPU}"
  --ft_epochs "${FT_EPOCHS}"
  --ft_steps "${FT_STEPS}"
  --ft_lr "${FT_LR}"
  --ft_weight_decay "${FT_WEIGHT_DECAY}"
  --grad_accum "${GRAD_ACCUM}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --kl_temperature "${KL_TEMPERATURE}"
  --kl_alpha "${KL_ALPHA}"
  --ce_alpha "${CE_ALPHA}"
  --ft_log_interval "${FT_LOG_INTERVAL}"
  --eval_batches "${EVAL_BATCHES}"
  --checkpoint_every "${CHECKPOINT_EVERY}"
)

if [[ -n "${GPU_IDS}" ]]; then
  cmd+=(--gpu_ids "${GPU_IDS}")
fi

if [[ "${USE_MULTIBQQ}" == "1" ]]; then
  cmd+=(--use_multibqq)
else
  cmd+=(--no_use_multibqq)
fi
if [[ "${NO_SCALE_REFINE}" == "1" ]]; then
  cmd+=(--no_scale_refine)
fi
if [[ "${LDLQ_ACT_ORDER}" == "1" ]]; then
  cmd+=(--ldlq_act_order)
fi
if [[ "${NO_FINETUNE}" == "1" ]]; then
  cmd+=(--no_finetune)
fi
if [[ "${TRAIN_EMBEDDINGS}" == "1" ]]; then
  cmd+=(--train_embeddings)
fi
if [[ "${TRAIN_QUANTIZED_CONTINUOUS}" == "1" ]]; then
  cmd+=(--train_quantized_continuous)
fi
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  cmd+=(--gradient_checkpointing)
fi
if [[ "${FRESH}" == "1" ]]; then
  cmd+=(--fresh)
fi
if [[ -n "${WORK_DIR}" ]]; then
  cmd+=(--work_dir "${WORK_DIR}")
fi
if [[ -n "${OUTPUT_PATH}" ]]; then
  cmd+=(--output_path "${OUTPUT_PATH}")
fi
cmd+=("$@")

echo "[e2e] ${cmd[*]}"
"${cmd[@]}"
