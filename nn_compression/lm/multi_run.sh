#!/bin/bash

NUM_GPUS=4
declare -a PIDS
GPU_INDEX=0

# ======= for gptq models ========
# MODEL_PATHS=(
#     "/work2/k-kuroki/GPTQModel/quantized_model/DeepSeek-R1-Distill-Qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B-gptq-2bit-128gs"
#     "/work2/k-kuroki/GPTQModel/quantized_model/DeepSeek-R1-Distill-Qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B-gptq-3bit-128gs"
#     "/work2/k-kuroki/GPTQModel/quantized_model/DeepSeek-R1-Distill-Qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B-gptq-4bit-128gs"
#     "/work2/k-kuroki/GPTQModel/quantized_model/Qwen2.5-0.5B/Qwen2.5-0.5B-gptq-2bit-128gs"
#     "/work2/k-kuroki/GPTQModel/quantized_model/Qwen2.5-0.5B/Qwen2.5-0.5B-gptq-3bit-128gs"
#     "/work2/k-kuroki/GPTQModel/quantized_model/Qwen2.5-0.5B/Qwen2.5-0.5B-gptq-4bit-128gs"
#     "/work2/k-kuroki/GPTQModel/quantized_model/Qwen2.5-1.5B/Qwen2.5-1.5B-gptq-2bit-128gs"
#     "/work2/k-kuroki/GPTQModel/quantized_model/Qwen2.5-1.5B/Qwen2.5-1.5B-gptq-3bit-128gs"
#     "/work2/k-kuroki/GPTQModel/quantized_model/Qwen2.5-1.5B/Qwen2.5-1.5B-gptq-4bit-128gs"
# )

# ======== for bqq models =======
MODEL_PATHS=(
    "/work2/k-kuroki/BQQLLM/quantized_model_data/DeepSeek-R1-Distill-Qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B-2bit-128gs-50000step.pth"
    "/work2/k-kuroki/BQQLLM/quantized_model_data/DeepSeek-R1-Distill-Qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B-3bit-128gs-50000step.pth"
    "/work2/k-kuroki/BQQLLM/quantized_model_data/DeepSeek-R1-Distill-Qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B-4bit-128gs-50000step.pth"
    "/work2/k-kuroki/BQQLLM/quantized_model_data/Qwen2.5-0.5B/Qwen2.5-0.5B-2bit-128gs-50000step.pth"
    "/work2/k-kuroki/BQQLLM/quantized_model_data/Qwen2.5-0.5B/Qwen2.5-0.5B-3bit-128gs-50000step.pth"
    "/work2/k-kuroki/BQQLLM/quantized_model_data/Qwen2.5-0.5B/Qwen2.5-0.5B-4bit-128gs-50000step.pth"
    "/work2/k-kuroki/BQQLLM/quantized_model_data/Qwen2.5-1.5B/Qwen2.5-1.5B-2bit-128gs-50000step.pth"
    "/work2/k-kuroki/BQQLLM/quantized_model_data/Qwen2.5-1.5B/Qwen2.5-1.5B-3bit-128gs-50000step.pth"
    "/work2/k-kuroki/BQQLLM/quantized_model_data/Qwen2.5-1.5B/Qwen2.5-1.5B-4bit-128gs-50000step.pth"
)

# MODEL_NAMESを対応する形で繰り返す
MODEL_NAMES=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-1.5B"
    "Qwen/Qwen2.5-1.5B"
    "Qwen/Qwen2.5-1.5B"
)

for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"

    DEVICE="cuda:$GPU_INDEX"
    echo "=============================="
    echo "Running Model: $MODEL_NAME | Path: $MODEL_PATH | Device: $DEVICE"
    echo "=============================="

    CUDA_VISIBLE_DEVICES=$GPU_INDEX python evaluation.py --model_path "$MODEL_PATH" --model_name "$MODEL_NAME" &

    PIDS+=($!)

    GPU_INDEX=$(( (GPU_INDEX + 1) % NUM_GPUS ))

    while [ $(jobs -r | wc -l) -ge $NUM_GPUS ]; do
        sleep 5
    done
done

wait
echo "✅ すべての評価処理が完了しました"
