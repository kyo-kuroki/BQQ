# Experimental vLLM BQQ Integration

This directory contains the initial bridge for serving packed BQQ checkpoints
with vLLM.

## Current Status

Implemented:

- `export_vllm_bqq.py` exports a packed BQQ `.pth` checkpoint to a vLLM-style
  model directory with `model.safetensors`, `bqq_config.json`, and
  `quantization_config.json`.
- `vllm_quantization.py` defines and registers a custom vLLM quantization
  method named `bqq`.
- vLLM fused modules are mapped to the corresponding HF BQQ parts:
  `q/k/v -> qkv_proj`, `gate/up -> gate_up_proj`,
  `in_proj_qkv/in_proj_z -> in_proj_qkvz`, and
  `in_proj_b/in_proj_a -> in_proj_ba`.
- `BQQLinearMethod` holds one or more `PackedBinaryQuadratic` runtimes per
  vLLM Linear and concatenates fused outputs in vLLM order.

Current limitations:

- Single GPU only.
- No tensor parallel sharding.
- GPU validation requires a CUDA-compatible PyTorch/vLLM install.  On the
  current cluster environment, vLLM installed PyTorch `2.11.0+cu130`, while the
  visible NVIDIA driver reports CUDA driver `12.2`; `torch.cuda.is_available()`
  is therefore false until the driver or PyTorch/vLLM wheel is changed.

## Export

```bash
cd /work2/k-kuroki/BQQ/neural_network_compression

PYTHONPATH=/work2/k-kuroki/BQQ \
/artic/k-kuroki/.conda/envs/py311/bin/python bqqkernel/export_vllm_bqq.py \
  --model-name Qwen/Qwen3.5-4B \
  --model-path lm/fine_tuned_models/Qwen3.5-4B/Qwen3.5-4B-1bit-64gs-blockwise-finetuned-packed.pth \
  --output-dir lm/fine_tuned_models/Qwen3.5-4B-vllm-bqq
```

For a quick metadata-only smoke test:

```bash
PYTHONPATH=/work2/k-kuroki/BQQ \
/artic/k-kuroki/.conda/envs/py311/bin/python bqqkernel/export_vllm_bqq.py \
  --model-name Qwen/Qwen3.5-4B \
  --model-path lm/fine_tuned_models/Qwen3.5-4B/Qwen3.5-4B-1bit-64gs-blockwise-finetuned-packed.pth \
  --output-dir /tmp/bqq_vllm_export_smoke \
  --metadata-only
```

## Registering in vLLM

Inside the vLLM runtime, import the plugin before loading the model:

```python
import neural_network_compression.bqqkernel.vllm_quantization
```

On this cluster, use the conda `libstdc++` when importing vLLM:

```bash
LD_LIBRARY_PATH=/artic/k-kuroki/.conda/envs/py311/lib:$LD_LIBRARY_PATH \
PYTHONPATH=/work2/k-kuroki/BQQ \
/artic/k-kuroki/.conda/envs/py311/bin/python -c \
  "import neural_network_compression.bqqkernel.vllm_quantization"
```

The import registers `quant_method="bqq"` via vLLM's
`register_quantization_config` hook.

The exported directory contains:

```json
{
  "quant_method": "bqq",
  "format": "packed_binary_quadratic",
  "version": 1,
  "bqq_config": "bqq_config.json"
}
```
