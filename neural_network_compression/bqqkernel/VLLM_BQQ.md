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
- The initial `BQQLinearMethod` delegates linear application to
  `PackedBinaryQuadratic.forward`, so it can call the existing fused BQQ CUDA
  decode kernel.

Current limitations:

- Single GPU only.
- No tensor parallel sharding.
- Only layer prefixes matching the original HF module names are supported.
- vLLM fused modules such as `qkv_proj` and `gate_up_proj` still need explicit
  mapping from separate HF modules (`q_proj`, `k_proj`, `v_proj`,
  `gate_proj`, `up_proj`).

The fused-module mapping is the next required step for Qwen-style models in
stock vLLM.

## Export

```bash
cd /work2/k-kuroki/BQQ/neural_network_compression

PYTHONPATH=/work2/k-kuroki/BQQ \
/artic/k-kuroki/.conda/envs/py311/bin/python bqqkernel/export_vllm_bqq.py \
  --model-name Qwen/Qwen3.5-4B \
  --model-path lm/fine_tuned_models/Qwen3.5-4B/Qwen3.5-4B-1bit-64gs-blockwise-finetuned-packed.pth \
  --output-dir lm/fine_tuned_models/Qwen3.5-4B-vllm-bqq
```

## Registering in vLLM

Inside the vLLM runtime, import the plugin before loading the model:

```python
import neural_network_compression.bqqkernel.vllm_quantization
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

## Next Step

Stock vLLM Qwen models fuse several projections:

- HF `q_proj`, `k_proj`, `v_proj` -> vLLM `qkv_proj`
- HF `gate_proj`, `up_proj` -> vLLM `gate_up_proj`

BQQ stores each projection as an independent packed low-rank binary quadratic
module.  The next implementation step is a fused BQQ method that holds multiple
packed BQQ runtimes and concatenates their outputs to match vLLM fused linear
layers.
