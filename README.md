# Binary Quadratic Quantization (BQQ)

Binary Quadratic Quantization represents each weight matrix as a sum of **binary outer-product terms**:

$$W \approx \sum_{b=1}^{B} \bigl( a_b^{(0)}\, y_b z_b^\top + a_b^{(1)}\, y_b \mathbf{1}^\top + a_b^{(2)}\, \mathbf{1} z_b^\top + a_b^{(3)} \bigr)$$

where $y_b, z_b \in \{0,1\}^n$ and the scalar coefficients $a_b$ are jointly optimised by simulated annealing.
Each bit layer $b$ minimises the residual left by the previous layers, so **bit depth can be extended incrementally** without re-running earlier layers.

## Repository structure

```
BQQ/
├── quantizer.py                         # Core BQQ algorithms
├── matrix_compression/                  # Standalone matrix compression experiments
└── neural_network_compression/
    ├── lm/                              # Language model quantization
    │   ├── layerwise_quant.py           # Hessian-aware layerwise quantization
    │   ├── blockwise_quant.py           # Blockwise/progressive quantization and tuning
    │   ├── fine_tuning.py               # STE fine-tuning for assembled BQQ models
    │   ├── src/build_bqq_model.py       # Replace Linear→BinaryQuadratic, build model from patches or blocks
    │   ├── scale_refine_bqq.py          # Post-quantization scale refinement
    │   └── src/evaluation.py            # PPL / downstream evaluation
    ├── run_layerwise_quant.sh           # Fast layerwise wrapper
    ├── run_blockwise_quant.sh           # Medium blockwise wrapper
    ├── run_fine_tuning.sh               # Slow/high-accuracy fine-tuning wrapper
    └── cv/                              # Vision model quantization
```

## Quick Start (Language Model Quantization)

Choose the entry point by the speed/accuracy tradeoff you want:

- Fast / mild accuracy: `neural_network_compression/run_layerwise_quant.sh`
- Medium speed / medium accuracy: `neural_network_compression/run_blockwise_quant.sh`
- Slow / high accuracy: `neural_network_compression/run_fine_tuning.sh` after a quantized model has been built

Minimal examples:

```bash
cd neural_network_compression

# Fast: independent layerwise Hessian-aware quantization.
./run_layerwise_quant.sh

# Medium: blockwise quantization/tuning.
./run_blockwise_quant.sh

# Slow: fine-tune an assembled quantized model.
./run_fine_tuning.sh
```

`run_layerwise_quant.sh` assigns one transformer block to each GPU job, collects all Hessians for that block once, then quantizes the block's Linear layers in parallel with `LAYERS_PER_GPU`.

`run_blockwise_quant.sh` assigns transformer blocks to GPU jobs and performs blockwise quantization/tuning. The current default is `PROGRESSIVE=1` with `PROGRESSIVE_MODE=layer-tune`; set `PROGRESSIVE=0` to use the older layerwise-first then block-tuning flow.

`run_fine_tuning.sh` starts from an assembled quantized model and performs STE fine-tuning.

Useful overrides:

```bash
MODEL_NAME=Qwen/Qwen3.5-2B BIT_WIDTH=2 GROUP_SIZE=64 ./run_layerwise_quant.sh
MODEL_NAME=Qwen/Qwen3.5-2B BIT_WIDTH=2 GROUP_SIZE=64 ./run_blockwise_quant.sh
MODEL_NAME=Qwen/Qwen3.5-2B MODEL_PATH=/path/to/quantized.pth ./run_fine_tuning.sh

# Select GPUs and parallelism.
GPU_IDS=0,1,2,3 LAYERS_PER_GPU=4 ./run_layerwise_quant.sh
GPU_IDS=0,1,2,3 BLOCKS_PER_GPU=1 ./run_blockwise_quant.sh

# Process one block manually.
BLOCK_IDX=0 ./run_layerwise_quant.sh
BLOCK_IDX=0 ./run_blockwise_quant.sh
```

Notes:

- Default calibration dataset in the wrapper scripts: `slimpajama`
- Default assembled quantized model directory: `neural_network_compression/lm/src/quantized_models/<model>/`
- Default fine-tuned model directory: `neural_network_compression/lm/fine_tuned_models/<model>/`
- Default Hessian-aware compensation mode: `ldlq`
- Layerwise outputs are saved under `neural_network_compression/lm/src/bqq_compressed_data/<model>-<bit>bit-<gs>gs-<steps>step/`

## Standard Outputs

Typical LM directories:

```text
neural_network_compression/
├── lm/
│   ├── src/
│   │   ├── bqq_compressed_data/<model>-<bit>bit-<gs>gs-<steps>step/
│   │   │   └── _consolidated/<full_layer_name>.pth
│   │   └── quantized_models/<model>/
│   │       ├── <model>-<bit>bit-<gs>gs.pth
│   │       └── <model>-<bit>bit-<gs>gs-blockwise.pth
│   ├── blockwise_output/<model>/
│   │   ├── block_0.pth
│   │   ├── block_1.pth
│   │   └── ...
│   └── fine_tuned_models/<model>/
│       └── <model>-...-finetuned.pth
└── cv/
```

## LM Wrapper Configuration

Common environment variables:

- `MODEL_NAME`: Hugging Face model id
- `BIT_WIDTH`: BQQ bit width
- `GROUP_SIZE`: column group / patch width
- `DATASET`: calibration dataset, one of `wikitext2`, `ptb`, `c4`, `slimpajama`
- `NSAMPLES`: number of calibration samples
- `SEQLEN`: calibration sequence length
- `SEED`: random seed
- `GPU_IDS`: comma-separated GPU ids; defaults to all visible GPUs
- `USE_MULTIBQQ`: `1` uses joint multi-bit BQQ optimization, `0` disables it
- `COMPENSATION_MODE`: `ldlq`, `gptq`, or `none`; default is `ldlq`
- `NO_SCALE_REFINE`: `1` disables scale refinement, `0` enables it when the wrapper supports it

`run_layerwise_quant.sh` variables:

- `BLOCK_IDX`: `all` or a single transformer block index
- `LAYERS_PER_GPU`: number of layer quantization workers inside each block job
- `LAYERWISE_ANNEAL_STEPS`: BQQ annealing steps
- `LAYERWISE_STE_STEPS`: optional STE refinement steps after layerwise BQQ
- `LAYERWISE_STE_BINARY_LR`: STE learning rate for binary factors
- `LAYERWISE_STE_CONTINUOUS_LR`: STE learning rate for continuous parameters
- `LAYERWISE_ROW_GROUP_BATCH_SIZE`: optional row-group minibatch size during STE refinement
- `LAYERWISE_FIX_THETA`: `1` freezes thresholds
- `LAYERWISE_FIX_BETA`: `1` freezes sigmoid temperature
- `LAYERWISE_DIR`: output directory for per-layer BQQ tensors
- `ASSEMBLED_OUTPUT_DIR`: output directory for the assembled model

`run_blockwise_quant.sh` variables:

- `BLOCK_IDX`: `all` or a single transformer block index
- `BLOCKS_PER_GPU`: number of block jobs per GPU
- `PROGRESSIVE`: `1` uses progressive blockwise quantization; `0` runs separate layerwise quantization first
- `PROGRESSIVE_MODE`: `layer-tune`, `closed-form-layer`, or `patch`
- `LAYERWISE_WORKERS_PER_GPU`: layerwise worker count used when `PROGRESSIVE=0`
- `BLOCKWISE_EPOCHS`: block tuning epochs
- `BLOCKWISE_OPTIMIZER`: `adamw`, `adam`, or `sgd`
- `BLOCKWISE_LR`: base blockwise learning rate
- `BLOCKWISE_BINARY_LR`: blockwise learning rate for binary factors
- `BLOCKWISE_CONTINUOUS_LR`: blockwise learning rate for continuous parameters
- `BLOCKWISE_TUNE_BATCH_SIZE`: block tuning minibatch size
- `STE_REFINE`: `1` enables extra STE refinement inside blockwise quantization
- `STE_REFINE_STEPS`: extra STE refinement steps when `STE_REFINE=1`
- `NO_IO_CACHE`: `1` avoids writing block I/O caches to disk
- `BLOCKWISE_SAVE_DIR`: output directory for `block_<idx>.pth`

`run_fine_tuning.sh` variables:

- `MODEL_PATH`: assembled quantized model path
- `FINETUNE_EPOCHS`: number of fine-tuning epochs
- `FINETUNE_LR`: base learning rate
- `FINETUNE_BINARY_LR`: learning rate for trainable binary factors
- `FINETUNE_CONTINUOUS_LR`: learning rate for continuous parameters
- `GRADIENT_ACCUMULATION_STEPS`: gradient accumulation
- `MAX_SEQ_LENGTH`: SFT sequence length
- `FINETUNE_FIX_THETA`: `1` freezes thresholds
- `FINETUNE_FIX_BETA`: `1` freezes sigmoid temperature
- `OUTPUT_DIR`: fine-tuned model output directory

## Evaluation

Perplexity evaluation example:

```bash
cd neural_network_compression

python lm/src/evaluation.py \
  --model_name Qwen/Qwen3.5-2B \
  --model_path lm/src/quantized_models/Qwen3.5-2B/Qwen3.5-2B-2bit-64gs.pth \
  --device cuda:0 \
  --seq_len 2048
```

Results are written under `neural_network_compression/lm/src/results/`.

## Inference kernels

BQQ models use `PackedBinaryQuadratic` layers where Y, Z are stored as packed uint8 (8x smaller than bool). Three inference backends are available:

| Backend | Kernel | Description |
|---------|--------|-------------|
| **CUDA** (default) | `bqq_cuda.cu` mode=3 | Multi-warp shuffle: 4 warps split `col_width`, conditional-add inner loop. Compiled on first import via `torch.utils.cpp_extension`. |
| **Triton** | `bqq_triton_kernel.py` v1/v2 | Fused kernel with `t_aug = a*t + b*xsum` optimisation. v2 parallelises `bit_width` via 2D tensor ops. |
| **W-reconstruction** | `bqq_modules.py` | Unpack Y/Z → cuBLAS matmul for W, then `X @ W.T`. Fallback for large batch. |

Auto-selection in `PackedBinaryQuadratic.forward()`:
- `use_zy_x_kernel = True`: tries CUDA → Triton → W-reconstruction
- seq_len > `zy_x_recon_threshold` (default 32): W-reconstruction + cuBLAS

### CUDA kernel modes

```python
from bqq_cuda_ext import cuda_bqq_forward

# mode=0 (auto=5), mode=1 (1-warp shuffle), mode=3 (multi-warp shuffle),
# mode=5 (grid-split + uint32), mode=6 (SRAM-tiled)
out = cuda_bqq_forward(Y_packed, Z_packed, X, a, b, c, d, mode=5)
```

### Performance (2048x2048, 4-bit, gs=32, RTX 4500 Ada)

| seq_len | FP16 cuBLAS | Best BQQ kernel | Kernel | FP16 ratio |
|---------|-------------|-----------------|--------|------------|
| 1       | 33 us       | 47 us           | CUDA v5 | 1.4x      |
| 4       | 36 us       | 169 us          | CUDA v5 | 4.7x      |
| 16      | 36 us       | 682 us          | CUDA v5 | 19x       |
| 64      | 35 us       | 1,621 us        | W-recon | 46x       |
| 256     | 40 us       | 1,715 us        | W-recon | 43x       |

At seq_len=1 (autoregressive decode), BQQ is only 1.4x slower than FP16.
At seq_len≥32, W-reconstruction + cuBLAS Tensor Core becomes faster.
BQQ's primary value is **2-4x model size reduction**, enabling larger models on smaller GPUs.

## `quantizer.py`

| Class | Description |
|-------|-------------|
| `BinaryQuadraticQuantization` | Main BQQ class for all LM and CV workflows. `binarize_scaling=True` for V1 mode (binarized scaling), `False` (default) for V2 mode (continuous scaling). |

`torch.compile` defaults to `mode="reduce-overhead"`. Pass `compile_mode="max-autotune"` for autotuning (slower compilation but potentially faster kernels).
