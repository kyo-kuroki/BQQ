# neural_network_compression

Tools for applying BQQ to pretrained neural networks.
The current LM path is centered on `layerwise -> blockwise -> fine_tuning`, with wrapper scripts for the standard flow.

## Quick Start

The standard LM workflow is:

1. `run_blockwise_quant.sh`
2. `run_fine_tuning.sh`

Minimal example:

```bash
cd neural_network_compression

./run_blockwise_quant.sh
./run_fine_tuning.sh
```

By default, `run_blockwise_quant.sh` now quantizes all transformer blocks in order.
It runs block-level layerwise quantization first, then blockwise tuning, and assembles the full quantized model after the last block.

Useful overrides:

```bash
MODEL_NAME=Qwen/Qwen3.5-2B BIT_WIDTH=2 GROUP_SIZE=64 ./run_blockwise_quant.sh
MODEL_NAME=Qwen/Qwen3.5-2B BLOCK_IDX=0 ./run_blockwise_quant.sh
MODEL_NAME=Qwen/Qwen3.5-2B ./run_fine_tuning.sh
```

Notes:

- Default calibration dataset in the wrapper scripts: `slimpajama`
- Default assembled quantized model directory: `lm/src/quantized_models/<model>/`
- Default fine-tuned model directory: `lm/fine_tuned_models/<model>/`

## Standard Outputs

Typical directories:

```text
neural_network_compression/
├── lm/
│   ├── src/
│   │   ├── bqq_compressed_data/<model>-<gs>gs-<steps>step>/
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

## LM Workflow

### Recommended pipeline

The recommended LM pipeline is:

1. `blockwise_quant.py`
2. `fine_tuning.py`

In practice, the wrapper `run_blockwise_quant.sh` is the normal entry point.
Its default behavior is:

1. Determine the number of transformer blocks
2. Run `layerwise_quant.py --block_idx <i>` for each block
3. Run `blockwise_quant.py --block_idx <i>` for each block
4. Save `block_<i>.pth` into `lm/blockwise_output/<model>/`
5. Assemble the full blockwise model once, after the final block

You only need `BLOCK_IDX=<n>` when you want to process a single block manually.

### Wrapper configuration

`run_blockwise_quant.sh` exposes the main knobs as environment variables.
Important ones are:

- `MODEL_NAME`
- `BLOCK_IDX`
  Default: `all`
- `BIT_WIDTH`
- `GROUP_SIZE`
- `LAYERWISE_ANNEAL_STEPS`
- `LAYERWISE_STE_STEPS`
- `BLOCKWISE_EPOCHS`
- `DATASET`
- `NSAMPLES`
- `SEQLEN`

Example:

```bash
MODEL_NAME=Qwen/Qwen3.5-2B \
BIT_WIDTH=2 \
GROUP_SIZE=64 \
LAYERWISE_ANNEAL_STEPS=10000 \
BLOCKWISE_EPOCHS=3 \
./run_blockwise_quant.sh
```

### Separate learning rates for binary and continuous parameters

The current implementation can use different learning rates for binary variables and continuous variables.

Layerwise STE refinement:

- base LR: `LAYERWISE_STE_LR`
- binary factor LR: `LAYERWISE_STE_BINARY_LR`
- continuous parameter LR: `LAYERWISE_STE_CONTINUOUS_LR`

Blockwise tuning:

- base LR: `BLOCKWISE_LR`
- binary factor LR: `BLOCKWISE_BINARY_LR`
- continuous parameter LR: `BLOCKWISE_CONTINUOUS_LR`

Fine-tuning:

- base LR: `FINETUNE_LR`
- binary factor LR: `FINETUNE_BINARY_LR`
- continuous parameter LR: `FINETUNE_CONTINUOUS_LR`

Here, binary parameters mean STE-trainable `Y/Z` factors.
Continuous parameters include `theta`, `a,b,c,d`, bias, and ordinary floating-point model parameters.

Example:

```bash
LAYERWISE_STE_STEPS=200 \
LAYERWISE_STE_BINARY_LR=3e-4 \
LAYERWISE_STE_CONTINUOUS_LR=1e-3 \
BLOCKWISE_OPTIMIZER=sgd \
BLOCKWISE_MOMENTUM=0.9 \
BLOCKWISE_BINARY_LR=3e-4 \
BLOCKWISE_CONTINUOUS_LR=1e-3 \
./run_blockwise_quant.sh
```

### Layerwise quantization

You can still run layerwise quantization directly when you want only layerwise results.

```bash
cd neural_network_compression/lm

python layerwise_quant.py \
    --model_name Qwen/Qwen3.5-2B \
    --block_idx 0 \
    --bit_width 2 \
    --group_size 64 \
    --num_steps 10000 \
    --dataset slimpajama \
    --nsamples 1024 \
    --seqlen 2048 \
    --save_dir src/bqq_compressed_data/Qwen3.5-2B-64gs-10000step
```

This path includes:

- intra-layer Hessian-aware quantization
- default `multibqq` joint bit optimization
- optional STE refinement
- automatic full-model assembly by default

Useful arguments:

- `--ste_refine_steps`
- `--ste_refine_binary_lr`
- `--ste_refine_continuous_lr`
- `--refine_coeffs_only`
- `--fix_theta`
- `--no_use_multibqq`

### Blockwise tuning

Direct use of `blockwise_quant.py` is still available.

```bash
cd neural_network_compression/lm

python blockwise_quant.py \
    --model_name Qwen/Qwen3.5-2B \
    --block_idx 0 \
    --bit_width 2 \
    --group_size 64 \
    --num_steps 10000 \
    --dataset slimpajama \
    --nsamples 1024 \
    --seqlen 2048 \
    --epochs 5 \
    --optimizer sgd \
    --momentum 0.9 \
    --lr 1e-3 \
    --binary_lr 3e-4 \
    --continuous_lr 1e-3 \
    --device cuda:0 \
    --save_dir blockwise_output/Qwen3.5-2B
```

Standard mode does this:

1. Cache block input/output activations
2. Ensure block-local layerwise outputs exist
3. Load the block with STE-trainable BQQ modules
4. Optimize block output MSE
5. Save `block_<idx>.pth`
6. Optionally assemble the full quantized model

Useful arguments:

- `--optimizer adamw|sgd`
- `--momentum`
- `--binary_lr`
- `--continuous_lr`
- `--refine_coeffs_only`
- `--fix_theta`
- `--no_assemble_full_model`

`--progressive` remains available for the older progressive patch-wise quantization path, but it is not the recommended default flow.

### Fine-tuning

Standard fine-tuning is wrapped by `run_fine_tuning.sh`.

```bash
cd neural_network_compression

./run_fine_tuning.sh
```

Direct use:

```bash
cd neural_network_compression/lm

python fine_tuning.py \
    --model_name Qwen/Qwen3.5-2B \
    --model_path src/quantized_models/Qwen3.5-2B/Qwen3.5-2B-2bit-64gs-blockwise.pth \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --binary_learning_rate 5e-6 \
    --continuous_learning_rate 2e-5
```

By default, fine-tuning optimizes:

- STE binary factors `Y/Z`
- thresholds `theta`
- coefficients `a,b,c,d`
- bias
- other ordinary trainable model parameters

Coefficient-only mode:

```bash
python fine_tuning.py \
    --model_name Qwen/Qwen3.5-2B \
    --model_path /path/to/model.pth \
    --refine_coeffs_only \
    --fix_theta
```

KL distillation is also supported:

```bash
python fine_tuning.py \
    --model_name Qwen/Qwen3.5-2B \
    --model_path /path/to/model.pth \
    --teacher_model_name Qwen/Qwen3.5-2B \
    --kl_alpha 1.0 \
    --kl_temperature 2.0
```

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

Results are written under `lm/src/results/`.

## Legacy / Auxiliary LM Tools

The following tools remain useful for experiments, but they are no longer the main recommended LM path.

- `lm/weight_aware_quant_cached.py`
- `lm/weight_aware_quant.py`
- `lm/scale_refine_bqq.py`

`extend-target` style bit-depth extension is now best viewed as an auxiliary workflow for comparisons or reuse of older low-bit results.
In the current implementation, `bit_width` is optimized directly in layerwise, blockwise, and fine-tuning stages.

## CV Workflow

```bash
cd neural_network_compression/cv/bqq

python weight_aware_quant_cached.py prepare-cache --model_name deit-s

./weight_aware_quant_cached_parallel.sh \
    --model_name deit-s \
    --gpu_ids 0,1,2,3 \
    --finalize \
    --evaluate \
    --data_path /path/to/imagenet
```

`--finalize` rebuilds the full BQQ model from patch files after quantization completes.
`--evaluate` runs ImageNet evaluation and writes results to `results/`.
