# neural_network_compression

Tools for applying BQQ to pretrained neural networks.
The current LM path has three recommended accuracy/speed tiers: layerwise quantization, blockwise quantization, and optional fine-tuning.

## Quick Start

Choose the entry point by the speed/accuracy tradeoff you want:

- Fast / mild accuracy: `run_lm_layerwise_quant.sh`
- Medium speed / medium accuracy: `run_lm_blockwise_quant.sh`
- Slow / high accuracy: `run_lm_fine_tuning.sh` after a quantized model has been built

Minimal examples:

```bash
cd neural_network_compression

# Fast: independent layerwise Hessian-aware quantization.
./run_lm_layerwise_quant.sh

# Medium: blockwise quantization/tuning.
./run_lm_blockwise_quant.sh

# Slow: fine-tune an assembled quantized model.
./run_lm_fine_tuning.sh
```

`run_lm_layerwise_quant.sh` assigns one transformer block to each GPU job, collects all Hessians for that block once, then quantizes the block's Linear layers in parallel with `LAYERS_PER_GPU`.

`run_lm_blockwise_quant.sh` assigns transformer blocks to GPU jobs and performs blockwise quantization/tuning. The current default is `PROGRESSIVE=1` with `PROGRESSIVE_MODE=layer-tune`; set `PROGRESSIVE=0` to use the older layerwise-first then block-tuning flow.

`run_lm_fine_tuning.sh` starts from an assembled quantized model and performs STE fine-tuning.

Useful overrides:

```bash
MODEL_NAME=Qwen/Qwen3.5-2B BIT_WIDTH=2 GROUP_SIZE=64 ./run_lm_layerwise_quant.sh
MODEL_NAME=Qwen/Qwen3.5-2B BIT_WIDTH=2 GROUP_SIZE=64 ./run_lm_blockwise_quant.sh
MODEL_NAME=Qwen/Qwen3.5-2B MODEL_PATH=/path/to/quantized.pth ./run_lm_fine_tuning.sh

# Select GPUs and parallelism.
GPU_IDS=0,1,2,3 LAYERS_PER_GPU=4 ./run_lm_layerwise_quant.sh
GPU_IDS=0,1,2,3 BLOCKS_PER_GPU=1 ./run_lm_blockwise_quant.sh

# Process one block manually.
BLOCK_IDX=0 ./run_lm_layerwise_quant.sh
BLOCK_IDX=0 ./run_lm_blockwise_quant.sh
```

Notes:

- Default calibration dataset in the wrapper scripts: `slimpajama`
- Default assembled quantized model directory: `lm/src/quantized_models/<model>/`
- Default fine-tuned model directory: `lm/fine_tuned_models/<model>/`
- Default Hessian-aware compensation mode: `ldlq`
- Layerwise outputs are saved under `lm/src/bqq_compressed_data/<model>-<bit>bit-<gs>gs-<steps>step/`

## Standard Outputs

Typical directories:

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

## LM Workflow

### Recommended pipeline

Use one of these wrapper scripts first:

1. `run_lm_layerwise_quant.sh` for the fastest layerwise-only quantized model
2. `run_lm_blockwise_quant.sh` for blockwise quantization/tuning
3. `run_lm_fine_tuning.sh` to improve an already assembled quantized model

### Wrapper Configuration

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

`run_lm_layerwise_quant.sh` variables:

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

`run_lm_blockwise_quant.sh` variables:

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

`run_lm_fine_tuning.sh` variables:

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

Examples:

```bash
MODEL_NAME=Qwen/Qwen3.5-2B \
BIT_WIDTH=2 \
GROUP_SIZE=64 \
LAYERWISE_ANNEAL_STEPS=50000 \
LAYERS_PER_GPU=4 \
./run_lm_layerwise_quant.sh

MODEL_NAME=Qwen/Qwen3.5-2B \
BIT_WIDTH=2 \
GROUP_SIZE=64 \
BLOCKWISE_EPOCHS=1 \
PROGRESSIVE=1 \
./run_lm_blockwise_quant.sh
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
./run_lm_blockwise_quant.sh
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
    --workers_per_gpu 4 \
    --compensation_mode ldlq \
    --save_dir src/bqq_compressed_data/Qwen3.5-2B-2bit-64gs-10000step \
    --no_assemble_full_model
```

This path includes:

- Hessian-aware quantization with `ldlq`, `gptq`, or no compensation
- default `multibqq` joint bit optimization
- optional STE refinement
- block-local Hessian collection when `--block_idx` is used
- automatic full-model assembly by default unless `--no_assemble_full_model` is passed

Useful arguments:

- `--block_idx`
- `--workers_per_gpu`
- `--compensation_mode ldlq|gptq|none`
- `--scale_refine`
- `--ste_refine_steps`
- `--ste_refine_binary_lr`
- `--ste_refine_continuous_lr`
- `--refine_coeffs_only`
- `--fix_theta`
- `--fix_beta`
- `--use_multibqq`
- `--no_use_multibqq`
- `--no_assemble_full_model`

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
    --layerwise_dir src/bqq_compressed_data/Qwen3.5-2B-2bit-64gs-10000step \
    --save_dir blockwise_output/Qwen3.5-2B \
    --progressive \
    --progressive_mode layer-tune \
    --no_io_cache \
    --no_scale_refine \
    --compensation_mode ldlq
```

Progressive layer-tune mode does this:

1. Collect/cache the block inputs needed for sequential layer quantization
2. Quantize Linear layers in block order with Hessian-aware BQQ
3. Optionally run STE refinement for each layer
4. Build the quantized block and tune block output MSE
5. Save `block_<idx>.pth`
6. Optionally assemble the full quantized model

With `PROGRESSIVE=0` in the wrapper, the flow becomes:

1. Run `layerwise_quant.py --block_idx <i>` first
2. Load those layerwise outputs into `blockwise_quant.py`
3. Tune the block-level output MSE

Useful arguments:

- `--progressive`
- `--progressive_mode layer-tune|closed-form-layer|patch`
- `--compensation_mode ldlq|gptq|none`
- `--use_multibqq`
- `--no_use_multibqq`
- `--no_io_cache`
- `--no_scale_refine`
- `--ste_refine_steps`
- `--ste_refine_binary_lr`
- `--ste_refine_continuous_lr`
- `--optimizer adamw|sgd`
- `--momentum`
- `--binary_lr`
- `--continuous_lr`
- `--refine_coeffs_only`
- `--fix_theta`
- `--fix_beta`
- `--no_assemble_full_model`

### Fine-tuning

Standard fine-tuning is wrapped by `run_lm_fine_tuning.sh`.

```bash
cd neural_network_compression

./run_lm_fine_tuning.sh
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
