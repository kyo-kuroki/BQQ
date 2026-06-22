# neural_network_compression

Tools for applying BQQ to pretrained neural networks.
Supports weight-aware quantization, incremental bit-depth extension, model reconstruction, evaluation, and optional fine-tuning.

---

## Contents

| File | Description |
|------|-------------|
| `lm/weight_aware_quant_cached.py` | Cache-first weight quantization for LMs (main entry point) |
| `lm/weight_aware_quant.py` | Original (non-cached) weight-aware quantization |
| `lm/scripts/qsub_submit_qwen35.sh` | Submit N-bit quantization array jobs on TSUBAME (Qwen3.5-2B/4B/9B) |
| `lm/scripts/qsub_patch_array_job.sh` | SGE array job body for `quantize-target` (1 task = 1 weight tensor) |
| `lm/scripts/qsub_extend_array_job.sh` | SGE array job body for `extend-target` (1 task = 1 weight tensor) |
| `lm/blockwise_quant.py` | Block-wise quantization with block output error optimization |
| `lm/build_bqq_model.py` | Replace Linear→BinaryQuadratic, build model from patches or blocks |
| `lm/scale_refine_bqq.py` | Hessian-based scale factor refinement (post-quantization) |
| `lm/fine_tuning.py` | Fine-tuning / KL distillation on a quantized model |
| `lm/evaluation.py` | Perplexity and task evaluation |

---

## LM workflow

### Recommended pipeline

The default LM pipeline is now:

1. `blockwise_quant.py`
2. `fine_tuning.py`

In standard mode, `blockwise_quant.py` uses layerwise BQQ results as initialization, then optimizes each transformer block by block-output MSE. If the required layerwise results for the target block do not exist yet, it automatically runs the corresponding block-level layerwise quantization internally and then continues to block tuning.

So in normal use, you usually only need to run:

```bash
cd neural_network_compression/lm

python blockwise_quant.py \
    --model_name Qwen/Qwen3-2B \
    --block_idx 0 \
    --bit_width 2 \
    --group_size 64 \
    --num_steps 10000 \
    --dataset c4 \
    --nsamples 128 \
    --seqlen 2048 \
    --epochs 5 \
    --lr 1e-5 \
    --device cuda:0 \
    --save_dir blockwise_output/Qwen3-2B
```

This does the following:

1. Cache block input/output activations for the target block
2. Check whether layerwise quantization outputs for that block already exist
3. If missing, run internal `layerwise_quantize_block(...)`
4. Load the block as STE-trainable BQQ layers
5. Optimize the block by minimizing block output MSE
6. Save `block_{idx}.pth`
7. Assemble the current full quantized model automatically

Default outputs:

- Layerwise patch data: `lm/src/bqq_compressed_data/<model>-<gs>gs-<steps>step/`
- Blockwise block file: `blockwise_output/Qwen3-2B/block_0.pth`
- Assembled full model: `lm/src/quantized_models/Qwen3-2B/Qwen3-2B-2bit-64gs-blockwise.pth`

Use `--no_assemble_full_model` if you want to skip the final reconstruction, or `--assembled_output_dir` to change the full-model output directory.

The default mode is the layerwise-initialize + blockwise-tune mode.
Use `--progressive` only if you explicitly want progressive patch-wise quantization.

### Fine-tuning after blockwise quantization

`blockwise_quant.py` now assembles the full model automatically by default, so the usual next step is just fine-tuning:

```bash
python fine_tuning.py \
    --model_name Qwen/Qwen3-2B \
    --model_path src/quantized_models/Qwen3-2B/Qwen3-2B-2bit-64gs-blockwise.pth \
    --num_train_epochs 3 \
    --learning_rate 2e-5
```

If you omit `--model_path`, `fine_tuning.py` also looks in `src/quantized_models/<model>/` by default.

By default, `fine_tuning.py` converts BQQ layers to STE-trainable modules and optimizes:

- binary factors `Y/Z`
- thresholds `theta`
- coefficients `a,b,c,d`
- bias

If you want coefficient-only tuning:

```bash
python fine_tuning.py \
    --model_name Qwen/Qwen3-2B \
    --model_path /path/to/model.pth \
    --refine_coeffs_only \
    --fix_theta
```

### Blockwise quantization modes

#### Standard mode

Standard mode is recommended.

```bash
python blockwise_quant.py \
    --model_name Qwen/Qwen3-2B \
    --block_idx 0 \
    --save_dir blockwise_output/Qwen3-2B
```

Options:

- `--layerwise_dir`: use an existing layerwise output directory explicitly
- `--refine_coeffs_only`: freeze binary factors during blockwise optimization
- `--fix_theta`: keep STE thresholds fixed at `0.5`

#### Progressive mode

Use this only when you explicitly want patch-wise progressive quantization instead of the standard layerwise-initialize flow.

```bash
python blockwise_quant.py \
    --model_name Qwen/Qwen3-2B \
    --block_idx 0 \
    --progressive \
    --bit_width 2 \
    --group_size 64 \
    --num_rounds 4 \
    --schedule geometric \
    --save_dir blockwise_output/Qwen3-2B-progressive
```

### Layerwise quantization

You can still run layerwise quantization directly when you want standalone layerwise outputs.

```bash
python layerwise_quant.py \
    --model_name Qwen/Qwen3-2B \
    --block_idx 0 \
    --bit_width 2 \
    --group_size 64 \
    --num_steps 10000 \
    --dataset c4 \
    --nsamples 128 \
    --seqlen 2048 \
    --save_dir bqq_compressed_data/Qwen3-2B-64gs-10000step
```

This mode already includes:

- intra-layer Hessian-aware quantization
- optional multibqq joint bit optimization
- STE refinement on `Y/Z/theta/coeff`
- automatic full-model reconstruction by default

Default outputs:

- Patch data: `lm/src/bqq_compressed_data/<model>-<gs>gs-<steps>step/`
- Assembled full model: `lm/src/quantized_models/<model>/<model>-<bit>bit-<gs>gs.pth`

In normal use you do not need to run this manually before `blockwise_quant.py`, because blockwise mode will generate missing block results automatically.

### Legacy cache-first weight quantization

`weight_aware_quant_cached.py` still exists, but it is no longer the main recommended LM pipeline.
It is useful mainly for:

- target-wise or patch-wise weight quantization jobs
- legacy experiments
- auxiliary comparisons with older workflows

Example:

```bash
python weight_aware_quant_cached.py prepare-cache     --model_name Qwen/Qwen3.5-2B     --layer_threshold 4     --cache_dir cache/Qwen3.5-2B-layer4
```

Then:

```bash
python weight_aware_quant_cached.py quantize-target     --cache_dir  cache/Qwen3.5-2B-layer4     --save_dir   bqq_compressed_data/Qwen3.5-2B-32gs-10000step     --target_name model.layers.4.mlp.down_proj.weight     --bit_width  2     --group_size 32     --num_steps  10000
```

#### About bit-depth extension

The older `extend-target` flow is now best viewed as an auxiliary / reference workflow rather than a standard pipeline.

Because the current implementation directly optimizes for the requested `bit_width` in:

- `layerwise_quant.py`
- `blockwise_quant.py`
- `fine_tuning.py`

there is usually no need to first build an N-bit result and then extend it to N+k bits.

`extend-target` is still useful when you specifically want to:

- reuse an existing lower-bit result
- run legacy residual-extension experiments
- compare direct N-bit optimization against incremental extension

### Fine-tuning / KL distillation

`fine_tuning.py` supports three modes:

| Mode | Command | Loss |
|------|---------|------|
| `SFT only` | default | Cross-entropy |
| `SFT + KL` | `--teacher_model_name ...` | `ce_alpha * CE + kl_alpha * KL` |
| `KL only` | `--teacher_model_name ... --ce_alpha 0` | `kl_alpha * KL` |

```bash
python fine_tuning.py \
    --model_name Qwen/Qwen3-2B \
    --model_path /path/to/model.pth \
    --teacher_model_name Qwen/Qwen3-2B \
    --kl_alpha 1.0 \
    --kl_temperature 2.0
```

### Output examples

Typical directories:

```text
neural_network_compression/lm/
├── src/
│   ├── bqq_compressed_data/<model>-<gs>gs-<steps>step>/
│   │   └── _consolidated/<full_layer_name>.pth
│   └── quantized_models/<model>/
│       ├── <model>-<bit>bit-<gs>gs.pth
│       └── <model>-<bit>bit-<gs>gs-blockwise.pth
├── blockwise_output/<model>/
│   ├── block_0.pth
│   ├── block_1.pth
│   └── ...
└── fine_tuned_models/<model>/
    └── <model>-...-finetuned.pth
```

## TSUBAME4 SGE workflow

### N-bit quantization

`qsub_submit_qwen35.sh` automates all three steps (prepare-cache, list-targets, qsub) for Qwen3.5-2B/4B/9B.

```bash
cd neural_network_compression/lm
bash qsub_submit_qwen35.sh \
    --bit_width 2 \
    --walltime  8:00:00 \
    --workers_per_gpu 384
```

Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `--bit_width N` | 2 | Quantization bits |
| `--group_size N` | 32 | Patch group size |
| `--num_steps N` | 10000 | Simulated-annealing steps per patch |
| `--walltime HH:MM:SS` | 4:00:00 | Per-task GPU walltime |
| `--workers_per_gpu N` | 1024 | Worker processes per GPU (capped at 384 on TSUBAME4) |
| `--gpu_resource STR` | gpu_1=1 | SGE GPU resource request |
| `--dry_run` | — | Print qsub commands without submitting |

### Extending to higher bit-depth

Use `qsub_extend_array_job.sh` to submit residual-optimisation jobs.
Pass env vars via `qsub -v`:

```bash
SCRIPT_DIR=/path/to/neural_network_compression/lm
MODEL=Qwen3.5-2B

qsub -g tga-artic \
    -l gpu_1=1 -l h_rt=8:00:00 \
    -t 1-155 -tc 100 \
    -N bqq_ext_${MODEL} \
    -o qsub_jobs/${MODEL}-bit3-gs32/logs/ \
    -v "HF_HOME=...,TARGETS_LIST_FILE=qsub_jobs/${MODEL}-bit2-gs32/targets.txt,\
CACHE_DIR=cache/${MODEL}-layer4,\
SOURCE_DIR=bqq_compressed_data/${MODEL}-32gs-10000step,\
SAVE_DIR=bqq_compressed_data/${MODEL}-32gs-10000step-3bit,\
SIF_PATH=...,LM_SCRIPT_DIR=${SCRIPT_DIR},\
EXTRA_BITS=1,GROUP_SIZE=32,NUM_STEPS=10000,WORKERS_PER_GPU=384" \
    qsub_extend_array_job.sh
```

Resubmitting is always safe: targets with an existing `{target_name}.pth` are skipped immediately, and targets with partial `_rowX_colY.pth` patch files resume from where they left off.

---

## Post-quantization refinement (optional)

### Scale refinement

Refines the BQQ scale factors (a, b, c, d) per patch using Hessian-based optimization (closed-form ridge regression). Binary parameters Y, Z remain fixed. This is fast and does not require gradient-based optimization.

```bash
cd neural_network_compression/lm

# From a saved BQQ model
python scale_refine_bqq.py \
    --model_name Qwen/Qwen3-2B \
    --bqq_model quantized_model.pth \
    --output refined_model.pth \
    --dataset wikitext2 \
    --nsamples 128 \
    --seqlen 2048 \
    --damping 1e-6

# Or rebuild from compressed patch data
python scale_refine_bqq.py \
    --model_name Qwen/Qwen3-2B \
    --compressed_data bqq_compressed_data/Qwen3-2B-32gs-10000step \
    --bit_width 2 \
    --output refined_model.pth
```

| Option | Default | Description |
|--------|---------|-------------|
| `--bqq_model` | — | Path to saved BQQ model (mutually exclusive with `--compressed_data`) |
| `--compressed_data` | — | Path to BQQ patch files directory |
| `--damping` | 1e-6 | Relative diagonal damping for Cholesky stability |
| `--dataset` | wikitext2 | Calibration dataset (`wikitext2`, `ptb`, `c4`) |
| `--nsamples` | 128 | Number of calibration sequences |

### Fine-tuning / KL distillation

Fine-tune a quantized model with three loss modes:

| Mode | Command | Loss |
|------|---------|------|
| **SFT only** | (default) | Cross-entropy |
| **SFT + KL** | `--teacher_model_name ...` | `ce_alpha * CE + kl_alpha * KL` |
| **KL only** | `--teacher_model_name ... --ce_alpha 0` | `kl_alpha * KL` |

```bash
cd neural_network_compression/lm

# Standard SFT
python fine_tuning.py \
    --model_name Qwen/Qwen3-2B \
    --model_path src/quantized_models/Qwen3-2B/Qwen3-2B-2bit-64gs-blockwise.pth \
    --num_train_epochs 3 \
    --learning_rate 2e-5

# SFT + KL distillation (pretrained as teacher)
python fine_tuning.py \
    --model_name Qwen/Qwen3-2B \
    --model_path src/quantized_models/Qwen3-2B/Qwen3-2B-2bit-64gs-blockwise.pth \
    --teacher_model_name Qwen/Qwen3-2B \
    --kl_alpha 1.0 \
    --kl_temperature 2.0

# KL distillation only (no CE)
python fine_tuning.py \
    --model_name Qwen/Qwen3-2B \
    --model_path src/quantized_models/Qwen3-2B/Qwen3-2B-2bit-64gs-blockwise.pth \
    --teacher_model_name Qwen/Qwen3-2B \
    --ce_alpha 0 \
    --kl_alpha 1.0
```

| Option | Default | Description |
|--------|---------|-------------|
| `--teacher_model_name` | None | Teacher model for KL distillation (omit for SFT only) |
| `--ce_alpha` | 1.0 | Weight for cross-entropy loss (0 = KL only) |
| `--kl_alpha` | 1.0 | Weight for KL divergence loss |
| `--kl_temperature` | 2.0 | Temperature for softmax in KL divergence |
| `--max_seq_length` | 512 | Maximum sequence length |
| `--gradient_accumulation_steps` | 4 | Gradient accumulation steps |

Default fine-tuning output: `lm/fine_tuned_models/<model>/` unless `--output_dir` is specified.

---

## CV workflow

```bash
cd neural_network_compression/cv/bqq

# Cache
python weight_aware_quant_cached.py prepare-cache --model_name deit-s

# Quantize in parallel (local multi-GPU)
./weight_aware_quant_cached_parallel.sh \
    --model_name deit-s \
    --gpu_ids 0,1,2,3 \
    --finalize \
    --evaluate \
    --data_path /path/to/imagenet
```

`--finalize` rebuilds the full BQQ model from patch files after quantization completes.
`--evaluate` runs ImageNet evaluation and writes results to `results/`.

---

## Output directory layout

```
neural_network_compression/lm/
├── cache/<model>-layer<N>/
│   ├── metadata.json
│   ├── targets.txt
│   └── weights/<target_name>.pt
├── bqq_compressed_data/<model>-<gs>gs-<steps>step/
│   ├── <target_name>.pth               # reconstructed float tensor
│   └── <target_name>_row{i}_col{j}.pth # per-patch BQQ decomposition
└── qsub_jobs/<model>-bit<N>-gs<M>/
    ├── targets.txt
    └── logs/
```
