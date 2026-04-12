# nn_compression

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
| `lm/block_wise_quant.py` | Block-wise quantization with block output error optimization |
| `lm/build_bqq_model.py` | Replace Linear→BinaryQuadratic, build model from patches or blocks |
| `lm/scale_refine_bqq.py` | Hessian-based scale factor refinement (post-quantization) |
| `lm/fine_tuning.py` | Fine-tuning / KL distillation on a quantized model |
| `lm/evaluation.py` | Perplexity and task evaluation |

---

## LM workflow

### Step 1 — prepare cache

Load the model once and save quantization-target weights to disk.

```bash
cd nn_compression/lm

python weight_aware_quant_cached.py prepare-cache \
    --model_name Qwen/Qwen3.5-2B \
    --layer_threshold 4 \
    --cache_dir cache/Qwen3.5-2B-layer4
```

`--layer_threshold N` skips layers whose index is below N (e.g. embedding layers).

### Step 2 — quantize (N-bit)

Each `quantize-target` call quantizes one weight tensor by splitting it into patches and running BQQ in parallel.

```bash
python weight_aware_quant_cached.py quantize-target \
    --cache_dir  cache/Qwen3.5-2B-layer4 \
    --save_dir   bqq_compressed_data/Qwen3.5-2B-32gs-10000step \
    --target_name model.layers.4.mlp.down_proj.weight \
    --bit_width  2 \
    --group_size 32 \
    --num_steps  10000 \
    --workers_per_gpu 384
```

Output files per target:

- `{save_dir}/{target_name}.pth` — reconstructed tensor (float)
- `{save_dir}/{target_name}_row{i}_col{j}.pth` — per-patch BQQ decomposition (list of bit-layer dicts)

### Step 3 — extend to higher bit-depth (optional)

Given a completed N-bit result, optimise the residual to produce an (N+k)-bit result without re-running the earlier bits.

```bash
python weight_aware_quant_cached.py extend-target \
    --cache_dir  cache/Qwen3.5-2B-layer4 \
    --source_dir bqq_compressed_data/Qwen3.5-2B-32gs-10000step \
    --save_dir   bqq_compressed_data/Qwen3.5-2B-32gs-10000step-3bit \
    --target_name model.layers.4.mlp.down_proj.weight \
    --extra_bits 1 \
    --group_size 32 \
    --num_steps  10000 \
    --workers_per_gpu 384
```

The saved patch files in `--save_dir` contain all N + k bit-layer entries, so the format is identical to a native (N+k)-bit run.
Existing output files are skipped automatically, so interrupted jobs can be safely resubmitted.

### Step 2b — block-wise quantization (alternative)

Block-wise quantization optimizes all continuous parameters in a transformer block (BQQ scale factors + unquantized Linear weights + LayerNorm) to minimize the block output error against the pretrained model. Each block is independent and can be processed in parallel.

```bash
cd nn_compression/lm

# Quantize a single block
python block_wise_quant.py \
    --model_name Qwen/Qwen3-2B \
    --block_idx 0 \
    --bit_width 2 \
    --group_size 32 \
    --num_steps 20000 \
    --dataset c4 \
    --nsamples 128 \
    --seqlen 2048 \
    --epochs 10 \
    --lr 1e-5 \
    --device cuda:0 \
    --save_dir blockwise_output/Qwen3-2B
```

Output: `blockwise_output/Qwen3-2B/block_0.pth` (full module with optimized parameters).

For each Linear weight in the block, the pipeline:
1. BQQ quantizes the weight → replaces Linear with BinaryQuadratic (Y, Z fixed as buffers)
2. Optimizes **all** continuous params in the block (BQQ a,b,c,d + remaining unquantized weights + norms) via AdamW to minimize `||block(input) - pretrained_output||²`
3. Moves to the next weight

As quantization progresses, the number of free parameters decreases (large Linear weights are replaced by small a,b,c,d scale factors).

**Parallel execution** across blocks:

```bash
for i in $(seq 0 23); do
    python block_wise_quant.py --model_name Qwen/Qwen3-2B \
        --block_idx $i --save_dir blockwise_output/Qwen3-2B \
        --device cuda:$((i % 4)) &
done
wait
```

**Assemble full model from blocks:**

```bash
python build_bqq_model.py \
    --model_name Qwen/Qwen3-2B \
    --block_dir blockwise_output/Qwen3-2B
```

Blocks that are missing will be kept as pretrained (partial quantization is supported).

### Other commands

```bash
# List all quantization targets in a cache
python weight_aware_quant_cached.py list-targets --cache_dir cache/Qwen3.5-2B-layer4

# List patches for a target (useful for fine-grained parallelism)
python weight_aware_quant_cached.py list-patches \
    --cache_dir cache/Qwen3.5-2B-layer4 --group_size 32
```

---

## TSUBAME4 SGE workflow

### N-bit quantization

`qsub_submit_qwen35.sh` automates all three steps (prepare-cache, list-targets, qsub) for Qwen3.5-2B/4B/9B.

```bash
cd nn_compression/lm
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
SCRIPT_DIR=/path/to/nn_compression/lm
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
cd nn_compression/lm

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
cd nn_compression/lm

# Standard SFT
python fine_tuning.py \
    --model_name Qwen/Qwen3-2B \
    --model_path quantized_model.pth \
    --num_train_epochs 3 \
    --learning_rate 2e-5

# SFT + KL distillation (pretrained as teacher)
python fine_tuning.py \
    --model_name Qwen/Qwen3-2B \
    --model_path quantized_model.pth \
    --teacher_model_name Qwen/Qwen3-2B \
    --kl_alpha 1.0 \
    --kl_temperature 2.0

# KL distillation only (no CE)
python fine_tuning.py \
    --model_name Qwen/Qwen3-2B \
    --model_path quantized_model.pth \
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

Output: `{output_dir}/trained_model.pth`

---

## CV workflow

```bash
cd nn_compression/cv/bqq

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
nn_compression/lm/
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
