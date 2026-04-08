# nn_compression

Tools for applying BQQ to pretrained neural networks.
Supports weight-aware quantization, incremental bit-depth extension, model reconstruction, evaluation, and optional fine-tuning.

---

## Contents

| File | Description |
|------|-------------|
| `lm/weight_aware_quant_cached.py` | Cache-first weight quantization for LMs (main entry point) |
| `lm/weight_aware_quant.py` | Original (non-cached) weight-aware quantization |
| `lm/qsub_submit_qwen35.sh` | Submit N-bit quantization array jobs on TSUBAME (Qwen3.5-2B/4B/9B) |
| `lm/qsub_patch_array_job.sh` | SGE array job body for `quantize-target` (1 task = 1 weight tensor) |
| `lm/qsub_extend_array_job.sh` | SGE array job body for `extend-target` (1 task = 1 weight tensor) |
| `lm/binary_quadratic_network.py` | BQQ Linear layer definition |
| `lm/make_bqq_model_from_compressed_data.py` | Reconstruct a full model from saved BQQ patch files |
| `lm/evaluation.py` | Perplexity and task evaluation |
| `lm/fine_tuning.py` | Fine-tuning on a quantized model |

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

### TSUBAME-specific notes

**Always specify `-g tga-artic`.** Without it, TSUBAME runs in trial mode (walltime capped at 3 minutes).

**Walltime is per task, not per array.** Each SGE task covers one weight tensor.
8 hours (`8:00:00`) is recommended; 4 hours may be too short for large matrices.

**`workers_per_gpu` effective ceiling is 384.** TSUBAME4 nodes have 384 CPU threads; values above 384 have no effect with `gpu_1=1`.

### Monitoring and resubmission

```bash
# Job status
qstat -u $USER

# Completion count (excludes patch files)
SAVEDIR=bqq_compressed_data/Qwen3.5-9B-32gs-10000step
ls "$SAVEDIR" | grep -v "_row" | wc -l
wc -l < qsub_jobs/Qwen3.5-9B-bit2-gs32/targets.txt
```

Resubmitting is always safe: targets with an existing `{target_name}.pth` are skipped immediately, and targets with partial `_rowX_colY.pth` patch files resume from where they left off.

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
