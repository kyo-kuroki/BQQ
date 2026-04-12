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
    │   ├── weight_aware_quant_cached.py # Main entry point (prepare-cache / quantize-target / extend-target)
    │   ├── build_bqq_model.py  # Replace Linear→BinaryQuadratic, build model from patches or blocks
    │   ├── scale_refine_bqq.py          # Post-quantization scale refinement
    │   ├── evaluation.py                # PPL / downstream evaluation
    │   ├── qsub_submit_qwen35.sh        # SGE job submission orchestrator
    │   ├── qsub_patch_array_job.sh      # SGE array job: quantize-target
    │   └── qsub_extend_array_job.sh     # SGE array job: extend-target
    └── cv/                              # Vision model quantization
```

## Quick start (Language model quantization)

The following workflow is designed for an SGE cluster with GPUs and Apptainer.

### Step 0: Environment variables

Set paths according to your environment:

```bash
export BQQ_ROOT=/path/to/BQQ
export LM_DIR=$BQQ_ROOT/neural_network_compression/lm
export HF_HOME=/path/to/hf_cache
export SIF_PATH=/path/to/pytorch_llm_vllm.sif
```

### Step 1: N-bit quantization (batch submission)

`qsub_submit_qwen35.sh` automates the full pipeline: cache creation, target listing, and SGE array job submission.

```bash
cd $LM_DIR

# Quantize Qwen3.5-{2B, 4B, 9B} at 2-bit, group_size=32
bash qsub_submit_qwen35.sh \
    --bit_width 2 \
    --group_size 32 \
    --num_steps 10000 \
    --walltime 8:00:00 \
    --workers_per_gpu 1024
```

**What it does** (for each model):

1. `prepare-cache` -- Load the model and cache all Linear weight tensors to disk
2. `list-targets` -- Write all quantization target names to `targets.txt`
3. `qsub` -- Submit an SGE array job (1 task = 1 weight matrix, each task runs 384 parallel workers on 1 GPU)

**Output**:
- Patch files: `bqq_compressed_data/{model}-{gs}gs-{steps}step/{target}_row{i}_col{j}.pth`
- Reconstructed tensors: `bqq_compressed_data/{model}-{gs}gs-{steps}step/{target}.pth`

### Step 2: Extend to higher bit-depth (extend-target)

Extend N-bit results to (N+k)-bit by optimising the residual. Run after the base quantization completes.

```bash
MODEL=Qwen3.5-2B
JOB_DIR=$LM_DIR/qsub_jobs/${MODEL}-bit3-gs32

# Reuse targets.txt from the 2-bit job
mkdir -p $JOB_DIR/logs
cp $LM_DIR/qsub_jobs/${MODEL}-bit2-gs32/targets.txt $JOB_DIR/targets.txt

N_TARGETS=$(wc -l < $JOB_DIR/targets.txt)

qsub -g <group> -t 1-${N_TARGETS}:1 \
    -l gpu_1=1 -l h_rt=8:00:00 -j y \
    -N "ext3b_${MODEL}" \
    -o "$JOB_DIR/logs/" \
    -v LM_SCRIPT_DIR=$LM_DIR \
    -v TARGETS_LIST_FILE=$JOB_DIR/targets.txt \
    -v SOURCE_DIR=$LM_DIR/bqq_compressed_data/${MODEL}-32gs-10000step \
    -v SAVE_DIR=$LM_DIR/bqq_compressed_data/${MODEL}-32gs-10000step-3bit \
    -v CACHE_DIR=$LM_DIR/cache/${MODEL}-layer0 \
    -v SIF_PATH=$SIF_PATH -v HF_HOME=$HF_HOME \
    -v EXTRA_BITS=1 -v GROUP_SIZE=32 -v NUM_STEPS=10000 \
    -v WORKERS_PER_GPU=384 -v RANK_SCALE=1.0 \
    $LM_DIR/qsub_extend_array_job.sh
```

### Step 3: Build BQQ model

Convert patch files into `BinaryQuadratic` modules and save a model that preserves the binary decomposition (Y, Z, A coefficients). On the first run, patch files are consolidated into per-target files under `_consolidated/` for fast loading.

```bash
qsub -g <group> -l gpu_1=1 -l h_rt=4:00:00 -j y \
    -N "mkbqq_${MODEL}" \
    -o "$LM_DIR/qsub_jobs/make_bqq_model/logs/" \
    -v BQQ_ROOT=$BQQ_ROOT -v HF_HOME=$HF_HOME -v SIF_PATH=$SIF_PATH \
    -v SCRIPT_DIR=$LM_DIR \
    -v MODEL_BASENAME=Qwen3.5-2B \
    -v MODEL_NAME=Qwen/Qwen3.5-2B \
    -v COMPRESSED_DATA_DIR=$LM_DIR/bqq_compressed_data/Qwen3.5-2B-32gs-10000step \
    -v OUTPUT_DIR=$LM_DIR/quantized_model_data/Qwen3.5-2B \
    $LM_DIR/qsub_jobs/make_bqq_model/make_bqq_job.sh
```

**Output**: `quantized_model_data/Qwen3.5-2B/Qwen3.5-2B-2bit-32gs-10000step.pth`

### Step 4: Scale refinement

Refine the scale coefficients (a, b, c, d) of each `BinaryQuadratic` layer while keeping the binary parameters (Y, Z) fixed. Uses calibration data to minimise `||WX - W'X||^2`.

```bash
qsub -g <group> -l gpu_1=1 -l h_rt=2:00:00 -j y \
    -N "refine_${MODEL}" \
    -o "$LM_DIR/qsub_jobs/make_bqq_model/logs/" \
    -v BQQ_ROOT=$BQQ_ROOT -v HF_HOME=$HF_HOME -v SIF_PATH=$SIF_PATH \
    -v SCRIPT_DIR=$LM_DIR \
    -v MODEL_NAME=Qwen/Qwen3.5-2B \
    -v BQQ_MODEL_PATH=$LM_DIR/quantized_model_data/Qwen3.5-2B/Qwen3.5-2B-2bit-32gs-10000step.pth \
    -v REFINED_OUTPUT_PATH=$LM_DIR/quantized_model_data/Qwen3.5-2B/Qwen3.5-2B-2bit-32gs-10000step-refined.pth \
    $LM_DIR/qsub_jobs/make_bqq_model/scale_refine_job.sh
```

### Step 5: Perplexity evaluation

```bash
qsub -g <group> -l gpu_1=1 -l h_rt=1:00:00 -j y \
    -N "ppl_${MODEL}" \
    -o "$LM_DIR/qsub_jobs/make_bqq_model/logs/" \
    -v BQQ_ROOT=$BQQ_ROOT -v HF_HOME=$HF_HOME -v SIF_PATH=$SIF_PATH \
    -v SCRIPT_DIR=$LM_DIR \
    -v MODEL_BASENAME=Qwen3.5-2B \
    -v MODEL_NAME=Qwen/Qwen3.5-2B \
    -v BQQ_MODEL_PATH=$LM_DIR/quantized_model_data/Qwen3.5-2B/Qwen3.5-2B-2bit-32gs-10000step.pth \
    $LM_DIR/qsub_jobs/make_bqq_model/ppl_eval_job.sh
```

Results are saved to `results/{model_label}.csv`.

## Monitoring progress

```bash
# Count completed targets
SAVEDIR=$LM_DIR/bqq_compressed_data/Qwen3.5-2B-32gs-10000step
ls "$SAVEDIR" | grep -v "_row" | wc -l

# Check SGE job status
qstat -u $USER
```

## Skip and resume behaviour

- **Target level**: If `{target_name}.pth` exists, the target is skipped immediately.
- **Patch level**: If `{target_name}_row{i}_col{j}.pth` exists, the patch is loaded from disk instead of recomputed.

Jobs killed by walltime can be resubmitted without redundant computation.

## SGE notes

- Always specify your group with `-g <group>` (omitting it limits walltime to 3 minutes in trial mode).
- Walltime (`h_rt`) is **per task**, not per array job.
- Effective worker count is capped at `min(mp.cpu_count(), workers_per_gpu)`.

## `quantizer.py`

| Class | Description |
|-------|-------------|
| `BinaryQuadraticQuantization` | Main BQQ class for all LM and CV workflows. `binarize_scaling=True` for V1 mode (binarized scaling), `False` (default) for V2 mode (continuous scaling). |

`torch.compile` defaults to `mode="reduce-overhead"`. Pass `compile_mode="max-autotune"` for autotuning (slower compilation but potentially faster kernels).
