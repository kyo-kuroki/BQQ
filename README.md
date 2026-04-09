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
└── nn_compression/
    ├── lm/                              # Language model quantization
    │   ├── weight_aware_quant_cached.py # Main entry point (prepare-cache / quantize-target / extend-target)
    │   ├── binary_quadratic_network.py  # BinaryQuadratic nn.Module definition
    │   ├── make_bqq_model_from_compressed_data.py  # Reconstruct full model from patches
    │   ├── scale_refine_bqq.py          # Post-quantization scale refinement
    │   ├── evaluation.py                # PPL / downstream evaluation
    │   ├── qsub_submit_qwen35.sh        # TSUBAME SGE job submission orchestrator
    │   ├── qsub_patch_array_job.sh      # SGE array job: quantize-target
    │   └── qsub_extend_array_job.sh     # SGE array job: extend-target
    └── cv/                              # Vision model quantization
```

## Quick start (Language model quantization)

以下は TSUBAME4 (SGE, H100) でのワークフローです。
Apptainer イメージ (`pytorch_llm_vllm.sif`) が必要です。

### Step 0: 環境変数

```bash
export BQQ_ROOT=/gs/bs/tga-artic/k-kuroki/BQQ
export LM_DIR=$BQQ_ROOT/nn_compression/lm
export HF_HOME=/gs/bs/tga-artic/k-kuroki/hf_cache
export SIF_PATH=/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif
```

### Step 1: 2-bit 量子化 (一括投入)

`qsub_submit_qwen35.sh` がキャッシュ作成 → ターゲットリスト生成 → SGE アレイジョブ投入を一括で行います。

```bash
cd $LM_DIR

# Qwen3.5-{2B, 4B, 9B} を 2-bit, group_size=32 で量子化
bash qsub_submit_qwen35.sh \
    --bit_width 2 \
    --group_size 32 \
    --num_steps 10000 \
    --walltime 8:00:00 \
    --workers_per_gpu 1024
```

**仕組み**: 各モデルについて以下を実行します。

1. `prepare-cache` — モデルをロードし、全 Linear 層の重みテンソルをディスクにキャッシュ
2. `list-targets` — キャッシュから量子化対象の一覧を `targets.txt` に出力
3. `qsub` — SGE アレイジョブ投入（1タスク = 1重み行列、GPU 1枚で384ワーカー並列）

**出力**:
- パッチファイル: `bqq_compressed_data/{model}-{gs}gs-{steps}step/{target}_row{i}_col{j}.pth`
- 再構成テンソル: `bqq_compressed_data/{model}-{gs}gs-{steps}step/{target}.pth`

### Step 2: 3-bit への拡張 (extend-target)

2-bit の結果の残差を最適化して 3-bit に拡張します。2-bit 量子化の完了後に実行。

```bash
MODEL=Qwen3.5-2B
JOB_DIR=$LM_DIR/qsub_jobs/${MODEL}-bit3-gs32

# targets.txt を 2-bit ジョブから流用
mkdir -p $JOB_DIR/logs
cp $LM_DIR/qsub_jobs/${MODEL}-bit2-gs32/targets.txt $JOB_DIR/targets.txt

N_TARGETS=$(wc -l < $JOB_DIR/targets.txt)

qsub -g tga-artic -t 1-${N_TARGETS}:1 \
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

### Step 3: BQQ モデルの構築

パッチファイルから `BinaryQuadratic` モジュールに変換し、推論可能なモデルを構築します。
初回実行時にパッチファイルの統合（`_consolidated/` へ保存）が行われます。

```bash
qsub -g tga-artic -l gpu_1=1 -l h_rt=4:00:00 -j y \
    -N "mkbqq_2B" \
    -o "$LM_DIR/qsub_jobs/make_bqq_model/logs/" \
    -v BQQ_ROOT=$BQQ_ROOT -v HF_HOME=$HF_HOME -v SIF_PATH=$SIF_PATH \
    -v SCRIPT_DIR=$LM_DIR \
    -v MODEL_BASENAME=Qwen3.5-2B \
    -v MODEL_NAME=Qwen/Qwen3.5-2B \
    -v COMPRESSED_DATA_DIR=$LM_DIR/bqq_compressed_data/Qwen3.5-2B-32gs-10000step \
    -v OUTPUT_DIR=$LM_DIR/quantized_model_data/Qwen3.5-2B \
    $LM_DIR/qsub_jobs/make_bqq_model/make_bqq_job.sh
```

**出力**: `quantized_model_data/Qwen3.5-2B/Qwen3.5-2B-2bit-32gs-10000step.pth`

### Step 4: スケール最適化 (Scale Refinement)

BQQ モデルの二値パラメータ (Y, Z) を固定したまま、スケール係数 (a, b, c, d) をキャリブレーションデータで再最適化します。

```bash
qsub -g tga-artic -l gpu_1=1 -l h_rt=2:00:00 -j y \
    -N "refine_2B" \
    -o "$LM_DIR/qsub_jobs/make_bqq_model/logs/" \
    -v BQQ_ROOT=$BQQ_ROOT -v HF_HOME=$HF_HOME -v SIF_PATH=$SIF_PATH \
    -v SCRIPT_DIR=$LM_DIR \
    -v MODEL_NAME=Qwen/Qwen3.5-2B \
    -v BQQ_MODEL_PATH=$LM_DIR/quantized_model_data/Qwen3.5-2B/Qwen3.5-2B-2bit-32gs-10000step.pth \
    -v REFINED_OUTPUT_PATH=$LM_DIR/quantized_model_data/Qwen3.5-2B/Qwen3.5-2B-2bit-32gs-10000step-refined.pth \
    $LM_DIR/qsub_jobs/make_bqq_model/scale_refine_job.sh
```

### Step 5: PPL 評価

```bash
qsub -g tga-artic -l gpu_1=1 -l h_rt=1:00:00 -j y \
    -N "ppl_2B" \
    -o "$LM_DIR/qsub_jobs/make_bqq_model/logs/" \
    -v BQQ_ROOT=$BQQ_ROOT -v HF_HOME=$HF_HOME -v SIF_PATH=$SIF_PATH \
    -v SCRIPT_DIR=$LM_DIR \
    -v MODEL_BASENAME=Qwen3.5-2B \
    -v MODEL_NAME=Qwen/Qwen3.5-2B \
    -v BQQ_MODEL_PATH=$LM_DIR/quantized_model_data/Qwen3.5-2B/Qwen3.5-2B-2bit-32gs-10000step.pth \
    $LM_DIR/qsub_jobs/make_bqq_model/ppl_eval_job.sh
```

**結果**: `results/{model_label}.csv` に PPL が保存されます。

## 進捗確認

```bash
# 完了ターゲット数
SAVEDIR=$LM_DIR/bqq_compressed_data/Qwen3.5-2B-32gs-10000step
ls "$SAVEDIR" | grep -v "_row" | wc -l

# SGE ジョブ状況
qstat -u $USER
```

## スキップ・再開の挙動

- **ターゲット単位**: `{target_name}.pth` が存在すれば即スキップ
- **パッチ単位**: `{target_name}_row{i}_col{j}.pth` が存在すれば再計算せず読み込んで復元

walltime kill 後の再投入で無駄な再計算なし。

## TSUBAME 固有の注意点

- `qsub` には必ず `-g tga-artic` を付ける（未指定だと walltime 3分のトライアルモード）
- walltime はジョブ全体ではなく**タスク単位**
- `workers_per_gpu` の実効上限: `min(mp.cpu_count(), workers_per_gpu)` = 384（ノードあたり384スレッド）
- H100 96GB / ノード

## `quantizer.py`

| Class | Description |
|-------|-------------|
| `BinaryQuadraticQuantization` | Original multi-bit implementation; includes activation-aware variant |
| `BinaryQuadraticQuantization2` | Refactored class used by all current LM and CV workflows |

`torch.compile` の mode はデフォルト `reduce-overhead`。`compile_mode="max-autotune"` を渡すと autotuning を有効化（コンパイルが遅くなるが最速カーネルを選択）。
