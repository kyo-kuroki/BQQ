#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l gpu_1=1
#$ -l h_rt=0:15:00

set -euo pipefail

SIF_PATH="/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif"
BQQ_ROOT="/gs/bs/tga-artic/k-kuroki/BQQ"

echo "Host: $(hostname)"
echo "Date: $(date --iso-8601=seconds)"

apptainer exec --nv \
  --bind "${HOME}:${HOME}" \
  --bind "${BQQ_ROOT}:${BQQ_ROOT}" \
  --env "OMP_NUM_THREADS=1" \
  --env "OPENBLAS_NUM_THREADS=1" \
  --pwd "${BQQ_ROOT}/neural_network_compression/src" \
  "${SIF_PATH}" \
  python bqq_triton_kernel.py

echo "Done: $(date --iso-8601=seconds)"
