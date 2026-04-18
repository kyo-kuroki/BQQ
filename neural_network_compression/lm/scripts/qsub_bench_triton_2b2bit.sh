#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l gpu_1=1
#$ -l h_rt=0:20:00

set -euo pipefail

SIF_PATH="/gs/bs/tga-artic/tmp/tsubame-handson/containers/pytorch_llm_vllm.sif"
BQQ_ROOT="/gs/bs/tga-artic/k-kuroki/BQQ"
LM_DIR="${BQQ_ROOT}/neural_network_compression/lm"
MODEL_PATH="${LM_DIR}/quantized_model_data/Qwen3.5-2B-2bit-32gs-blockwise-fwd.pth"

echo "Host: $(hostname)"
echo "Date: $(date --iso-8601=seconds)"
echo "Model: ${MODEL_PATH}"

apptainer exec --nv \
  --bind "${HOME}:${HOME}" \
  --bind "${BQQ_ROOT}:${BQQ_ROOT}" \
  --env "HF_HOME=/gs/bs/tga-artic/k-kuroki/hf_cache" \
  --env "OMP_NUM_THREADS=1" \
  --env "OPENBLAS_NUM_THREADS=1" \
  --pwd "${LM_DIR}" \
  "${SIF_PATH}" \
  python - <<'PYEOF'
import sys, os, time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('.')), 'neural_network_compression', 'src'))
sys.path.insert(0, '../src')

from bqq_modules import PackedBinaryQuadratic, BinaryQuadratic

MODEL_PATH = "quantized_model_data/Qwen3.5-2B-2bit-32gs-blockwise-fwd.pth"
DEVICE = "cuda:0"
N_WARMUP = 3
N_BENCH  = 10

from bqq_modules import pack_binaryquadratic_model

import copy
from bqq_modules import pack_binaryquadratic_model

print("Loading model (weights_only=False) ...")
model_bq = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
n_bq = sum(1 for m in model_bq.modules() if isinstance(m, BinaryQuadratic))
print(f"BinaryQuadratic layers: {n_bq}")

print("Packing copy → PackedBinaryQuadratic ...")
model_packed = copy.deepcopy(model_bq)
pack_binaryquadratic_model(model_packed)
n_packed = sum(1 for m in model_packed.modules() if isinstance(m, PackedBinaryQuadratic))
print(f"PackedBinaryQuadratic layers: {n_packed}")

model_bq    = model_bq.to(DEVICE).eval()
model_packed = model_packed.to(DEVICE).eval()

dummy_ids = torch.zeros(1, 64, dtype=torch.long, device=DEVICE)

# -------------------------------------------------------
# Helper: timed forward pass
# -------------------------------------------------------
def bench(label, mdl, use_kernel):
    PackedBinaryQuadratic.use_packed_kernel = use_kernel
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = mdl(dummy_ids)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_BENCH):
            out = mdl(dummy_ids)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / N_BENCH * 1000
    logits = out.logits.float()
    print(f"[{label}]  {elapsed:.1f} ms/fwd  |  logits[0,0,:5] = {logits[0,0,:5].tolist()}")
    return logits, elapsed

from bqq_triton_kernel import register_zy_x_kernel

def bench_mode(label, mdl, packed_kernel_on, zy_x_on):
    PackedBinaryQuadratic.use_packed_kernel = packed_kernel_on
    PackedBinaryQuadratic.use_zy_x_kernel   = zy_x_on
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = mdl(dummy_ids)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_BENCH):
            out = mdl(dummy_ids)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / N_BENCH * 1000
    logits = out.logits.float()
    print(f"[{label:30s}]  {ms:6.1f} ms/fwd  logits[0,0,:3]={logits[0,0,:3].tolist()}")
    return logits, ms

print("\n--- BinaryQuadratic (bool→bfloat16 + cuBLAS) ---")
logits_ref, t_bq = bench_mode("BinaryQuadratic",       model_bq,    False, False)

print("\n--- PackedBinaryQuadratic modes ---")
logits_cu,  t_cu  = bench_mode("Packed + cuBLAS unpack",  model_packed, False, False)
logits_tr1, t_tr1 = bench_mode("Packed + Triton AND+popc", model_packed, True,  False)
logits_zyx, t_zyx = bench_mode("Packed + Triton Y@(Z@X)", model_packed, False, True)

print("\n========== Summary ==========")
for label, t in [("BinaryQuadratic  ", t_bq),
                  ("Packed+cuBLAS    ", t_cu),
                  ("Packed+AND+popc  ", t_tr1),
                  ("Packed+Y@(Z@X)   ", t_zyx)]:
    print(f"  {label}: {t:6.1f} ms  ({t_bq/t:.2f}x vs BQ baseline)")

print()
for name, other in [("BQ vs Packed+cuBLAS",   logits_cu),
                    ("BQ vs AND+popcount",     logits_tr1),
                    ("BQ vs Y@(Z@X)",          logits_zyx)]:
    d = (logits_ref - other).abs()
    ok = "✓" if d.max() < 1e-3 else "✗"
    print(f"  {ok} {name}  max={d.max().item():.2e}  mean={d.mean().item():.2e}")
PYEOF

echo "Done: $(date --iso-8601=seconds)"
