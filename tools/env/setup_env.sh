#!/usr/bin/env bash
# Build the CUDA extensions this repo needs from source.
#
# The published wheels are unusable here (glibc 2.31 vs their GLIBC_2.32, and the
# PyPI flash-linear-attention wheel is missing fla/modules entirely), so every
# package is compiled locally.  See tools/env/README.md.
#
#   bash tools/env/setup_env.sh            # fla + causal-conv1d + fast-hadamard-transform
#   bash tools/env/setup_env.sh --quip     # also build quiptools (QuIP# comparison)
#   bash tools/env/setup_env.sh --check    # only report what is importable
set -uo pipefail

PY="${PY:-/artic/k-kuroki/.conda/envs/py311/bin/python}"
PIP="${PY} -m pip"
QUIP_SHARP_DIR="${QUIP_SHARP_DIR:-/work2/k-kuroki/quip-sharp}"
SRC_DIR="${SRC_DIR:-$(mktemp -d /tmp/bqq-env-src.XXXXXX)}"
WITH_QUIP=0
CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --quip)  WITH_QUIP=1 ;;
    --check) CHECK_ONLY=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

check() {
  BQQ_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" "${PY}" - <<'EOF'
import importlib, os, sys
# The fla/causal_conv1d shims only resolve with neural_network_compression on
# the path -- mirror how the run scripts invoke python.
repo = os.environ["BQQ_REPO"]
sys.path.insert(0, os.path.join(repo, "neural_network_compression"))
sys.path.insert(0, repo)

pip_built = [
    ("fast_hadamard_transform",  "RHT for Incoherent/PackedIncoherent BQQ"),
    ("quiptools_cuda",           "QuIP# E8P kernel (benchmark only, optional)"),
]
shims = [
    ("fla",                      "Qwen3.5 linear_attn shim (repo)"),
    ("fla.modules",              "  -> FusedRMSNormGated"),
    ("causal_conv1d",            "Qwen3.5 short-conv shim (repo)"),
]
for name, what in pip_built + shims:
    try:
        m = importlib.import_module(name)
        where = "repo shim" if "neural_network_compression" in getattr(m, "__file__", "") else "installed"
        print(f"  OK   {name:26s} {what}  [{where}]")
    except Exception as e:
        print(f"  MISS {name:26s} {what}  [{type(e).__name__}]")
EOF
}

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "== import check =="
  check
  exit 0
fi

# --no-deps everywhere: an unqualified pip install in this env has silently
# uninstalled working packages before.
PIP_FLAGS=(--no-deps --no-build-isolation)

# NOTE: do NOT pip install flash-linear-attention or causal-conv1d.  The repo
# ships pure-torch shims at neural_network_compression/{fla,causal_conv1d} that
# are picked up when neural_network_compression is on PYTHONPATH, and vLLM has
# its own vendored fla, so a site-packages copy is redundant -- it just shadows
# the shim depending on path order and drifts out of sync with transformers.
# The shim trails the real Triton kernels by ~11% on the transformers-direct
# decode path only; if you specifically need that speed, build them from source
# the same way (git clone + --no-build-isolation) and put site-packages ahead
# of neural_network_compression on PYTHONPATH.

echo "== fast-hadamard-transform (source build; the sdist ships no csrc) =="
git clone --depth 1 https://github.com/Dao-AILab/fast-hadamard-transform.git "${SRC_DIR}/fht" \
  && MAX_JOBS="${MAX_JOBS:-8}" ${PIP} install "${PIP_FLAGS[@]}" "${SRC_DIR}/fht"

if [[ "${WITH_QUIP}" == "1" ]]; then
  echo "== quiptools (QuIP# E8P kernel; c++20 -> c++17 for gcc 9.4) =="
  if [[ -d "${QUIP_SHARP_DIR}/quiptools" ]]; then
    cp -r "${QUIP_SHARP_DIR}/quiptools" "${SRC_DIR}/quiptools"
    sed -i "s/-std=c++20/-std=c++17/g" "${SRC_DIR}/quiptools/setup.py"
    ${PIP} install "${PIP_FLAGS[@]}" "${SRC_DIR}/quiptools"
  else
    echo "  skipped: ${QUIP_SHARP_DIR}/quiptools not found (set QUIP_SHARP_DIR)"
  fi
fi

echo
echo "== import check =="
check
echo
echo "sources were built in ${SRC_DIR} (safe to delete)"
