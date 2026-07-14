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
  "${PY}" - <<'EOF'
mods = [
    ("fla",                      "Qwen3.5 linear_attn (gated delta-net)"),
    ("fla.modules",              "  -> FusedRMSNormGated"),
    ("fla.ops.gated_delta_rule", "  -> gated delta rule kernels"),
    ("causal_conv1d",            "Qwen3.5 linear_attn short conv"),
    ("causal_conv1d_cuda",       "  -> its CUDA extension"),
    ("fast_hadamard_transform",  "RHT for Incoherent/PackedIncoherent BQQ"),
    ("quiptools_cuda",           "QuIP# E8P kernel (benchmark only, optional)"),
]
import importlib
for name, what in mods:
    try:
        importlib.import_module(name)
        print(f"  OK   {name:28s} {what}")
    except Exception as e:
        print(f"  MISS {name:28s} {what}  [{type(e).__name__}]")
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

echo "== flash-linear-attention (from GitHub; the PyPI wheel has no fla/modules) =="
git clone --depth 1 https://github.com/fla-org/flash-linear-attention.git "${SRC_DIR}/fla" \
  && ${PIP} install "${PIP_FLAGS[@]}" --force-reinstall "${SRC_DIR}/fla"

echo "== causal-conv1d (source build; release wheels need GLIBC_2.32) =="
git clone --depth 1 https://github.com/Dao-AILab/causal-conv1d.git "${SRC_DIR}/causal-conv1d" \
  && CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS:-8}" \
     ${PIP} install "${PIP_FLAGS[@]}" --force-reinstall "${SRC_DIR}/causal-conv1d"

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
