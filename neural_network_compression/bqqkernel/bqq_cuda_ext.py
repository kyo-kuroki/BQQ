"""Python wrapper for the BQQ CUDA kernel.

Compiles and caches the CUDA extension on first import.

Usage::

    from .bqq_cuda_ext import cuda_bqq_forward
    out = cuda_bqq_forward(packed.Y_packed, packed.Z_packed, X,
                           packed.a, packed.b, packed.c, packed.d)
"""

import os
import sys
import torch
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
_ext = None
_forward_flat_op = None


def _use_blackwell_fallback():
    if os.environ.get("BQQ_CUDA_FORCE_BLACKWELL_FALLBACK") == "1":
        return True
    if os.environ.get("BQQ_CUDA_DISABLE_BLACKWELL_FALLBACK") == "1":
        return False
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 12


def _get_ext():
    global _ext
    if _ext is None:
        python_bin = os.path.dirname(sys.executable)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if python_bin and python_bin not in path_parts:
            os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
        use_blackwell = _use_blackwell_fallback()
        source = 'bqq_cuda_blackwell.cu' if use_blackwell else 'bqq_cuda.cu'
        name = 'bqq_cuda_blackwell' if use_blackwell else 'bqq_cuda'
        _ext = load(
            name=name,
            sources=[os.path.join(_dir, source)],
            extra_cuda_cflags=['-O3', '--use_fast_math', '-std=c++17'],
            verbose=True,
        )
    return _ext


def bqq_forward(Y_packed, Z_packed, X, a, b, c, d, bias):
    """BQQ forward — single call, all reshape handled in C++."""
    return _get_ext().bqq_forward(Y_packed, Z_packed, X, a, b, c, d, bias)


def _get_forward_flat_op():
    """Return a Dynamo-visible opaque op for the flat fused kernel.

    vLLM uses fullgraph TorchDynamo capture before CUDA graph replay.  A raw
    pybind extension function is not fake-tensor traceable, so expose it as a
    custom op with a fake implementation that only describes the output shape.
    """
    global _forward_flat_op
    if _forward_flat_op is not None:
        return _forward_flat_op

    # The fused decode epilogue uses ``ws`` as an accumulation buffer and
    # clears it before returning. Declaring that mutation is required for
    # TorchDynamo/Inductor and CUDA Graph memory planning to preserve ordering.
    @torch.library.custom_op("bqq::forward_flat", mutates_args=("ws",))
    def _op(
        Y_flat: torch.Tensor,
        Z_flat: torch.Tensor,
        X: torch.Tensor,
        a_flat: torch.Tensor,
        b_flat: torch.Tensor,
        c_flat: torch.Tensor,
        d_flat: torch.Tensor,
        bias: torch.Tensor,
        ws: torch.Tensor,
        bit_width: int,
        row_width: int,
        col_width: int,
        y_row: int,
        z_col: int,
    ) -> torch.Tensor:
        return _get_ext().bqq_forward_flat(
            Y_flat, Z_flat, X, a_flat, b_flat, c_flat, d_flat, bias, ws,
            bit_width, row_width, col_width, y_row, z_col)

    @_op.register_fake
    def _op_fake(
        Y_flat: torch.Tensor,
        Z_flat: torch.Tensor,
        X: torch.Tensor,
        a_flat: torch.Tensor,
        b_flat: torch.Tensor,
        c_flat: torch.Tensor,
        d_flat: torch.Tensor,
        bias: torch.Tensor,
        ws: torch.Tensor,
        bit_width: int,
        row_width: int,
        col_width: int,
        y_row: int,
        z_col: int,
    ) -> torch.Tensor:
        return X.new_empty((*X.shape[:-1], row_width * y_row))

    _forward_flat_op = _op
    return _forward_flat_op


def bqq_forward_flat(
    Y_flat, Z_flat, X, a_flat, b_flat, c_flat, d_flat, bias, ws,
    bit_width, row_width, col_width, y_row, z_col,
):
    return _get_forward_flat_op()(
        Y_flat, Z_flat, X, a_flat, b_flat, c_flat, d_flat, bias, ws,
        bit_width, row_width, col_width, y_row, z_col)
