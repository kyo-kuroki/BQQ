"""Python wrapper for the BQQ CUDA kernel.

Compiles and caches the CUDA extension on first import.

Usage::

    from bqq_cuda_ext import cuda_bqq_forward
    out = cuda_bqq_forward(packed.Y_packed, packed.Z_packed, X,
                           packed.a, packed.b, packed.c, packed.d)
"""

import os
import torch
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load(
            name='bqq_cuda',
            sources=[os.path.join(_dir, 'bqq_cuda.cu')],
            extra_cuda_cflags=['-O3', '--use_fast_math', '-std=c++17'],
            verbose=True,
        )
    return _ext


def cuda_bqq_forward(
    Y_packed: torch.Tensor,
    Z_packed: torch.Tensor,
    X: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """BQQ forward via warp-level CUDA kernel.

    Parameters
    ----------
    Y_packed : [bit, row, col, y_row, k8]  uint8
    Z_packed : [bit, row, col, z_col, k8]  uint8
    X        : [..., in_features]
    a, b, c  : [bit, row, col, 1, 1]
    d        : [row, col, 1, 1]
    bias     : [out_features] or None
    """
    ext = _get_ext()

    bit_width, row_width, col_width, y_row, k8 = Y_packed.shape
    z_col = Z_packed.shape[3]
    B_total = bit_width * row_width * col_width
    orig_dtype = X.dtype
    device = Y_packed.device

    orig_shape = X.shape
    X_2d = X.to(device).reshape(-1, orig_shape[-1])
    batch = X_2d.shape[0]

    Y_flat = Y_packed.reshape(B_total, y_row, k8).contiguous()
    Z_flat = Z_packed.reshape(B_total, z_col, k8).contiguous()
    X_view = X_2d.reshape(batch, col_width, z_col).float().contiguous()

    a_flat = a.squeeze(-1).squeeze(-1).reshape(B_total).float().contiguous()
    b_flat = b.squeeze(-1).squeeze(-1).reshape(B_total).float().contiguous()
    c_flat = c.squeeze(-1).squeeze(-1).reshape(B_total).float().contiguous()
    d_flat = d.squeeze(-1).squeeze(-1).reshape(
        row_width * col_width).float().contiguous()

    out = ext.bqq_forward_cuda(
        Y_flat, Z_flat, X_view,
        a_flat, b_flat, c_flat, d_flat,
        batch, row_width, col_width, bit_width,
        y_row, z_col, k8,
    )

    out = out.reshape(batch, row_width * y_row)
    if bias is not None:
        out = out + bias.float().to(device)

    return out.reshape(*orig_shape[:-1], row_width * y_row).to(orig_dtype)
