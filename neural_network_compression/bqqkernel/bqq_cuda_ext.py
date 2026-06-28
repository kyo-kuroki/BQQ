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


def _get_ext():
    global _ext
    if _ext is None:
        python_bin = os.path.dirname(sys.executable)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if python_bin and python_bin not in path_parts:
            os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
        _ext = load(
            name='bqq_cuda',
            sources=[os.path.join(_dir, 'bqq_cuda.cu')],
            extra_cuda_cflags=['-O3', '--use_fast_math', '-std=c++17'],
            verbose=True,
        )
    return _ext


def bqq_forward(Y_packed, Z_packed, X, a, b, c, d, bias):
    """BQQ forward — single call, all reshape handled in C++."""
    return _get_ext().bqq_forward(Y_packed, Z_packed, X, a, b, c, d, bias)
