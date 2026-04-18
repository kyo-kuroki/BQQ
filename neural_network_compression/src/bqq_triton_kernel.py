"""Triton AND+popcount kernel for PackedBinaryQuadratic.

Computes:  W_core[b, i, j] = Σ_k  popcount( Y_packed[b,i,k] AND Z_packed[b,j,k] )

Y_packed : [B, y_row, k8]  uint8  (rows are packed binary vectors)
Z_packed : [B, z_col, k8]  uint8  (already transposed; columns packed)
Output   : [B, y_row, z_col]  float32

Usage (one-time setup)::

    from bqq_triton_kernel import register_triton_kernel
    register_triton_kernel()   # sets PackedBinaryQuadratic.packed_kernel + use_packed_kernel=True

Or import bqq_modules after this file is on sys.path — it auto-registers in a try/except.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _packed_binary_matmul_kernel(
    Y_ptr,
    Z_ptr,
    Out_ptr,
    y_row,          # runtime scalar
    z_col,          # runtime scalar
    k8,             # runtime scalar — number of uint8 bytes per packed vector
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """Grid: (B, ceil(y_row/BLOCK_I), ceil(z_col/BLOCK_J))."""
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)  # [BLOCK_I]
    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)  # [BLOCK_J]

    acc = tl.zeros([BLOCK_I, BLOCK_J], dtype=tl.int32)

    Y_base = pid_b * y_row * k8
    Z_base = pid_b * z_col * k8

    for k in tl.range(k8):
        # Y[b, offs_i, k] — [BLOCK_I] uint8
        y_ptrs = Y_ptr + Y_base + offs_i * k8 + k
        y_bytes = tl.load(y_ptrs, mask=offs_i < y_row, other=0)

        # Z[b, offs_j, k] — [BLOCK_J] uint8
        z_ptrs = Z_ptr + Z_base + offs_j * k8 + k
        z_bytes = tl.load(z_ptrs, mask=offs_j < z_col, other=0)

        # AND outer product → [BLOCK_I, BLOCK_J], then popcount
        and_val = y_bytes.to(tl.int32)[:, None] & z_bytes.to(tl.int32)[None, :]
        acc += tl.extra.cuda.libdevice.popc(and_val)

    # Write [B, y_row, z_col] float32 output
    out_base = pid_b * y_row * z_col
    out_ptrs = Out_ptr + out_base + offs_i[:, None] * z_col + offs_j[None, :]
    tl.store(
        out_ptrs,
        acc.to(tl.float32),
        mask=(offs_i[:, None] < y_row) & (offs_j[None, :] < z_col),
    )


def packed_binary_matmul(
    Y_flat: torch.Tensor,
    Z_flat: torch.Tensor,
    y_row: int,
    z_col: int,
    inter_dimension: int,
) -> torch.Tensor:
    """Binary matmul via Triton AND+popcount.

    Parameters
    ----------
    Y_flat : [B, y_row, k8] uint8, contiguous, CUDA
    Z_flat : [B, z_col, k8] uint8, contiguous, CUDA  (Z already transposed)
    y_row, z_col : output dimensions
    inter_dimension : original unpacked inner dim (informational; k8 comes from Y_flat.shape)

    Returns
    -------
    [B, y_row, z_col] float32 CUDA tensor
    """
    assert Y_flat.dtype == torch.uint8, f"Expected uint8, got {Y_flat.dtype}"
    assert Z_flat.dtype == torch.uint8, f"Expected uint8, got {Z_flat.dtype}"
    assert Y_flat.is_contiguous() and Z_flat.is_contiguous(), "Inputs must be contiguous"

    B, _, k8 = Y_flat.shape
    out = torch.empty(B, y_row, z_col, dtype=torch.float32, device=Y_flat.device)

    BLOCK_I = min(triton.next_power_of_2(y_row), 64)
    BLOCK_J = min(triton.next_power_of_2(z_col), 64)
    grid = (B, triton.cdiv(y_row, BLOCK_I), triton.cdiv(z_col, BLOCK_J))

    _packed_binary_matmul_kernel[grid](
        Y_flat, Z_flat, out,
        y_row, z_col, k8,
        BLOCK_I=BLOCK_I,
        BLOCK_J=BLOCK_J,
    )
    return out


# ---------------------------------------------------------------------------
# Phase-1 kernel: T = Z_packed_binary @ X_float
# T[b, n, k] = Σ_j  Z_bool[b, k, j] * X[n, col(b), j]
#
# Z is stored transposed: Z_packed[b, j, k//8] bit (7 - k%8) == Z_bool[b, k, j]
# col(b) = b % col_width
# ---------------------------------------------------------------------------

@triton.jit
def _z_x_kernel(
    Z_ptr,       # [B, z_col, k8]  uint8, contiguous
    X_ptr,       # [batch, col_width, z_col]  float16/bfloat16, contiguous
    T_ptr,       # [B, batch, K8x8]  float32 output  (K8x8 = k8*8 >= inter_dim)
    z_col, k8, col_width, batch,
    BLOCK_J: tl.constexpr,   # >= z_col, power-of-2
):
    """Grid: (B, batch)."""
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    col_b = pid_b % col_width
    j_offs = tl.arange(0, BLOCK_J)
    j_mask = j_offs < z_col

    # X slice for this (n, col_b): [BLOCK_J] floats
    x_f = tl.load(
        X_ptr + pid_n * col_width * z_col + col_b * z_col + j_offs,
        mask=j_mask, other=0.0,
    ).to(tl.float32)

    T_base = T_ptr + (pid_b * batch + pid_n) * k8 * 8

    for byte_k in tl.range(k8):
        # Z_packed[b, :, byte_k]: [BLOCK_J] uint8
        z_b = tl.load(
            Z_ptr + pid_b * z_col * k8 + j_offs * k8 + byte_k,
            mask=j_mask, other=0,
        ).to(tl.int32)

        for bit in tl.static_range(8):
            z_bits = (z_b >> (7 - bit)) & 1          # [BLOCK_J]
            t_val = tl.sum(z_bits.to(tl.float32) * x_f)
            tl.store(T_base + byte_k * 8 + bit, t_val)


def binary_z_x(
    Z_flat: torch.Tensor,   # [B, z_col, k8] uint8, contiguous, CUDA
    X_view: torch.Tensor,   # [batch, col_width, z_col] float, contiguous, CUDA
    inter_dim: int,
    col_width: int,
) -> torch.Tensor:
    """T[b, n, k] = Σ_j Z_bool[b,k,j] * X[n, col(b), j].  Returns [B, batch, k8*8] float32."""
    B, z_col, k8 = Z_flat.shape
    batch = X_view.shape[0]
    K8x8 = k8 * 8

    T = torch.empty(B, batch, K8x8, dtype=torch.float32, device=Z_flat.device)
    BLOCK_J = min(triton.next_power_of_2(z_col), 64)
    _z_x_kernel[(B, batch)](
        Z_flat, X_view, T,
        z_col, k8, col_width, batch,
        BLOCK_J=BLOCK_J,
    )
    return T


# ---------------------------------------------------------------------------
# Phase-2 kernel: core = Y_packed_binary @ T_float
# core[b, n, i] = Σ_k  Y_bool[b, i, k] * T[b, n, k]
# ---------------------------------------------------------------------------

@triton.jit
def _y_t_kernel(
    Y_ptr,    # [B, y_row, k8]  uint8, contiguous
    T_ptr,    # [B, batch, K8x8]  float32, contiguous
    Out_ptr,  # [B, batch, y_row]  float32 output
    y_row, k8, batch,
):
    """Grid: (B, batch, y_row)."""
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_i = tl.program_id(2)

    T_base = T_ptr + (pid_b * batch + pid_n) * k8 * 8
    acc = tl.zeros([], dtype=tl.float32)

    for byte_k in tl.range(k8):
        y_byte = tl.load(Y_ptr + pid_b * y_row * k8 + pid_i * k8 + byte_k).to(tl.int32)
        bit_offs = tl.arange(0, 8)
        t_vals = tl.load(T_base + byte_k * 8 + bit_offs).to(tl.float32)  # [8]
        y_bits = (y_byte >> (7 - bit_offs)) & 1                           # [8]
        acc += tl.sum(y_bits.to(tl.float32) * t_vals)

    tl.store(Out_ptr + (pid_b * batch + pid_n) * y_row + pid_i, acc)


def binary_y_t(
    Y_flat: torch.Tensor,   # [B, y_row, k8] uint8, contiguous, CUDA
    T: torch.Tensor,        # [B, batch, K8x8] float32, contiguous, CUDA
    inter_dim: int,
) -> torch.Tensor:
    """core[b, n, i] = Σ_k Y_bool[b,i,k] * T[b,n,k].  Returns [B, batch, y_row] float32."""
    B, y_row, k8 = Y_flat.shape
    batch = T.shape[1]

    out = torch.empty(B, batch, y_row, dtype=torch.float32, device=Y_flat.device)
    _y_t_kernel[(B, batch, y_row)](
        Y_flat, T, out,
        y_row, k8, batch,
    )
    return out


def register_triton_kernel() -> None:
    """Register packed_binary_matmul as the PackedBinaryQuadratic kernel."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bqq_modules import PackedBinaryQuadratic
    PackedBinaryQuadratic.packed_kernel = staticmethod(packed_binary_matmul)
    PackedBinaryQuadratic.use_packed_kernel = True
    print("[bqq_triton_kernel] Triton AND+popcount kernel registered.")


def register_zy_x_kernel() -> None:
    """Enable the Y@(Z@X) Triton forward path for PackedBinaryQuadratic."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bqq_modules import PackedBinaryQuadratic
    PackedBinaryQuadratic.use_zy_x_kernel = True
    print("[bqq_triton_kernel] ZYX (Y@(Z@X)) kernel path enabled.")


# ---------------------------------------------------------------------------
# Quick correctness test
# ---------------------------------------------------------------------------

def _test():
    """Validate Triton kernel output against numpy reference."""
    import numpy as np

    B, y_row, z_col, inter_dim = 6, 32, 32, 64
    k8 = (inter_dim + 7) // 8

    rng = np.random.default_rng(0)
    Y_np = rng.integers(0, 256, (B, y_row, k8), dtype=np.uint8)
    Z_np = rng.integers(0, 256, (B, z_col, k8), dtype=np.uint8)

    # Reference: naive triple loop
    ref = np.zeros((B, y_row, z_col), dtype=np.int32)
    for b in range(B):
        for i in range(y_row):
            for j in range(z_col):
                for k in range(k8):
                    ref[b, i, j] += bin(int(Y_np[b, i, k]) & int(Z_np[b, j, k])).count("1")

    Y_t = torch.from_numpy(Y_np).cuda()
    Z_t = torch.from_numpy(Z_np).cuda()
    result = packed_binary_matmul(Y_t, Z_t, y_row, z_col, inter_dim).cpu().numpy()

    max_diff = np.abs(ref.astype(np.float32) - result).max()
    assert max_diff == 0, f"Kernel mismatch: max_diff={max_diff}"
    print(f"[bqq_triton_kernel] _test passed  (B={B}, y={y_row}, z={z_col}, k8={k8})")


if __name__ == "__main__":
    _test()
