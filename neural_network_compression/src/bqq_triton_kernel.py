"""Triton kernels for PackedBinaryQuadratic forward pass.

Kernel families
---------------
1. ``packed_binary_matmul``
   AND + popcount for W_core reconstruction.  Used by the default
   ``PackedBinaryQuadratic.forward()`` (W-reconstruction path).

2. ``binary_z_x`` / ``binary_y_t``
   Separate Phase-1 / Phase-2 kernels for the ZYX path (fallback).

3. ``fused_bqq_forward``  (primary ZYX path)
   Fully fused forward that computes
     out = X @ (a*Y@Z + b*Ysum + c*Zsum + d)^T
   in one kernel launch, without materialising W or the intermediate T.
   All four output terms (a*core, b*Ysum*Xsum, c*Zsum*X, d*Xsum) are
   computed concurrently inside each thread block.

Usage::

    from bqq_triton_kernel import register_triton_kernel
    register_triton_kernel()    # AND+popcount for W-reconstruction path

    from bqq_triton_kernel import register_zy_x_kernel
    register_zy_x_kernel()      # enable fused ZYX forward path
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ===================================================================
# 1.  AND + popcount  (W_core reconstruction)
# ===================================================================

@triton.jit
def _packed_binary_matmul_kernel(
    Y_ptr, Z_ptr, Out_ptr,
    y_row, z_col, k8,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """Grid: (B, ceil(y_row/BLOCK_I), ceil(z_col/BLOCK_J))."""
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)

    acc = tl.zeros([BLOCK_I, BLOCK_J], dtype=tl.int32)

    Y_base = pid_b * y_row * k8
    Z_base = pid_b * z_col * k8

    for k in tl.range(k8):
        y_ptrs = Y_ptr + Y_base + offs_i * k8 + k
        y_bytes = tl.load(y_ptrs, mask=offs_i < y_row, other=0)
        z_ptrs = Z_ptr + Z_base + offs_j * k8 + k
        z_bytes = tl.load(z_ptrs, mask=offs_j < z_col, other=0)
        and_val = y_bytes.to(tl.int32)[:, None] & z_bytes.to(tl.int32)[None, :]
        acc += tl.extra.cuda.libdevice.popc(and_val)

    out_base = pid_b * y_row * z_col
    out_ptrs = Out_ptr + out_base + offs_i[:, None] * z_col + offs_j[None, :]
    tl.store(
        out_ptrs, acc.to(tl.float32),
        mask=(offs_i[:, None] < y_row) & (offs_j[None, :] < z_col),
    )


def packed_binary_matmul(
    Y_flat: torch.Tensor,
    Z_flat: torch.Tensor,
    y_row: int,
    z_col: int,
    inter_dimension: int,
) -> torch.Tensor:
    """Binary matmul via Triton AND+popcount.  Returns [B, y_row, z_col] fp32."""
    assert Y_flat.dtype == torch.uint8, f"Expected uint8, got {Y_flat.dtype}"
    assert Z_flat.dtype == torch.uint8, f"Expected uint8, got {Z_flat.dtype}"
    assert Y_flat.is_contiguous() and Z_flat.is_contiguous()

    B, _, k8 = Y_flat.shape
    out = torch.empty(B, y_row, z_col, dtype=torch.float32, device=Y_flat.device)

    BLOCK_I = min(triton.next_power_of_2(y_row), 64)
    BLOCK_J = min(triton.next_power_of_2(z_col), 64)
    grid = (B, triton.cdiv(y_row, BLOCK_I), triton.cdiv(z_col, BLOCK_J))

    _packed_binary_matmul_kernel[grid](
        Y_flat, Z_flat, out, y_row, z_col, k8,
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J,
    )
    return out


# ===================================================================
# 2.  Separate Phase-1 / Phase-2  (ZYX fallback)
# ===================================================================

@triton.jit
def _z_x_kernel(
    Z_ptr,       # [B, z_col, k8]  uint8
    X_ptr,       # [batch, col_width, z_col]  float32
    T_ptr,       # [B, batch, K_padded]  float32 output
    z_col, k8, col_width, batch, K_padded,
    BLOCK_J: tl.constexpr,
):
    """T[b,n,k] = Sigma_j Z_bool[b,k,j] * X[n,col(b),j].  Grid: (B, batch)."""
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    col_b = pid_b % col_width

    j_offs = tl.arange(0, BLOCK_J)
    j_mask = j_offs < z_col

    x_f = tl.load(
        X_ptr + pid_n * col_width * z_col + col_b * z_col + j_offs,
        mask=j_mask, other=0.0,
    ).to(tl.float32)

    T_base = T_ptr + (pid_b * batch + pid_n) * K_padded

    for byte_k in tl.range(k8):
        z_b = tl.load(
            Z_ptr + pid_b * z_col * k8 + j_offs * k8 + byte_k,
            mask=j_mask, other=0,
        ).to(tl.int32)
        for bit in tl.static_range(8):
            z_bits = (z_b >> (7 - bit)) & 1
            t_val = tl.sum(z_bits.to(tl.float32) * x_f)
            tl.store(T_base + byte_k * 8 + bit, t_val)


def binary_z_x(
    Z_flat: torch.Tensor,
    X_view: torch.Tensor,
    inter_dim: int,
    col_width: int,
) -> torch.Tensor:
    """Phase 1: T = Z_bool @ X.  Returns [B, batch, k8*8] fp32."""
    B, z_col, k8 = Z_flat.shape
    batch = X_view.shape[0]
    K_padded = k8 * 8

    T = torch.empty(B, batch, K_padded, dtype=torch.float32, device=Z_flat.device)
    # Use next_power_of_2 (not capped at 64) so BLOCK_J always >= z_col
    BLOCK_J = triton.next_power_of_2(z_col)

    _z_x_kernel[(B, batch)](
        Z_flat, X_view.float().contiguous(), T,
        z_col, k8, col_width, batch, K_padded,
        BLOCK_J=BLOCK_J,
    )
    return T


@triton.jit
def _y_t_kernel(
    Y_ptr,    # [B, y_row, k8]  uint8
    T_ptr,    # [B, batch, K_padded]  float32
    Out_ptr,  # [B, batch, y_row]  float32
    y_row, k8, batch, K_padded,
    BLOCK_I: tl.constexpr,
):
    """core[b,n,i] = Sigma_k Y_bool[b,i,k] * T[b,n,k].
    Grid: (B, batch, ceil(y_row/BLOCK_I))."""
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_tile = tl.program_id(2)

    i_offs = pid_tile * BLOCK_I + tl.arange(0, BLOCK_I)
    i_mask = i_offs < y_row

    T_base = T_ptr + (pid_b * batch + pid_n) * K_padded
    acc = tl.zeros([BLOCK_I], dtype=tl.float32)

    for byte_k in tl.range(k8):
        y_bytes = tl.load(
            Y_ptr + pid_b * y_row * k8 + i_offs * k8 + byte_k,
            mask=i_mask, other=0,
        ).to(tl.int32)
        for bit in tl.static_range(8):
            t_val = tl.load(T_base + byte_k * 8 + bit)
            y_bits = (y_bytes >> (7 - bit)) & 1
            acc += y_bits.to(tl.float32) * t_val

    out_offs = (pid_b * batch + pid_n) * y_row + i_offs
    tl.store(Out_ptr + out_offs, acc, mask=i_mask)


def binary_y_t(
    Y_flat: torch.Tensor,
    T: torch.Tensor,
    inter_dim: int,
) -> torch.Tensor:
    """Phase 2: core = Y_bool @ T.  Returns [B, batch, y_row] fp32."""
    B, y_row, k8 = Y_flat.shape
    batch = T.shape[1]
    K_padded = T.shape[2]

    out = torch.empty(B, batch, y_row, dtype=torch.float32, device=Y_flat.device)
    BLOCK_I = min(triton.next_power_of_2(y_row), 64)

    _y_t_kernel[(B, batch, triton.cdiv(y_row, BLOCK_I))](
        Y_flat, T, out,
        y_row, k8, batch, K_padded,
        BLOCK_I=BLOCK_I,
    )
    return out


# ===================================================================
# 3.  Fused BQQ forward  (primary ZYX path)
# ===================================================================
#
# Key insight: Ysum = Y@1 and Zsum@x = sum(Z@x), so all four terms
# can be computed from the same Z@x and Y@t operations:
#
#   t[k] = Z_bool[k,:] @ x            (Phase 1: binary Z times float x)
#   t_aug[k] = a * t[k] + b * xsum    (fold b term into t)
#   acc[i] += Y_bool[i,:] @ t_aug     (Phase 2: fuses terms 1+2)
#   acc[i] += c * sum(t)              (term 3: free from Phase 1 sum)
#   acc[i] += d * xsum                (term 4: scalar broadcast)
#
# Derivation:
#   Y@t_aug = Y @ (a*t + b*xsum*1) = a*(Y@t) + b*xsum*(Y@1)
#           = a*core + b*Ysum*xsum    (terms 1+2 combined!)
#   sum(t) = 1^T @ Z @ x = Zsum @ x  (term 3 for free!)
#
# No Y_sum_eff or Z_sum_eff precomputation needed.
# No intermediate T or W_core tensors materialised.

@triton.jit
def _fused_bqq_forward_kernel(
    # Packed binary matrices
    Y_ptr,              # [B_total, y_row, k8]          uint8
    Z_ptr,              # [B_total, z_col, k8]          uint8
    # Input activations
    X_ptr,              # [batch, col_width, z_col]     float32
    # Flattened coefficients
    a_ptr,              # [B_total]                     float32
    b_ptr,              # [B_total]                     float32
    c_ptr,              # [B_total]                     float32
    d_ptr,              # [row_width * col_width]       float32
    # Output
    Out_ptr,            # [batch, row_width, y_row]     float32
    # Scalar dimensions
    batch, row_width, col_width, bit_width,
    y_row, z_col, k8,
    # Tile sizes (constexpr)
    BLOCK_I: tl.constexpr,      # tile size for y_row
    BLOCK_J: tl.constexpr,      # >= z_col (power-of-2)
):
    """Fused BQQ forward.  Grid: (row_width, batch, ceil(y_row / BLOCK_I))."""
    pid_r = tl.program_id(0)       # row block index
    pid_n = tl.program_id(1)       # batch index
    pid_tile = tl.program_id(2)    # y_row tile index

    i_offs = pid_tile * BLOCK_I + tl.arange(0, BLOCK_I)    # [BLOCK_I]
    i_mask = i_offs < y_row
    j_offs = tl.arange(0, BLOCK_J)                          # [BLOCK_J]
    j_mask = j_offs < z_col

    acc = tl.zeros([BLOCK_I], dtype=tl.float32)

    for c_idx in tl.range(col_width):
        rc_idx = pid_r * col_width + c_idx

        # --- Load X[n, c, :] (reused across all bit_width iterations) ---
        x_vals = tl.load(
            X_ptr + pid_n * col_width * z_col + c_idx * z_col + j_offs,
            mask=j_mask, other=0.0,
        ).to(tl.float32)
        x_col_sum = tl.sum(x_vals)

        # --- Terms 1+2+3 fused per bit_width ---
        for p in tl.range(bit_width):
            B_idx = p * row_width * col_width + rc_idx
            a_val = tl.load(a_ptr + B_idx).to(tl.float32)
            b_val = tl.load(b_ptr + B_idx).to(tl.float32)
            c_val = tl.load(c_ptr + B_idx).to(tl.float32)

            t_sum = tl.zeros([], dtype=tl.float32)      # sum(t) for term 3

            for byte_k in tl.range(k8):
                # Z_packed[B_idx, :, byte_k] -> [BLOCK_J] uint8
                z_bytes = tl.load(
                    Z_ptr + B_idx * z_col * k8 + j_offs * k8 + byte_k,
                    mask=j_mask, other=0,
                ).to(tl.int32)
                # Y_packed[B_idx, i_offs, byte_k] -> [BLOCK_I] uint8
                y_bytes = tl.load(
                    Y_ptr + B_idx * y_row * k8 + i_offs * k8 + byte_k,
                    mask=i_mask, other=0,
                ).to(tl.int32)

                for bit in tl.static_range(8):
                    # Phase 1: t_k = Z_bool[k,:] @ x  (scalar)
                    z_bits = (z_bytes >> (7 - bit)) & 1
                    t_k = tl.sum(z_bits.to(tl.float32) * x_vals)

                    # Phase 2 fused: Y @ (a*t + b*xsum) = a*core + b*Ysum*xsum
                    t_aug_k = a_val * t_k + b_val * x_col_sum
                    y_bits = (y_bytes >> (7 - bit)) & 1
                    acc += y_bits.to(tl.float32) * t_aug_k  # terms 1+2

                    # Accumulate sum(t) for term 3
                    t_sum += t_k

            # Term 3: c * sum(t) = c * Zsum @ x  (scalar broadcast)
            acc += c_val * t_sum

        # --- Term 4: d * x_col_sum (once per c, outside p loop) ---
        acc += tl.load(d_ptr + rc_idx).to(tl.float32) * x_col_sum

    # --- Store output ---
    out_offs = (pid_n * row_width + pid_r) * y_row + i_offs
    tl.store(Out_ptr + out_offs, acc, mask=i_mask)


def fused_bqq_forward(
    Y_packed: torch.Tensor,
    Z_packed: torch.Tensor,
    X: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fully-fused BQQ forward.

    Computes  ``X @ W.T + bias``  where
    ``W = sum_p (a_p * Y_p@Z_p  +  b_p * Ysum_p  +  c_p * Zsum_p) + d``
    without materialising W, T, Y_sum_eff, or Z_sum_eff.

    The b term is folded into the Y multiplication via
    ``t_aug[k] = a*t[k] + b*xsum``, and the c term is obtained for free
    as ``c * sum(t) = c * Zsum @ x``.

    Parameters
    ----------
    Y_packed : [bit, row, col, y_row, k8]  uint8
    Z_packed : [bit, row, col, z_col, k8]  uint8
    X        : [..., in_features]           any float dtype
    a, b, c  : [bit, row, col, 1, 1]       float
    d        : [row, col, 1, 1]             float
    bias     : [out_features] or None

    Returns
    -------
    [..., out_features]  same dtype as X
    """
    bit_width, row_width, col_width, y_row, k8 = Y_packed.shape
    z_col = Z_packed.shape[3]
    B_total = bit_width * row_width * col_width
    orig_dtype = X.dtype
    device = Y_packed.device

    # Flatten leading dims of X
    orig_shape = X.shape
    X_2d = X.to(device).reshape(-1, orig_shape[-1])
    batch = X_2d.shape[0]

    # Reshape packed matrices to [B_total, ...]
    Y_flat = Y_packed.reshape(B_total, y_row, k8).contiguous()
    Z_flat = Z_packed.reshape(B_total, z_col, k8).contiguous()
    X_view = X_2d.reshape(batch, col_width, z_col).float().contiguous()

    # Flatten coefficients to [B_total]
    a_flat = a.squeeze(-1).squeeze(-1).reshape(B_total).float().contiguous()
    b_flat = b.squeeze(-1).squeeze(-1).reshape(B_total).float().contiguous()
    c_flat = c.squeeze(-1).squeeze(-1).reshape(B_total).float().contiguous()
    d_flat = d.squeeze(-1).squeeze(-1).reshape(
        row_width * col_width).float().contiguous()

    # Allocate output
    out = torch.empty(batch, row_width, y_row, dtype=torch.float32, device=device)

    BLOCK_I = min(triton.next_power_of_2(y_row), 64)
    BLOCK_J = triton.next_power_of_2(z_col)
    grid = (row_width, batch, triton.cdiv(y_row, BLOCK_I))

    _fused_bqq_forward_kernel[grid](
        Y_flat, Z_flat, X_view,
        a_flat, b_flat, c_flat, d_flat,
        out,
        batch, row_width, col_width, bit_width,
        y_row, z_col, k8,
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J,
    )

    # Reshape to [..., out_features]
    out = out.reshape(batch, row_width * y_row)
    if bias is not None:
        out = out + bias.float().to(device)

    return out.reshape(*orig_shape[:-1], row_width * y_row).to(orig_dtype)


# ===================================================================
# Registration helpers
# ===================================================================

def register_triton_kernel() -> None:
    """Register packed_binary_matmul as the PackedBinaryQuadratic kernel."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bqq_modules import PackedBinaryQuadratic
    PackedBinaryQuadratic.packed_kernel = staticmethod(packed_binary_matmul)
    PackedBinaryQuadratic.use_packed_kernel = True
    print("[bqq_triton_kernel] Triton AND+popcount kernel registered.")


def register_zy_x_kernel() -> None:
    """Enable the fused ZYX forward path for PackedBinaryQuadratic."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bqq_modules import PackedBinaryQuadratic
    PackedBinaryQuadratic.use_zy_x_kernel = True
    print("[bqq_triton_kernel] Fused ZYX forward path enabled.")


# ===================================================================
# Correctness tests
# ===================================================================

def _test_popcount():
    """Validate AND+popcount kernel against numpy reference."""
    import numpy as np

    B, y_row, z_col, inter_dim = 6, 32, 32, 64
    k8 = (inter_dim + 7) // 8

    rng = np.random.default_rng(0)
    Y_np = rng.integers(0, 256, (B, y_row, k8), dtype=np.uint8)
    Z_np = rng.integers(0, 256, (B, z_col, k8), dtype=np.uint8)

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
    print(f"[popcount] PASSED  (B={B}, y={y_row}, z={z_col}, k8={k8})")


def _test_fused():
    """Validate fused forward against W-reconstruction reference."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bqq_modules import BinaryQuadratic, PackedBinaryQuadratic
    import torch.nn as nn

    torch.manual_seed(42)
    bit_width, row_width, col_width = 4, 8, 8
    y_row, z_col, inter_dim = 32, 32, 32

    # Create random BinaryQuadratic
    Y = torch.randint(0, 2, (bit_width, row_width, col_width, y_row, inter_dim)).bool()
    Z = torch.randint(0, 2, (bit_width, row_width, col_width, inter_dim, z_col)).bool()
    A = torch.randn(bit_width, row_width, col_width, 4)

    bq = BinaryQuadratic.__new__(BinaryQuadratic)
    nn.Module.__init__(bq)
    bq.bit_width = bit_width
    bq.row_width = row_width
    bq.col_width = col_width
    bq.y_row = y_row
    bq.inter_dimension = inter_dim
    bq.z_col = z_col
    bq.register_buffer('Y', Y)
    bq.register_buffer('Z', Z)
    bq.a = nn.Parameter(A[..., 0].unsqueeze(-1).unsqueeze(-1))
    bq.b = nn.Parameter(A[..., 1].unsqueeze(-1).unsqueeze(-1))
    bq.c = nn.Parameter(A[..., 2].unsqueeze(-1).unsqueeze(-1))
    bq.d = nn.Parameter(A[..., 3].unsqueeze(-1).unsqueeze(-1).sum(dim=0))
    bq.bias = None

    packed = PackedBinaryQuadratic.from_unpacked(bq).cuda()

    X = torch.randn(2, col_width * z_col, dtype=torch.float32).cuda()

    # Reference: W-reconstruction forward (use_zy_x_kernel must be False)
    old_flag = PackedBinaryQuadratic.use_zy_x_kernel
    PackedBinaryQuadratic.use_zy_x_kernel = False
    ref = packed.forward(X)
    PackedBinaryQuadratic.use_zy_x_kernel = old_flag

    # Fused forward
    fused = fused_bqq_forward(
        packed.Y_packed, packed.Z_packed, X,
        packed.a, packed.b, packed.c, packed.d,
        bias=packed.bias,
    )

    max_diff = (ref.float() - fused.float()).abs().max().item()
    ref_max = ref.float().abs().max().item()
    rel_diff = max_diff / max(ref_max, 1e-8)
    print(f"[fused] max_diff={max_diff:.6e}, rel_diff={rel_diff:.6e}")
    assert rel_diff < 1e-4, f"Fused kernel mismatch: rel_diff={rel_diff}"
    print(f"[fused] PASSED  (bit={bit_width}, row={row_width}, col={col_width}, "
          f"yr={y_row}, zc={z_col}, batch=2)")


def _test_fused_with_bias():
    """Validate fused forward with bias."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bqq_modules import BinaryQuadratic, PackedBinaryQuadratic
    import torch.nn as nn

    torch.manual_seed(123)
    bit_width, row_width, col_width = 2, 4, 4
    y_row, z_col, inter_dim = 16, 16, 16

    Y = torch.randint(0, 2, (bit_width, row_width, col_width, y_row, inter_dim)).bool()
    Z = torch.randint(0, 2, (bit_width, row_width, col_width, inter_dim, z_col)).bool()
    A = torch.randn(bit_width, row_width, col_width, 4)
    bias = torch.randn(row_width * y_row)

    bq = BinaryQuadratic.__new__(BinaryQuadratic)
    nn.Module.__init__(bq)
    bq.bit_width = bit_width
    bq.row_width = row_width
    bq.col_width = col_width
    bq.y_row = y_row
    bq.inter_dimension = inter_dim
    bq.z_col = z_col
    bq.register_buffer('Y', Y)
    bq.register_buffer('Z', Z)
    bq.a = nn.Parameter(A[..., 0].unsqueeze(-1).unsqueeze(-1))
    bq.b = nn.Parameter(A[..., 1].unsqueeze(-1).unsqueeze(-1))
    bq.c = nn.Parameter(A[..., 2].unsqueeze(-1).unsqueeze(-1))
    bq.d = nn.Parameter(A[..., 3].unsqueeze(-1).unsqueeze(-1).sum(dim=0))
    bq.bias = nn.Parameter(bias)

    packed = PackedBinaryQuadratic.from_unpacked(bq).cuda()

    X = torch.randn(4, 8, col_width * z_col, dtype=torch.float32).cuda()

    old_flag = PackedBinaryQuadratic.use_zy_x_kernel
    PackedBinaryQuadratic.use_zy_x_kernel = False
    ref = packed.forward(X)
    PackedBinaryQuadratic.use_zy_x_kernel = old_flag

    fused = fused_bqq_forward(
        packed.Y_packed, packed.Z_packed, X,
        packed.a, packed.b, packed.c, packed.d,
        bias=packed.bias,
    )

    max_diff = (ref.float() - fused.float()).abs().max().item()
    ref_max = ref.float().abs().max().item()
    rel_diff = max_diff / max(ref_max, 1e-8)
    print(f"[fused+bias] max_diff={max_diff:.6e}, rel_diff={rel_diff:.6e}")
    assert rel_diff < 1e-4, f"Fused kernel mismatch (bias): rel_diff={rel_diff}"
    print(f"[fused+bias] PASSED  (bit={bit_width}, row={row_width}, col={col_width}, "
          f"yr={y_row}, zc={z_col}, batch=(4,8))")


if __name__ == "__main__":
    _test_popcount()
    _test_fused()
    _test_fused_with_bias()
