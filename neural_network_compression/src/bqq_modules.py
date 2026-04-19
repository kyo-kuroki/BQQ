"""
BQQ (Binary Quadratic Quantization) module definitions.

Shared between LM and CV workflows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np


# ---------------------------------------------------------------------------
# Core BQQ modules (shared)
# ---------------------------------------------------------------------------

class BinaryQuadratic(nn.Module):
    """BQQ layer using {0,1} binary representation."""

    def __init__(self, Y, Z, A, bias=None):
        super().__init__()
        self.bit_width, self.row_width, self.col_width, self.y_row, self.inter_dimension = Y.shape
        _, _, _, _, self.z_col = Z.shape

        self.register_buffer("Y", (Y > 0.5))
        self.register_buffer("Z", (Z > 0.5))
        self.a = nn.Parameter(A[..., 0].unsqueeze(-1).unsqueeze(-1).clone())
        self.b = nn.Parameter(A[..., 1].unsqueeze(-1).unsqueeze(-1).clone())
        self.c = nn.Parameter(A[..., 2].unsqueeze(-1).unsqueeze(-1).clone())
        self.d = nn.Parameter(A[..., 3].unsqueeze(-1).unsqueeze(-1).sum(dim=0))
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, X):
        dtype = X.dtype
        device = self.Y.device
        W_core = torch.matmul(self.Y.type(dtype), self.Z.type(dtype))
        Y_sum = self.Y.sum(dim=-1, keepdim=True).type(dtype)
        Z_sum = self.Z.sum(dim=-2, keepdim=True).type(dtype)
        W = self.a.type(dtype) * W_core + self.b.type(dtype) * Y_sum + self.c.type(dtype) * Z_sum
        W = W.sum(dim=0) + self.d.type(dtype)
        W = W.permute(0, 2, 1, 3).reshape(self.row_width * self.y_row, self.col_width * self.z_col)
        if self.bias is None:
            return X.to(device) @ W.T
        else:
            return X.to(device) @ W.T + self.bias.type(dtype).to(device)

    def get_weight(self, dtype=torch.float32):
        W_core = torch.matmul(self.Y.type(dtype), self.Z.type(dtype))
        Y_sum = self.Y.sum(dim=-1, keepdim=True).type(dtype)
        Z_sum = self.Z.sum(dim=-2, keepdim=True).type(dtype)
        W = self.a.type(dtype) * W_core + self.b.type(dtype) * Y_sum + self.c.type(dtype) * Z_sum
        W = W.sum(dim=0) + self.d.type(dtype)
        W = W.permute(0, 2, 1, 3).reshape(self.row_width * self.y_row, self.col_width * self.z_col)
        return W


# ---------------------------------------------------------------------------
# Packed BQQ module (bit-packed Y/Z for 8x memory reduction)
# ---------------------------------------------------------------------------

class PackedBinaryQuadratic(nn.Module):
    """BQQ layer with Y and Z stored as packed uint8 (8x smaller than bool).

    Forward uses integer AND + popcount to compute W_core without unpacking
    to float, keeping packed bits in memory throughout.

    Storage layout
    --------------
    Y_packed : [bit_width, row_width, col_width, y_row, ceil(inter_dim/8)]  uint8
    Z_packed : [bit_width, row_width, col_width, z_col, ceil(inter_dim/8)]  uint8
                (Z is transposed before packing so inter_dim is the last axis)
    Y_sum_i16: [bit_width, row_width, col_width, y_row, 1]  int16  (precomputed)
    Z_sum_i16: [bit_width, row_width, col_width, 1, z_col]  int16  (precomputed)
    """

    def __init__(self, Y_packed, Z_packed, a, b, c, d,
                 Y_sum_i16, Z_sum_i16,
                 inter_dimension, y_row, z_col,
                 bias=None):
        super().__init__()
        self.bit_width, self.row_width, self.col_width = Y_packed.shape[:3]
        self.y_row = y_row
        self.z_col = z_col
        self.inter_dimension = inter_dimension
        self._k8 = Y_packed.shape[-1]  # ceil(inter_dimension / 8)

        self.register_buffer("Y_packed", Y_packed)
        self.register_buffer("Z_packed", Z_packed)
        self.register_buffer("Y_sum_i16", Y_sum_i16)
        self.register_buffer("Z_sum_i16", Z_sum_i16)

        self.a = nn.Parameter(a)
        self.b = nn.Parameter(b)
        self.c = nn.Parameter(c)
        self.d = nn.Parameter(d)
        self.bias = nn.Parameter(bias) if bias is not None else None

    @staticmethod
    def _np_packbits(t: torch.Tensor, last_dim_size: int) -> torch.Tensor:
        """Pack a bool tensor along its last axis using numpy packbits.

        t must be on CPU and have last-dimension size == last_dim_size.
        Returns a uint8 tensor with last dimension == ceil(last_dim_size / 8).
        numpy packbits uses big-endian ordering (first element → MSB), which
        matches the ordering assumed by _popcount_uint8 in forward.
        """
        arr = t.numpy().astype(np.uint8)          # flatten to numpy
        flat = arr.reshape(-1, last_dim_size)
        packed_flat = np.packbits(flat, axis=-1)  # big-endian, pads with 0
        k8 = packed_flat.shape[-1]
        result = torch.from_numpy(packed_flat).reshape(*t.shape[:-1], k8)
        return result.contiguous()

    @classmethod
    def from_unpacked(cls, bq: 'BinaryQuadratic') -> 'PackedBinaryQuadratic':
        """Convert a BinaryQuadratic layer to PackedBinaryQuadratic."""
        Y = bq.Y.cpu()  # [B, r, c, y_row, inter_dim] bool
        Z = bq.Z.cpu()  # [B, r, c, inter_dim, z_col] bool

        bit_width, row_width, col_width, y_row, inter_dim = Y.shape
        z_col = Z.shape[-1]

        # Pack Y along inter_dimension (last axis)
        Y_packed = cls._np_packbits(Y, inter_dim)  # [..., y_row, ceil(inter_dim/8)]

        # Pack Z along inter_dimension (axis -2); transpose first so inter_dim is last
        Z_t = Z.permute(0, 1, 2, 4, 3).contiguous()  # [..., z_col, inter_dim]
        Z_packed = cls._np_packbits(Z_t, inter_dim)   # [..., z_col, ceil(inter_dim/8)]

        # Precompute sums as int16 (values 0..inter_dim; fits in int16 for any realistic rank)
        Y_sum = Y.sum(dim=-1, keepdim=True).to(torch.int16)   # [..., y_row, 1]
        Z_sum = Z.sum(dim=-2, keepdim=True).to(torch.int16)   # [..., 1, z_col]

        return cls(
            Y_packed=Y_packed,
            Z_packed=Z_packed,
            a=bq.a.data.clone(),
            b=bq.b.data.clone(),
            c=bq.c.data.clone(),
            d=bq.d.data.clone(),
            Y_sum_i16=Y_sum,
            Z_sum_i16=Z_sum,
            inter_dimension=inter_dim,
            y_row=y_row,
            z_col=z_col,
            bias=bq.bias.data.clone() if bq.bias is not None else None,
        )

    # ------------------------------------------------------------------
    # Kernel selection flag.
    # Set PackedBinaryQuadratic.use_packed_kernel = True once a custom
    # CUDA kernel (AND + popcount binary matmul) is registered.
    # When False (default), forward unpacks to bool and uses cuBLAS.
    # ------------------------------------------------------------------
    use_packed_kernel: bool = False

    # ------------------------------------------------------------------
    # ZYX kernel flag.
    # When True, forward uses Y@(Z@X) via Triton binary-float kernels
    # (avoids materialising W_core; two binary-masked float accumulations).
    # Registered automatically when bqq_triton_kernel is imported.
    # ------------------------------------------------------------------
    use_zy_x_kernel: bool = False

    def _unpack_to_bool(self, packed: torch.Tensor, n_bits: int) -> torch.Tensor:
        """Unpack uint8 tensor to bool along the last axis.

        packed : [..., k8]  uint8, big-endian bit order (MSB = index 0)
        returns: [..., n_bits] bool  (trailing padding bits are dropped)
        """
        # shifts: [8] tensor – extracts each bit from MSB to LSB
        shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=packed.device)
        # [..., k8, 8] → [..., k8*8] → [..., n_bits]
        unpacked = ((packed.unsqueeze(-1) >> shifts) & 1).reshape(*packed.shape[:-1], -1)
        return unpacked[..., :n_bits].bool()

    def _matmul_via_unpack(self, dtype: torch.dtype) -> torch.Tensor:
        """Unpack Y/Z to bool, cast to dtype, use cuBLAS matmul.

        Returns [bit_width, row_width, col_width, y_row, z_col].
        """
        Y = self._unpack_to_bool(self.Y_packed, self.inter_dimension)
        # Z was stored transposed as [..., z_col, inter_dim]; transpose back
        Z_t = self._unpack_to_bool(self.Z_packed, self.inter_dimension)
        # Y: [..., y_row, inter_dim]  Z_t: [..., z_col, inter_dim]
        return torch.matmul(Y.to(dtype), Z_t.to(dtype).transpose(-2, -1))

    def _matmul_via_packed_kernel(self, dtype: torch.dtype) -> torch.Tensor:
        """Binary matmul using custom CUDA AND+popcount kernel (not yet implemented).

        Placeholder: raises NotImplementedError until a kernel is registered.
        To add a kernel, assign a callable to PackedBinaryQuadratic.packed_kernel:

            PackedBinaryQuadratic.packed_kernel = my_kernel_fn
            PackedBinaryQuadratic.use_packed_kernel = True

        The kernel must accept (Y_packed, Z_packed, y_row, z_col, inter_dim)
        and return a float tensor of shape [B, y_row, z_col] where
        B = bit_width * row_width * col_width.
        """
        if not hasattr(PackedBinaryQuadratic, 'packed_kernel'):
            raise NotImplementedError(
                "Set PackedBinaryQuadratic.packed_kernel and use_packed_kernel=True "
                "after loading the custom CUDA extension."
            )
        B = self.bit_width * self.row_width * self.col_width
        Y_flat = self.Y_packed.reshape(B, self.y_row, self._k8)
        Z_flat = self.Z_packed.reshape(B, self.z_col, self._k8)
        W_core = PackedBinaryQuadratic.packed_kernel(
            Y_flat, Z_flat, self.y_row, self.z_col, self.inter_dimension
        ).to(dtype)
        return W_core.reshape(
            self.bit_width, self.row_width, self.col_width, self.y_row, self.z_col
        )

    def _compute_W_core(self, dtype: torch.dtype) -> torch.Tensor:
        if PackedBinaryQuadratic.use_packed_kernel:
            return self._matmul_via_packed_kernel(dtype)
        return self._matmul_via_unpack(dtype)

    # Kernel selection:
    #   "cuda"   — warp-level CUDA kernel (fastest, requires compilation)
    #   "triton" — Triton fused kernel (v1/v2 auto-selected)
    #   "recon"  — W-reconstruction + cuBLAS (fallback)
    #   "auto"   — try cuda → triton → recon
    zy_x_kernel: str = "auto"
    # Batch threshold: above this, W-reconstruction is used
    zy_x_recon_threshold: int = 64

    def _forward_zy_x(self, X: torch.Tensor) -> torch.Tensor:
        """Auto-select fastest kernel."""
        batch = X.reshape(-1, X.shape[-1]).shape[0]
        kernel = self.zy_x_kernel

        if kernel == "recon" or (kernel == "auto" and batch > self.zy_x_recon_threshold):
            return self._forward_w_recon(X)

        # Try CUDA warp kernel
        if kernel in ("cuda", "auto"):
            try:
                from bqq_cuda_ext import cuda_bqq_forward
                return cuda_bqq_forward(
                    self.Y_packed, self.Z_packed,
                    X.to(self.Y_packed.device),
                    self.a, self.b, self.c, self.d,
                    bias=self.bias,
                )
            except Exception:
                if kernel == "cuda":
                    raise
                # auto mode: fall through to triton

        # Triton fallback
        if kernel in ("triton", "auto"):
            from bqq_triton_kernel import fused_bqq_forward
            version = 1 if batch <= 1 else 2
            return fused_bqq_forward(
                self.Y_packed, self.Z_packed,
                X.to(self.Y_packed.device),
                self.a, self.b, self.c, self.d,
                bias=self.bias,
                kernel_version=version,
            )

        return self._forward_w_recon(X)

    def _forward_w_recon(self, X: torch.Tensor) -> torch.Tensor:
        """W-reconstruction forward: build W then use cuBLAS for X @ W.T."""
        dtype = X.dtype
        device = self.Y_packed.device

        W_core = self._compute_W_core(dtype)
        Y_sum = self.Y_sum_i16.to(dtype)
        Z_sum = self.Z_sum_i16.to(dtype)

        W = (self.a.to(dtype) * W_core
             + self.b.to(dtype) * Y_sum
             + self.c.to(dtype) * Z_sum)
        W = W.sum(dim=0) + self.d.to(dtype)
        W = W.permute(0, 2, 1, 3).reshape(
            self.row_width * self.y_row, self.col_width * self.z_col
        )

        if self.bias is None:
            return X.to(device) @ W.T
        else:
            return X.to(device) @ W.T + self.bias.to(dtype=dtype, device=device)

    def forward(self, X):
        if PackedBinaryQuadratic.use_zy_x_kernel:
            return self._forward_zy_x(X)
        return self._forward_w_recon(X)

    def get_weight(self, dtype=torch.float32):
        W_core = self._compute_W_core(dtype)
        Y_sum = self.Y_sum_i16.to(dtype)
        Z_sum = self.Z_sum_i16.to(dtype)
        W = (self.a.to(dtype) * W_core
             + self.b.to(dtype) * Y_sum
             + self.c.to(dtype) * Z_sum)
        W = W.sum(dim=0) + self.d.to(dtype)
        W = W.permute(0, 2, 1, 3).reshape(
            self.row_width * self.y_row, self.col_width * self.z_col
        )
        return W


class PartialBQQLinear(nn.Module):
    """Mixed-precision linear layer for progressive patch-wise quantization.

    Patches are quantized incrementally:
      - Quantized patches: Y, Z stored as bool buffers (frozen);
        a, b, c, d are nn.Parameters (trainable).
      - Unquantized patches: represented by float_weight (trainable nn.Parameter).

    Forward assembles W via::

        W = torch.where(mask_full, W_bqq, float_weight)

    so gradients automatically flow to a/b/c/d for quantized patches and to
    float_weight for unquantized patches — no manual gradient masking needed.

    Workflow
    --------
    1. Create from an nn.Linear.
    2. Repeatedly call quantize_patch(i, j, A_ij, Y_ij, Z_ij) for batches of
       patches, interleaved with block-level MSE fine-tuning.
    3. When all patches are done, call to_binaryquadratic() to obtain a
       standard BinaryQuadratic module (float_weight is discarded).
    """

    def __init__(self, weight: torch.Tensor, bias, group_size: int, bit_width: int):
        super().__init__()
        out_features, in_features = weight.shape

        # dim must be exactly divisible by group_size.
        # (dim == group_size is the degenerate case: 1 patch of size gs×gs.)
        def _patch_dims(dim, gs, name):
            if dim % gs != 0:
                raise ValueError(
                    f"{name}={dim} is not divisible by group_size={gs}."
                )
            return dim // gs, gs

        self.group_size = group_size
        self.bit_width = bit_width
        self.row_width, self.y_row = _patch_dims(out_features, group_size, 'out_features')
        self.col_width, self.z_col = _patch_dims(in_features,  group_size, 'in_features')
        self.inter_dimension = None  # set lazily on first quantize_patch call

        # Full float weight — trainable for unquantized patches
        self.float_weight = nn.Parameter(weight.clone().float())
        self.bias_param = nn.Parameter(bias.clone().float()) if bias is not None else None

        # quantized_mask[i, j] = True iff patch (i, j) has been BQQ-quantized
        self.register_buffer(
            'quantized_mask',
            torch.zeros(self.row_width, self.col_width, dtype=torch.bool),
        )
        # Y, Z buffers and a/b/c/d params are registered lazily in _init_bqq_tensors()
        # once inter_dimension is known from the first quantize_patch call.

    # ------------------------------------------------------------------
    # Lazy initialisation of BQQ tensors
    # ------------------------------------------------------------------

    def _init_bqq_tensors(self, inter_dimension: int) -> None:
        bw = self.bit_width
        rw, cw = self.row_width, self.col_width
        yr, zc = self.y_row, self.z_col
        dev = self.float_weight.device

        self.inter_dimension = inter_dimension
        self.register_buffer(
            'Y', torch.zeros(bw, rw, cw, yr, inter_dimension, dtype=torch.bool, device=dev)
        )
        self.register_buffer(
            'Z', torch.zeros(bw, rw, cw, inter_dimension, zc, dtype=torch.bool, device=dev)
        )
        self.a = nn.Parameter(torch.zeros(bw, rw, cw, 1, 1, device=dev))
        self.b = nn.Parameter(torch.zeros(bw, rw, cw, 1, 1, device=dev))
        self.c = nn.Parameter(torch.zeros(bw, rw, cw, 1, 1, device=dev))
        self.d = nn.Parameter(torch.zeros(rw, cw, 1, 1, device=dev))

    # ------------------------------------------------------------------
    # Patch registration
    # ------------------------------------------------------------------

    def quantize_patch(
        self,
        i: int,
        j: int,
        A_ij: torch.Tensor,
        Y_ij: torch.Tensor,
        Z_ij: torch.Tensor,
    ) -> None:
        """Register a BQQ-quantized patch.

        Parameters
        ----------
        i, j  : patch row / column indices
        A_ij  : (bit_width, 4)             – BQQ coefficients from quantize_weight_to_bqq
        Y_ij  : (bit_width, y_row, inter_dim) – binary factor matrix (already optimised)
        Z_ij  : (bit_width, inter_dim, z_col)  – binary factor matrix (already optimised)
        """
        inter_dim = Y_ij.shape[-1]
        if self.inter_dimension is None:
            self._init_bqq_tensors(inter_dim)

        dev = self.float_weight.device
        with torch.no_grad():
            self.Y[:, i, j] = (Y_ij > 0.5).to(dev)
            self.Z[:, i, j] = (Z_ij > 0.5).to(dev)
            # A_ij[:, 0..2] → a, b, c per-patch; A_ij[:, 3].sum() → d (matches BinaryQuadratic)
            self.a.data[:, i, j, 0, 0] = A_ij[:, 0].to(dev)
            self.b.data[:, i, j, 0, 0] = A_ij[:, 1].to(dev)
            self.c.data[:, i, j, 0, 0] = A_ij[:, 2].to(dev)
            self.d.data[i, j, 0, 0] = A_ij[:, 3].sum().to(dev)
            self.quantized_mask[i, j] = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _bqq_weight(self, dtype: torch.dtype) -> torch.Tensor:
        """Reconstruct full W from BQQ parameters (all patches, shape (out, in))."""
        W_core = torch.matmul(self.Y.to(dtype), self.Z.to(dtype))
        Y_sum = self.Y.sum(dim=-1, keepdim=True).to(dtype)
        Z_sum = self.Z.sum(dim=-2, keepdim=True).to(dtype)
        W = (self.a.to(dtype) * W_core
             + self.b.to(dtype) * Y_sum
             + self.c.to(dtype) * Z_sum)
        W = W.sum(dim=0) + self.d.to(dtype)  # (row_width, col_width, y_row, z_col)
        W = W.permute(0, 2, 1, 3).reshape(
            self.row_width * self.y_row, self.col_width * self.z_col
        )
        return W

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        dtype = X.dtype
        dev = self.float_weight.device

        if self.inter_dimension is None or not self.quantized_mask.any():
            # No patches quantized yet — pure float forward
            W = self.float_weight.to(dtype)
        elif self.quantized_mask.all():
            # All patches quantized — pure BQQ forward
            W = self._bqq_weight(dtype)
        else:
            # Mixed: select quantized patches from BQQ, rest from float_weight.
            # torch.where gradient routing:
            #   - quantized positions → grad flows to a/b/c/d (via W_bqq); float_weight gets 0
            #   - unquantized positions → grad flows to float_weight; a/b/c/d get 0
            W_bqq = self._bqq_weight(dtype)
            mask = (self.quantized_mask
                    .repeat_interleave(self.y_row, dim=0)
                    .repeat_interleave(self.z_col, dim=1))
            W = torch.where(mask, W_bqq, self.float_weight.to(dtype))

        result = X.to(dev) @ W.T
        if self.bias_param is not None:
            result = result + self.bias_param.to(dtype=dtype, device=dev)
        return result

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_binaryquadratic(self) -> 'BinaryQuadratic':
        """Convert to BinaryQuadratic. Raises if any patch is still unquantized."""
        if not self.quantized_mask.all():
            n = (~self.quantized_mask).sum().item()
            raise RuntimeError(
                f"to_binaryquadratic() called but {n} patches are still unquantized"
            )
        # Build BinaryQuadratic directly to avoid the constructor
        # (which expects a packed A tensor; we already have decomposed a/b/c/d)
        bqq = BinaryQuadratic.__new__(BinaryQuadratic)
        nn.Module.__init__(bqq)
        bqq.bit_width = self.bit_width
        bqq.row_width = self.row_width
        bqq.col_width = self.col_width
        bqq.y_row = self.y_row
        bqq.inter_dimension = self.inter_dimension
        bqq.z_col = self.z_col
        bqq.register_buffer('Y', self.Y.clone())
        bqq.register_buffer('Z', self.Z.clone())
        bqq.a = nn.Parameter(self.a.data.clone())
        bqq.b = nn.Parameter(self.b.data.clone())
        bqq.c = nn.Parameter(self.c.data.clone())
        bqq.d = nn.Parameter(self.d.data.clone())
        bqq.bias = (nn.Parameter(self.bias_param.data.clone())
                    if self.bias_param is not None else None)
        return bqq


def pack_binaryquadratic_model(model: nn.Module) -> nn.Module:
    """Recursively replace all BinaryQuadratic layers with PackedBinaryQuadratic."""
    for name, module in list(model.named_children()):
        if isinstance(module, BinaryQuadratic):
            setattr(model, name, PackedBinaryQuadratic.from_unpacked(module))
        else:
            pack_binaryquadratic_model(module)
    return model


# ---------------------------------------------------------------------------
# Merge utilities (shared)
# ---------------------------------------------------------------------------

def merge_binary_quadratic(diff_layer: BinaryQuadratic, quant_layer: BinaryQuadratic) -> BinaryQuadratic:
    merged_Y = torch.cat([quant_layer.Y, diff_layer.Y], dim=0)
    merged_Z = torch.cat([quant_layer.Z, diff_layer.Z], dim=0)
    merged_a = torch.cat([quant_layer.a, diff_layer.a], dim=0)
    merged_b = torch.cat([quant_layer.b, diff_layer.b], dim=0)
    merged_c = torch.cat([quant_layer.c, diff_layer.c], dim=0)
    merged_d = quant_layer.d + diff_layer.d
    merged_bias = quant_layer.bias

    return BinaryQuadratic(merged_Y, merged_Z, torch.cat([
        merged_a.squeeze(-1).squeeze(-1).unsqueeze(-1),
        merged_b.squeeze(-1).squeeze(-1).unsqueeze(-1),
        merged_c.squeeze(-1).squeeze(-1).unsqueeze(-1),
        merged_d.unsqueeze(-1),
    ], dim=-1), bias=merged_bias)


def merge_binaryquadratic_recursive(model_q: nn.Module, model_d: nn.Module, prefix=''):
    for (name_q, module_q), (name_d, module_d) in zip(model_q.named_children(), model_d.named_children()):
        assert name_q == name_d, f"Module name mismatch: {name_q} != {name_d}"
        full_name = f"{prefix}.{name_q}" if prefix else name_q

        if isinstance(module_q, BinaryQuadratic) and isinstance(module_d, BinaryQuadratic):
            merged = merge_binary_quadratic(module_d, module_q)
            setattr(model_q, name_q, merged)
            print(f"Merged BinaryQuadratic at {full_name}")
        else:
            merge_binaryquadratic_recursive(module_q, module_d, prefix=full_name)

    return model_q


# ---------------------------------------------------------------------------
# Patch → tensor conversion (shared)
# ---------------------------------------------------------------------------

def get_matrices(patch_list, bit_width):
    """Convert a flat list of patch dicts into (A, Y, Z) tensors."""
    row_width = max(patch['patch_row'] for patch in patch_list) + 1
    col_width = max(patch['patch_col'] for patch in patch_list) + 1
    m, l = patch_list[0]['mat1'].shape
    _, n = patch_list[0]['mat2'].shape
    coeff_dtype = patch_list[0]['coeff'].dtype
    matrix_dtype = patch_list[0]['mat1'].dtype

    A = torch.zeros((bit_width, row_width, col_width, 4), dtype=coeff_dtype)
    Y = torch.zeros((bit_width, row_width, col_width, m, l), dtype=matrix_dtype)
    Z = torch.zeros((bit_width, row_width, col_width, l, n), dtype=matrix_dtype)

    for patch in patch_list:
        i, j = patch['patch_row'], patch['patch_col']
        a, y, z, k = patch['coeff'], patch['mat1'], patch['mat2'], patch['bit_idx']
        if k >= bit_width:
            continue
        A[k, i, j] = a
        Y[k, i, j] = y
        Z[k, i, j] = z

    return A, Y, Z


# ---------------------------------------------------------------------------
# CV-specific: {-1, +1} representation and trainable layers
# ---------------------------------------------------------------------------

def transform_A(A, l):
    """Transform scaling coefficients from {0,1} to {-1,1} binary representation."""
    A0, A1, A2, A3 = A[..., 0], A[..., 1], A[..., 2], A[..., 3]
    new0 = A0 / 4
    new1 = A1 / 2 + A0 / 4
    new2 = A2 / 2 + A0 / 4
    new3 = (A0 / 4 + A1 / 2 + A2 / 2) * l + A3
    return torch.stack([new0, new1, new2, new3], dim=-1)


class SymQuantSTE(Function):
    """Symmetric quantization with straight-through estimator."""

    @staticmethod
    def forward(ctx, input, scale, num_bits):
        if num_bits == 1:
            s = scale.abs()
            output = s * torch.sgn(input)
        else:
            s = scale.abs().clamp_min(1e-8)
            qmax = 2 ** (num_bits - 1) - 1
            q = torch.clamp(torch.round(input / s), -qmax, qmax)
            output = q * s
        ctx.save_for_backward(input, s)
        ctx.num_bits = num_bits
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, s = ctx.saved_tensors
        num_bits = ctx.num_bits
        if num_bits == 1:
            mask = (input.abs() <= s).to(grad_output.dtype)
        else:
            qmax = 2 ** (num_bits - 1) - 1
            mask = (input.abs() <= qmax * s).to(grad_output.dtype)
        grad_input = grad_output * mask
        return grad_input, None, None


class BQQLinear(nn.Module):
    """
    Trainable BQQ linear layer using {-1,+1} representation.
    Y_fp, Z_fp are real-valued parameters; 1-bit quantization is applied in forward via SymQuantSTE.
    """

    def __init__(self, Y, Z, A, bias=None, act_bits=None, quant_bias=True):
        super().__init__()
        self.Y_fp = nn.Parameter(Y.clone().float())
        self.Z_fp = nn.Parameter(Z.clone().float())
        self.quant_bias = quant_bias
        self.A = nn.Parameter(A.clone().float())
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None
        self.act_bits = act_bits
        if act_bits is not None:
            self.act_scale = nn.Parameter(torch.tensor(1e-3))

        p, j, k, m, l = Y.shape
        _, _, _, _, n = Z.shape
        self.p, self.j, self.k, self.m, self.l, self.n = p, j, k, m, l, n
        self.in_features = k * n
        self.out_features = j * m

    def forward(self, input):
        orig_dtype = input.dtype
        device = self.Y_fp.device
        X = input.to(device=device, dtype=torch.float32)

        if self.act_bits is not None:
            X = SymQuantSTE.apply(X, self.act_scale, self.act_bits)

        Y_fp = self.Y_fp.to(device=device, dtype=torch.float32)
        Z_fp = self.Z_fp.to(device=device, dtype=torch.float32)
        Y_scale = Y_fp.abs().mean(dim=(-2, -1), keepdim=True)
        Z_scale = Z_fp.abs().mean(dim=(-2, -1), keepdim=True)
        Y_q = SymQuantSTE.apply(Y_fp, Y_scale, 1)
        Z_q = SymQuantSTE.apply(Z_fp, Z_scale, 1)

        p, j, k, m, l = Y_q.shape
        n = Z_q.shape[-1]

        orig_shape = X.shape[:-1]
        X_2d = X.reshape(-1, self.in_features)
        B = X_2d.shape[0]
        X_kn = X_2d.view(B, k, n)

        T = torch.einsum("bkn,pjkln->bpjkl", X_kn, Z_q)
        core = torch.einsum("pjkml,bpjkl->bpjkm", Y_q, T)

        if self.quant_bias:
            A = self.A.to(device=device, dtype=torch.float32)
            a = A[..., 0].unsqueeze(0).unsqueeze(-1)
            out1 = (core * a).sum(dim=(1, 3))

            Y_sum = Y_q.sum(dim=-1)
            b = A[..., 1]
            B_coef = (b.unsqueeze(-1) * Y_sum).sum(dim=0)
            Sx = X_kn.sum(dim=-1)
            out2 = torch.einsum("bk,jkm->bjm", Sx, B_coef)

            Zs = Z_q.sum(dim=-2)
            Tz = torch.einsum("bkn,pjkn->bpjk", X_kn, Zs)
            c = A[..., 2]
            out3 = (Tz * c.unsqueeze(0)).sum(dim=(1, 3)).unsqueeze(-1).expand(-1, -1, m)

            d = A[..., 3].unsqueeze(-1).unsqueeze(-1).sum(dim=0)
            D_coef = d[..., 0, 0]
            out4 = torch.einsum("bk,jk->bj", Sx, D_coef).unsqueeze(-1).expand(-1, -1, m)

            out_bjm = out1 + out2 + out3 + out4
        else:
            a = self.A.to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            out_bjm = (core * a).sum(dim=(1, 3))

        out_2d = out_bjm.reshape(B, self.out_features)
        if self.bias is not None:
            out_2d = out_2d + self.bias.to(out_2d.device, dtype=torch.float32)
        out = out_2d.view(*orig_shape, self.out_features)
        return out.to(dtype=input.dtype)


# Auto-register Triton AND+popcount kernel if available (CUDA + Triton required).
# Falls back to cuBLAS unpack path silently if not available.
try:
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from bqq_triton_kernel import packed_binary_matmul as _pbm
    PackedBinaryQuadratic.packed_kernel = staticmethod(_pbm)
    PackedBinaryQuadratic.use_packed_kernel = True
except Exception:
    pass


class BQQLinearInference(nn.Module):
    """Inference-optimized BQQ linear layer (no gradients, int8 signs + fp16 scales)."""

    def __init__(self, Y_sign, Z_sign, Y_scale, Z_scale, A, bias=None,
                 act_bits=None, quant_bias=True):
        super().__init__()
        self.Y_sign = nn.Parameter(Y_sign, requires_grad=False)
        self.Z_sign = nn.Parameter(Z_sign, requires_grad=False)
        self.Y_scale = nn.Parameter(Y_scale, requires_grad=False)
        self.Z_scale = nn.Parameter(Z_scale, requires_grad=False)
        self.quant_bias = quant_bias
        self.A = nn.Parameter(A, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self.act_bits = act_bits

        p, j, k, m, l = Y_sign.shape
        _, _, _, _, n = Z_sign.shape
        self.p, self.j, self.k, self.m, self.l, self.n = p, j, k, m, l, n
        self.in_features = k * n
        self.out_features = j * m

    @classmethod
    def from_trained(cls, layer: BQQLinear, sign_dtype=torch.int8, scale_dtype=torch.float16):
        device = layer.Y_fp.device
        with torch.no_grad():
            Y_fp = layer.Y_fp.detach().to(torch.float16)
            Z_fp = layer.Z_fp.detach().to(torch.float16)
            Y_scale = Y_fp.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8).to(scale_dtype)
            Z_scale = Z_fp.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8).to(scale_dtype)
            Y_sign = torch.sign(Y_fp).to(sign_dtype)
            Z_sign = torch.sign(Z_fp).to(sign_dtype)
            A = layer.A.detach().to(scale_dtype).clone()
            bias = layer.bias.detach().to(scale_dtype).clone() if layer.bias is not None else None
            return cls(
                Y_sign=Y_sign.to(device), Z_sign=Z_sign.to(device),
                Y_scale=Y_scale.to(device), Z_scale=Z_scale.to(device),
                A=A.to(device), bias=bias.to(device) if bias is not None else None,
                act_bits=layer.act_bits, quant_bias=layer.quant_bias,
            )

    def forward(self, input):
        orig_dtype = input.dtype
        device = self.Y_sign.device
        X = input.to(device=device, dtype=torch.float16)

        Y_q = self.Y_sign.to(dtype=torch.float16) * self.Y_scale.to(dtype=torch.float16)
        Z_q = self.Z_sign.to(dtype=torch.float16) * self.Z_scale.to(dtype=torch.float16)

        p, j, k, m, l = Y_q.shape
        n = Z_q.shape[-1]

        orig_shape = X.shape[:-1]
        X_2d = X.reshape(-1, self.in_features)
        B = X_2d.shape[0]
        X_kn = X_2d.view(B, k, n)

        T = torch.einsum("bkn,pjkln->bpjkl", X_kn, Z_q)
        core = torch.einsum("pjkml,bpjkl->bpjkm", Y_q, T)

        if self.quant_bias:
            A = self.A.to(dtype=torch.float16, device=device)
            a = A[..., 0].unsqueeze(0).unsqueeze(-1)
            out1 = (core * a).sum(dim=(1, 3))

            Y_sum = Y_q.sum(dim=-1)
            b = A[..., 1]
            B_coef = (b.unsqueeze(-1) * Y_sum).sum(dim=0)
            Sx = X_kn.sum(dim=-1)
            out2 = torch.einsum("bk,jkm->bjm", Sx, B_coef)

            Zs = Z_q.sum(dim=-2)
            Tz = torch.einsum("bkn,pjkn->bpjk", X_kn, Zs)
            c = A[..., 2]
            out3 = (Tz * c.unsqueeze(0)).sum(dim=(1, 3)).unsqueeze(-1).expand(-1, -1, m)

            d = A[..., 3].unsqueeze(-1).unsqueeze(-1).sum(dim=0)
            D_coef = d[..., 0, 0]
            out4 = torch.einsum("bk,jk->bj", Sx, D_coef).unsqueeze(-1).expand(-1, -1, m)

            out_bjm = out1 + out2 + out3 + out4
        else:
            a = self.A.to(dtype=torch.float16, device=device).unsqueeze(0).unsqueeze(-1)
            out_bjm = (core * a).sum(dim=(1, 3))

        out_2d = out_bjm.reshape(B, self.out_features)
        if self.bias is not None:
            out_2d = out_2d + self.bias.to(out_2d.device, dtype=torch.float16)
        out = out_2d.view(*orig_shape, self.out_features)
        return out.to(dtype=orig_dtype)
