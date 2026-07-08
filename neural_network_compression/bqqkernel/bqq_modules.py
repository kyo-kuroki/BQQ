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


_HADAMARD_FNS = None


def _hadamard_transforms():
    """Lazily import the (heavy) Hadamard transform tables, only when an
    incoherence-processed layer is actually used."""
    global _HADAMARD_FNS
    if _HADAMARD_FNS is None:
        try:
            from .hadamard import matmul_hadU, matmul_hadUt
        except ImportError:
            from hadamard import matmul_hadU, matmul_hadUt
        _HADAMARD_FNS = (matmul_hadU, matmul_hadUt)
    return _HADAMARD_FNS


class IncoherentBinaryQuadratic(BinaryQuadratic):
    """BQQ layer whose stored weight lives in the incoherent (RHT) space.

    At quantization time the weight was randomized-Hadamard transformed,
    Wr = RHT_W(W, SU, SV), and BQQ-quantized in that space. This module stores the
    BQQ factors of the transformed quantized weight Wr_q plus the sign vectors
    SU (in-side) and SV (out-side), and applies the Hadamard transform to the input
    and the inverse to the output so the effective linear map equals the
    original-space quantized weight W_q = incoherence_process(Wr_q, SU, SV):

        W_q = diag(SV) G_out  Wr_q  G_in^T diag(SU)
        y = W_q x  =>  v = hadUt(x ⊙ SU);  w = v @ Wr_q^T;  y = hadU(w) ⊙ SV

    where G_in / G_out are the (orthonormal) Hadamard transforms applied by
    matmul_hadU / matmul_hadUt. This matches QUIP-Sharp's incoherent inference.
    """

    def __init__(self, Y, Z, A, SU, SV, bias=None):
        super().__init__(Y, Z, A, bias=bias)
        self.register_buffer("SU", SU.detach().float().reshape(-1))  # [in_features]
        self.register_buffer("SV", SV.detach().float().reshape(-1))  # [out_features]

    def forward(self, X):
        matmul_hadU, matmul_hadUt = _hadamard_transforms()
        dtype = X.dtype
        device = self.Y.device
        Wr_q = self.get_weight(dtype=dtype)                      # transformed-space weight [out, in]
        SU = self.SU.to(device=device, dtype=dtype)
        SV = self.SV.to(device=device, dtype=dtype)
        Xt = matmul_hadUt(X.to(device) * SU)                     # [..., in]
        out = Xt @ Wr_q.T                                        # [..., out]
        out = matmul_hadU(out) * SV
        if self.bias is not None:
            out = out + self.bias.type(dtype).to(device)
        return out

    def get_dense_weight(self, dtype=torch.float32):
        """Return the effective original-space quantized weight W_q [out, in]."""
        matmul_hadU, _ = _hadamard_transforms()
        Wr_q = self.get_weight(dtype=dtype)
        SU = self.SU.to(device=Wr_q.device, dtype=dtype)
        SV = self.SV.to(device=Wr_q.device, dtype=dtype)
        # incoherence_process(Wr_q, SU, SV)
        return (matmul_hadU((matmul_hadU(Wr_q) * SU).T) * SV).T


class BinarySTE01(Function):
    """Binary {0,1} quantization with learnable threshold and sigmoid STE."""

    @staticmethod
    def forward(ctx, input, theta, beta):
        centered = input - theta
        beta_clamped = torch.clamp(beta, min=1e-6)
        output = (centered > 0).to(input.dtype)
        ctx.save_for_backward(centered, beta_clamped)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        centered, beta = ctx.saved_tensors
        scaled = beta * centered
        sigma = torch.sigmoid(scaled)
        sigma_grad = sigma * (1.0 - sigma)
        surrogate = beta * sigma_grad
        grad_common = grad_output * surrogate
        grad_beta = torch.sum(grad_output * centered * sigma_grad).reshape_as(beta)
        return grad_common, -grad_common, grad_beta


class TrainableSTEBinaryQuadratic(nn.Module):
    """BQQ layer whose binary factors are optimized with scalar-threshold STE."""

    def __init__(
        self,
        Y,
        Z,
        A,
        bias=None,
        *,
        optimize_factors=True,
        optimize_coeffs=True,
        optimize_theta=True,
        optimize_beta=True,
        init_theta=0.5,
    ):
        super().__init__()
        self.bit_width, self.row_width, self.col_width, self.y_row, self.inter_dimension = Y.shape
        _, _, _, _, self.z_col = Z.shape

        y_init = Y.clone().float()
        z_init = Z.clone().float()
        a_init = A[..., 0].unsqueeze(-1).unsqueeze(-1).clone().float()
        b_init = A[..., 1].unsqueeze(-1).unsqueeze(-1).clone().float()
        c_init = A[..., 2].unsqueeze(-1).unsqueeze(-1).clone().float()
        d_init = A[..., 3].unsqueeze(-1).unsqueeze(-1).sum(dim=0).clone().float()

        self.Y_fp = nn.Parameter(y_init, requires_grad=optimize_factors)
        self.Z_fp = nn.Parameter(z_init, requires_grad=optimize_factors)
        self.Y_theta = nn.Parameter(torch.tensor(float(init_theta), dtype=torch.float32), requires_grad=optimize_theta)
        self.Z_theta = nn.Parameter(torch.tensor(float(init_theta), dtype=torch.float32), requires_grad=optimize_theta)
        self.Y_beta = nn.Parameter(torch.tensor(4.0, dtype=torch.float32), requires_grad=optimize_beta)
        self.Z_beta = nn.Parameter(torch.tensor(4.0, dtype=torch.float32), requires_grad=optimize_beta)
        self.a = nn.Parameter(a_init, requires_grad=optimize_coeffs)
        self.b = nn.Parameter(b_init, requires_grad=optimize_coeffs)
        self.c = nn.Parameter(c_init, requires_grad=optimize_coeffs)
        self.d = nn.Parameter(d_init, requires_grad=optimize_coeffs)
        self.bias = nn.Parameter(bias.clone().float(), requires_grad=optimize_coeffs) if bias is not None else None

    @classmethod
    def from_binaryquadratic(
        cls,
        layer: 'BinaryQuadratic',
        *,
        optimize_factors=True,
        optimize_coeffs=True,
        optimize_theta=True,
        optimize_beta=True,
        init_theta=0.5,
    ) -> 'TrainableSTEBinaryQuadratic':
        d_terms = torch.zeros(
            layer.bit_width, layer.row_width, layer.col_width, 1,
            dtype=layer.d.dtype, device=layer.d.device,
        )
        d_terms[0] = layer.d.detach()[..., 0, 0].unsqueeze(-1)
        A = torch.cat([
            layer.a.detach().squeeze(-1).squeeze(-1).unsqueeze(-1),
            layer.b.detach().squeeze(-1).squeeze(-1).unsqueeze(-1),
            layer.c.detach().squeeze(-1).squeeze(-1).unsqueeze(-1),
            d_terms,
        ], dim=-1)
        bias = layer.bias.detach().clone() if layer.bias is not None else None
        return cls(
            layer.Y.detach().float(),
            layer.Z.detach().float(),
            A,
            bias=bias,
            optimize_factors=optimize_factors,
            optimize_coeffs=optimize_coeffs,
            optimize_theta=optimize_theta,
            optimize_beta=optimize_beta,
            init_theta=init_theta,
        )

    def _quantized_factors(self, dtype):
        Y_q = BinarySTE01.apply(self.Y_fp, self.Y_theta, self.Y_beta).to(dtype)
        Z_q = BinarySTE01.apply(self.Z_fp, self.Z_theta, self.Z_beta).to(dtype)
        return Y_q, Z_q

    def get_weight(self, dtype=torch.float32):
        Y_q, Z_q = self._quantized_factors(dtype)
        W_core = torch.matmul(Y_q, Z_q)
        Y_sum = Y_q.sum(dim=-1, keepdim=True)
        Z_sum = Z_q.sum(dim=-2, keepdim=True)
        W = self.a.to(dtype) * W_core + self.b.to(dtype) * Y_sum + self.c.to(dtype) * Z_sum
        W = W.sum(dim=0) + self.d.to(dtype)
        return W.permute(0, 2, 1, 3).reshape(self.row_width * self.y_row, self.col_width * self.z_col)

    def forward(self, X):
        dtype = X.dtype
        device = self.Y_fp.device
        W = self.get_weight(dtype=dtype)
        out = X.to(device) @ W.T
        if self.bias is not None:
            out = out + self.bias.to(dtype=dtype, device=device)
        return out

    def to_binaryquadratic(self) -> 'BinaryQuadratic':
        with torch.no_grad():
            Y_q = (self.Y_fp > self.Y_theta).bool()
            Z_q = (self.Z_fp > self.Z_theta).bool()
            d_terms = torch.zeros(
                self.bit_width, self.row_width, self.col_width, 1,
                dtype=self.d.dtype, device=self.d.device,
            )
            d_terms[0] = self.d.detach()[..., 0, 0].unsqueeze(-1)
            A = torch.cat([
                self.a.detach().squeeze(-1).squeeze(-1).unsqueeze(-1),
                self.b.detach().squeeze(-1).squeeze(-1).unsqueeze(-1),
                self.c.detach().squeeze(-1).squeeze(-1).unsqueeze(-1),
                d_terms,
            ], dim=-1)
            bias = self.bias.detach().clone() if self.bias is not None else None
            return BinaryQuadratic(Y_q.float(), Z_q.float(), A, bias=bias)




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

    _empty_bias = None  # cached empty tensor for no-bias case

    def _get_flat_cache(self) -> dict:
        """Flat uint8 weights + fp16 coefficients for the CUDA kernel.

        The kernel consumes [B, y_row/z_col, k8] uint8 and flat fp16 a/b/c/d
        (converted to fp32 in-kernel; accumulation stays fp32).
        Rebuilding those views and dtype-converting the coefficients on every
        call adds ~6 CUDA ops per layer, which dominates single-token decode,
        so cache them here.  The cache key is the data_ptr of Y_packed and a,
        which changes whenever .to()/.half()/.cuda() rebuilds storage.
        In-place edits of a/b/c/d (e.g. fine-tuning) are NOT detected — call
        _invalidate_flat_cache() after mutating coefficients.
        """
        cache = self.__dict__.get("_flat_cache")
        key = (self.Y_packed.data_ptr(), self.a.data_ptr())
        if cache is not None and cache["key"] == key:
            return cache
        B = self.bit_width * self.row_width * self.col_width
        cache = {
            "key": key,
            "Y_flat": self.Y_packed.reshape(B, self.y_row, self._k8).contiguous(),
            "Z_flat": self.Z_packed.reshape(B, self.z_col, self._k8).contiguous(),
            "a_flat": self.a.detach().reshape(B).half().contiguous(),
            "b_flat": self.b.detach().reshape(B).half().contiguous(),
            "c_flat": self.c.detach().reshape(B).half().contiguous(),
            "d_flat": self.d.detach().reshape(
                self.row_width * self.col_width).half().contiguous(),
            # Pre-zeroed fp32 accumulation workspace for the fused decode
            # epilogue (bqq_ws_store_half_zero_kernel re-zeroes after use).
            "ws": torch.zeros(1, self.row_width, self.y_row,
                              dtype=torch.float32,
                              device=self.Y_packed.device),
        }
        self.__dict__["_flat_cache"] = cache
        return cache

    def _invalidate_flat_cache(self) -> None:
        self.__dict__.pop("_flat_cache", None)

    _forward_flat = None  # cached ext.bqq_forward_flat (avoids per-call lookup)

    def _needs_grad(self, X: torch.Tensor) -> bool:
        return torch.is_grad_enabled() and (
            X.requires_grad
            or self.a.requires_grad or self.b.requires_grad
            or self.c.requires_grad or self.d.requires_grad
            or (self.bias is not None and self.bias.requires_grad))

    def forward(self, X):
        if self._needs_grad(X):
            # Differentiable fallback: the CUDA kernel has no autograd
            # support and reads detached fp16 coefficient caches, so when
            # gradients are needed rebuild W from a/b/c/d (Y/Z stay frozen
            # bits) exactly like BinaryQuadratic's training forward.
            # Gradients flow to a/b/c/d, bias and X.  The flat cache is
            # dropped here because the optimizer updates a/b/c/d in place,
            # which the data_ptr cache key cannot detect.
            self._invalidate_flat_cache()
            W = self.get_weight(X.dtype)
            out = X @ W.T
            if self.bias is not None:
                out = out + self.bias.type(X.dtype)
            return out
        fwd = PackedBinaryQuadratic._forward_flat
        if fwd is None:
            from .bqq_cuda_ext import _get_ext
            fwd = PackedBinaryQuadratic._forward_flat = _get_ext().bqq_forward_flat
        if self.bias is not None:
            b = self.bias
        else:
            if PackedBinaryQuadratic._empty_bias is None or \
               PackedBinaryQuadratic._empty_bias.device != self.Y_packed.device:
                PackedBinaryQuadratic._empty_bias = torch.empty(
                    0, device=self.Y_packed.device)
            b = PackedBinaryQuadratic._empty_bias
        cache = self._get_flat_cache()
        return fwd(
            cache["Y_flat"], cache["Z_flat"], X,
            cache["a_flat"], cache["b_flat"], cache["c_flat"], cache["d_flat"],
            b, cache["ws"],
            self.bit_width, self.row_width, self.col_width,
            self.y_row, self.z_col)

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

    def to_unpacked(self) -> 'BinaryQuadratic':
        """Convert back to BinaryQuadratic (inverse of from_unpacked).

        For tools that operate on the unpacked layout (.Y/.Z bool tensors),
        e.g. scale refinement.  Numerically exact: bits are unpacked
        losslessly and A is reassembled so that BinaryQuadratic.__init__
        reproduces a/b/c/d (d is split evenly across bits before the
        sum(dim=0) in __init__).
        """
        Y = self._unpack_to_bool(self.Y_packed, self.inter_dimension)
        # Z was stored transposed as [..., z_col, inter_dim]; transpose back
        Z = self._unpack_to_bool(self.Z_packed, self.inter_dimension) \
            .transpose(-2, -1).contiguous()

        A = torch.empty(
            self.bit_width, self.row_width, self.col_width, 4,
            dtype=self.a.dtype, device=self.a.device)
        A[..., 0] = self.a.detach()[..., 0, 0]
        A[..., 1] = self.b.detach()[..., 0, 0]
        A[..., 2] = self.c.detach()[..., 0, 0]
        A[..., 3] = (self.d.detach()[..., 0, 0] / self.bit_width).expand(
            self.bit_width, -1, -1)

        return BinaryQuadratic(
            Y, Z, A,
            bias=self.bias.data.clone() if self.bias is not None else None,
        )


class PartialBQQLinear(nn.Module):
    """Mixed-precision linear layer for progressive patch-wise quantization.

    Patches are quantized incrementally:
      - Quantized patches: Y/Z are trainable via STE with elementwise theta;
        a, b, c, d are nn.Parameters.
      - Unquantized patches: represented by float_weight (trainable nn.Parameter).

    Forward assembles W via::

        W = torch.where(mask_full, W_bqq, float_weight)

    so gradients automatically flow to BQQ parameters for quantized patches and to
    float_weight for unquantized patches.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias,
        group_size: int,
        bit_width: int,
        *,
        optimize_factors: bool = True,
        optimize_coeffs: bool = True,
        optimize_theta: bool = True,
        init_theta: float = 0.5,
    ):
        super().__init__()
        out_features, in_features = weight.shape

        def _patch_dims(dim, gs, name):
            if dim % gs != 0:
                raise ValueError(f"{name}={dim} is not divisible by group_size={gs}.")
            return dim // gs, gs

        self.group_size = group_size
        self.bit_width = bit_width
        self.row_width, self.y_row = _patch_dims(out_features, group_size, 'out_features')
        self.col_width, self.z_col = _patch_dims(in_features, group_size, 'in_features')
        self.inter_dimension = None
        self.optimize_factors = optimize_factors
        self.optimize_coeffs = optimize_coeffs
        self.optimize_theta = optimize_theta
        self.init_theta = float(init_theta)

        self.float_weight = nn.Parameter(weight.clone().float())
        self.bias_param = nn.Parameter(bias.clone().float()) if bias is not None else None

        self.register_buffer(
            'quantized_mask',
            torch.zeros(self.row_width, self.col_width, dtype=torch.bool),
        )

    def _init_bqq_tensors(self, inter_dimension: int) -> None:
        bw = self.bit_width
        rw, cw = self.row_width, self.col_width
        yr, zc = self.y_row, self.z_col
        dev = self.float_weight.device

        self.inter_dimension = inter_dimension
        self.Y_fp = nn.Parameter(
            torch.zeros(bw, rw, cw, yr, inter_dimension, dtype=torch.float32, device=dev),
            requires_grad=self.optimize_factors,
        )
        self.Z_fp = nn.Parameter(
            torch.zeros(bw, rw, cw, inter_dimension, zc, dtype=torch.float32, device=dev),
            requires_grad=self.optimize_factors,
        )
        self.Y_theta = nn.Parameter(
            torch.full((bw, rw, cw, yr, inter_dimension), self.init_theta, dtype=torch.float32, device=dev),
            requires_grad=self.optimize_theta,
        )
        self.Z_theta = nn.Parameter(
            torch.full((bw, rw, cw, inter_dimension, zc), self.init_theta, dtype=torch.float32, device=dev),
            requires_grad=self.optimize_theta,
        )
        self.a = nn.Parameter(
            torch.zeros(bw, rw, cw, 1, 1, dtype=torch.float32, device=dev),
            requires_grad=self.optimize_coeffs,
        )
        self.b = nn.Parameter(
            torch.zeros(bw, rw, cw, 1, 1, dtype=torch.float32, device=dev),
            requires_grad=self.optimize_coeffs,
        )
        self.c = nn.Parameter(
            torch.zeros(bw, rw, cw, 1, 1, dtype=torch.float32, device=dev),
            requires_grad=self.optimize_coeffs,
        )
        self.d = nn.Parameter(
            torch.zeros(rw, cw, 1, 1, dtype=torch.float32, device=dev),
            requires_grad=self.optimize_coeffs,
        )

    def quantize_patch(
        self,
        i: int,
        j: int,
        A_ij: torch.Tensor,
        Y_ij: torch.Tensor,
        Z_ij: torch.Tensor,
    ) -> None:
        inter_dim = Y_ij.shape[-1]
        if self.inter_dimension is None:
            self._init_bqq_tensors(inter_dim)

        dev = self.float_weight.device
        with torch.no_grad():
            self.Y_fp.data[:, i, j] = Y_ij.to(dev=dev, dtype=torch.float32)
            self.Z_fp.data[:, i, j] = Z_ij.to(dev=dev, dtype=torch.float32)
            self.Y_theta.data[:, i, j].fill_(self.init_theta)
            self.Z_theta.data[:, i, j].fill_(self.init_theta)
            self.a.data[:, i, j, 0, 0] = A_ij[:, 0].to(dev)
            self.b.data[:, i, j, 0, 0] = A_ij[:, 1].to(dev)
            self.c.data[:, i, j, 0, 0] = A_ij[:, 2].to(dev)
            self.d.data[i, j, 0, 0] = A_ij[:, 3].sum().to(dev)
            self.quantized_mask[i, j] = True

    def _bqq_weight(self, dtype: torch.dtype) -> torch.Tensor:
        Y_q = BinarySTE01.apply(self.Y_fp, self.Y_theta, self.Y_beta).to(dtype)
        Z_q = BinarySTE01.apply(self.Z_fp, self.Z_theta, self.Z_beta).to(dtype)
        W_core = torch.matmul(Y_q, Z_q)
        Y_sum = Y_q.sum(dim=-1, keepdim=True).to(dtype)
        Z_sum = Z_q.sum(dim=-2, keepdim=True).to(dtype)
        W = (self.a.to(dtype) * W_core
             + self.b.to(dtype) * Y_sum
             + self.c.to(dtype) * Z_sum)
        W = W.sum(dim=0) + self.d.to(dtype)
        return W.permute(0, 2, 1, 3).reshape(
            self.row_width * self.y_row, self.col_width * self.z_col
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        dtype = X.dtype
        dev = self.float_weight.device

        if self.inter_dimension is None or not self.quantized_mask.any():
            W = self.float_weight.to(dtype)
        elif self.quantized_mask.all():
            W = self._bqq_weight(dtype)
        else:
            W_bqq = self._bqq_weight(dtype)
            mask = (self.quantized_mask
                    .repeat_interleave(self.y_row, dim=0)
                    .repeat_interleave(self.z_col, dim=1))
            W = torch.where(mask, W_bqq, self.float_weight.to(dtype))

        result = X.to(dev) @ W.T
        if self.bias_param is not None:
            result = result + self.bias_param.to(dtype=dtype, device=dev)
        return result

    def to_binaryquadratic(self) -> 'BinaryQuadratic':
        if not self.quantized_mask.all():
            n = (~self.quantized_mask).sum().item()
            raise RuntimeError(
                f"to_binaryquadratic() called but {n} patches are still unquantized"
            )
        bqq = BinaryQuadratic.__new__(BinaryQuadratic)
        nn.Module.__init__(bqq)
        bqq.bit_width = self.bit_width
        bqq.row_width = self.row_width
        bqq.col_width = self.col_width
        bqq.y_row = self.y_row
        bqq.inter_dimension = self.inter_dimension
        bqq.z_col = self.z_col
        bqq.register_buffer('Y', (self.Y_fp.detach() > self.Y_theta.detach()))
        bqq.register_buffer('Z', (self.Z_fp.detach() > self.Z_theta.detach()))
        bqq.a = nn.Parameter(self.a.data.clone())
        bqq.b = nn.Parameter(self.b.data.clone())
        bqq.c = nn.Parameter(self.c.data.clone())
        bqq.d = nn.Parameter(self.d.data.clone())
        bqq.bias = (nn.Parameter(self.bias_param.data.clone())
                    if self.bias_param is not None else None)
        return bqq


class TrainableIncoherentSTEBinaryQuadratic(TrainableSTEBinaryQuadratic):
    """TrainableSTEBinaryQuadratic that keeps the RHT transform (SU/SV).

    The weight is stored and optimized in the incoherent (RHT) space, exactly
    like IncoherentBinaryQuadratic, but the binary factors Y/Z are continuous
    STE parameters so that continuous-param fine-tuning can update a/b/c/d.
    """

    def __init__(self, Y, Z, A, SU, SV, bias=None, **ste_kwargs):
        super().__init__(Y, Z, A, bias=bias, **ste_kwargs)
        self.register_buffer("SU", SU.detach().float().reshape(-1))
        self.register_buffer("SV", SV.detach().float().reshape(-1))

    @classmethod
    def from_incoherent_binaryquadratic(
        cls,
        layer: 'IncoherentBinaryQuadratic',
        *,
        optimize_factors: bool = True,
        optimize_coeffs: bool = True,
        optimize_theta: bool = True,
        optimize_beta: bool = True,
        init_theta: float = 0.5,
    ) -> 'TrainableIncoherentSTEBinaryQuadratic':
        d_terms = torch.zeros(
            layer.bit_width, layer.row_width, layer.col_width, 1,
            dtype=layer.d.dtype, device=layer.d.device,
        )
        d_terms[0] = layer.d.detach()[..., 0, 0].unsqueeze(-1)
        A = torch.cat([
            layer.a.detach().squeeze(-1).squeeze(-1).unsqueeze(-1),
            layer.b.detach().squeeze(-1).squeeze(-1).unsqueeze(-1),
            layer.c.detach().squeeze(-1).squeeze(-1).unsqueeze(-1),
            d_terms,
        ], dim=-1)
        bias = layer.bias.detach().clone() if layer.bias is not None else None
        return cls(
            layer.Y.detach().float(),
            layer.Z.detach().float(),
            A,
            SU=layer.SU.detach().clone(),
            SV=layer.SV.detach().clone(),
            bias=bias,
            optimize_factors=optimize_factors,
            optimize_coeffs=optimize_coeffs,
            optimize_theta=optimize_theta,
            optimize_beta=optimize_beta,
            init_theta=init_theta,
        )

    def forward(self, X):
        matmul_hadU, matmul_hadUt = _hadamard_transforms()
        dtype = X.dtype
        device = self.Y_fp.device
        Wr_q = self.get_weight(dtype=dtype)
        SU = self.SU.to(device=device, dtype=dtype)
        SV = self.SV.to(device=device, dtype=dtype)
        Xt = matmul_hadUt(X.to(device) * SU)
        out = Xt @ Wr_q.T
        out = matmul_hadU(out) * SV
        if self.bias is not None:
            out = out + self.bias.to(dtype=dtype, device=device)
        return out

    def to_incoherent_binaryquadratic(self) -> 'IncoherentBinaryQuadratic':
        with torch.no_grad():
            Y_q = (self.Y_fp > self.Y_theta).bool()
            Z_q = (self.Z_fp > self.Z_theta).bool()
            d_terms = torch.zeros(
                self.bit_width, self.row_width, self.col_width, 1,
                dtype=self.d.dtype, device=self.d.device,
            )
            d_terms[0] = self.d.detach()[..., 0, 0].unsqueeze(-1)
            A = torch.cat([
                self.a.detach().squeeze(-1).squeeze(-1).unsqueeze(-1),
                self.b.detach().squeeze(-1).squeeze(-1).unsqueeze(-1),
                self.c.detach().squeeze(-1).squeeze(-1).unsqueeze(-1),
                d_terms,
            ], dim=-1)
            bias = self.bias.detach().clone() if self.bias is not None else None
            return IncoherentBinaryQuadratic(
                Y_q.float(), Z_q.float(), A,
                SU=self.SU.detach().clone(),
                SV=self.SV.detach().clone(),
                bias=bias,
            )


def convert_binaryquadratic_model_to_ste(
    model: nn.Module,
    *,
    optimize_factors: bool = True,
    optimize_coeffs: bool = True,
    optimize_theta: bool = True,
    optimize_beta: bool = True,
    init_theta: float = 0.5,
) -> nn.Module:
    """Recursively replace BinaryQuadratic/IncoherentBinaryQuadratic with trainable STE variants."""
    for name, module in list(model.named_children()):
        if isinstance(module, IncoherentBinaryQuadratic):
            # Must check before BinaryQuadratic (parent class).
            setattr(
                model,
                name,
                TrainableIncoherentSTEBinaryQuadratic.from_incoherent_binaryquadratic(
                    module,
                    optimize_factors=optimize_factors,
                    optimize_coeffs=optimize_coeffs,
                    optimize_theta=optimize_theta,
                    optimize_beta=optimize_beta,
                    init_theta=init_theta,
                ),
            )
        elif isinstance(module, BinaryQuadratic):
            setattr(
                model,
                name,
                TrainableSTEBinaryQuadratic.from_binaryquadratic(
                    module,
                    optimize_factors=optimize_factors,
                    optimize_coeffs=optimize_coeffs,
                    optimize_theta=optimize_theta,
                    optimize_beta=optimize_beta,
                    init_theta=init_theta,
                ),
            )
        else:
            convert_binaryquadratic_model_to_ste(
                module,
                optimize_factors=optimize_factors,
                optimize_coeffs=optimize_coeffs,
                optimize_theta=optimize_theta,
                optimize_beta=optimize_beta,
                init_theta=init_theta,
            )
    return model


def convert_ste_model_to_binaryquadratic(model: nn.Module) -> nn.Module:
    """Recursively replace trainable STE layers with their frozen BQQ counterparts."""
    for name, module in list(model.named_children()):
        if isinstance(module, TrainableIncoherentSTEBinaryQuadratic):
            # Must check before TrainableSTEBinaryQuadratic (parent class).
            setattr(model, name, module.to_incoherent_binaryquadratic())
        elif isinstance(module, TrainableSTEBinaryQuadratic):
            setattr(model, name, module.to_binaryquadratic())
        else:
            convert_ste_model_to_binaryquadratic(module)
    return model


def pack_binaryquadratic_model(model: nn.Module) -> nn.Module:
    """Recursively replace all BinaryQuadratic layers with PackedBinaryQuadratic."""
    for name, module in list(model.named_children()):
        if isinstance(module, BinaryQuadratic):
            setattr(model, name, PackedBinaryQuadratic.from_unpacked(module))
        else:
            pack_binaryquadratic_model(module)
    return model


def unpack_binaryquadratic_model(model: nn.Module) -> nn.Module:
    """Recursively replace all PackedBinaryQuadratic layers with BinaryQuadratic."""
    for name, module in list(model.named_children()):
        if isinstance(module, PackedBinaryQuadratic):
            setattr(model, name, module.to_unpacked())
        else:
            unpack_binaryquadratic_model(module)
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
    from .bqq_triton_kernel import packed_binary_matmul as _pbm
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
