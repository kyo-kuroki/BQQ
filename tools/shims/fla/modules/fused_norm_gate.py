from __future__ import annotations

import torch
from torch import nn


class FusedRMSNormGated(nn.Module):
    """Pure-torch stand-in for fla.modules.FusedRMSNormGated.

    The keyword arguments mirror the real one: transformers' Qwen3-Next code
    constructs it with elementwise_affine/device/dtype, and dropping any of them
    makes the model fail to build.
    """

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-6,
        activation: str | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.eps = eps
        self.activation = activation
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(hidden_size, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(hidden_states.dtype)
        if self.weight is not None:
            x = x * self.weight
        if gate is not None:
            x = x * torch.nn.functional.silu(gate.to(torch.float32))
        return x.to(hidden_states.dtype)
