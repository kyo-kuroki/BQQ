from __future__ import annotations

import torch
from torch import nn


class FusedRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, activation: str | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(hidden_states.dtype) * self.weight
        if gate is not None:
            x = x * torch.nn.functional.silu(gate.to(torch.float32))
        return x.to(hidden_states.dtype)
