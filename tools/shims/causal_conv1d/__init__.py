from __future__ import annotations

import torch
import torch.nn.functional as F


def _apply_activation(x: torch.Tensor, activation: str | None) -> torch.Tensor:
    if activation is None or activation == "silu":
        return F.silu(x)
    if activation == "swish":
        return F.silu(x)
    if activation == "relu":
        return F.relu(x)
    raise ValueError(f"Unsupported activation for causal_conv1d fallback: {activation}")


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    seq_idx=None,
):
    del seq_idx
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=weight.shape[-1] - 1, groups=x.shape[1])
    out = out[:, :, : x.shape[-1]]
    return _apply_activation(out, activation)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
):
    _, channels, seq_len = x.shape
    state_len = conv_state.shape[-1]
    x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
    conv_state.copy_(x_new[:, :, -state_len:])
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=channels)
    out = out[:, :, -seq_len:]
    return _apply_activation(out, activation).to(x.dtype)
