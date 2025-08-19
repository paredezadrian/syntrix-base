from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm without bias.

    Normalizes over the last dimension and applies a learnable scale.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        dim = x.shape[-1]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x_norm = x * rms
        return x_norm * self.weight.view(*([1] * (x.ndim - 1)), dim)


class SwiGLU(nn.Module):
    """SwiGLU activation block: (W_up x) * silu(W_gate x).

    Produces a hidden projection with gating via SiLU, commonly used in modern Transformers.
    """

    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(input_dim, 2 * hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up, gate = self.proj(x).chunk(2, dim=-1)
        return up * F.silu(gate)


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE).

    This module provides cos/sin caches and an application helper that rotates
    even-odd feature pairs by a position-dependent angle. Works on inputs with
    shape (..., seq_len, dim) or (batch, heads, seq_len, dim).
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        self.dim = dim
        self.base = base
        # inv_freq has size dim/2
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def get_cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # freqs: (seq_len, dim/2)
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, D/2)
        cos = freqs.to(dtype).cos()
        sin = freqs.to(dtype).sin()
        return cos, sin  # shapes: (T, D/2)

    @staticmethod
    def _ensure_broadcastable(x_like: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Make y broadcastable to x_like by left-padding dimensions with 1s
        while y.ndim < x_like.ndim:
            y = y.unsqueeze(0)
        return y

    def apply_rotary(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply RoPE rotation to x using provided cos/sin.

        Args:
            x: tensor of shape (..., T, D)
            cos: tensor of shape (T, D/2)
            sin: tensor of shape (T, D/2)
        Returns:
            Tensor with same shape as x with rotary applied on the last dim.
        """
        dim = x.shape[-1]
        assert dim == self.dim, f"x last dim {dim} != rotary dim {self.dim}"

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        cos_b = self._ensure_broadcastable(x_even, cos)
        sin_b = self._ensure_broadcastable(x_even, sin)

        x_rot_even = x_even * cos_b - x_odd * sin_b
        x_rot_odd = x_even * sin_b + x_odd * cos_b

        # Interleave even/odd back
        out = torch.empty_like(x)
        out[..., ::2] = x_rot_even
        out[..., 1::2] = x_rot_odd
        return out
