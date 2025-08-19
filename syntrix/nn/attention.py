from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RotaryEmbedding, RMSNorm


class CausalSelfAttention(nn.Module):
    """Tiny causal self-attention with optional RoPE and pre-LN.

    Args:
        d_model: embedding dimension
        num_heads: number of attention heads
        rope: optional RotaryEmbedding applied to q and k
        bias: use bias in linear layers
        dropout: attention dropout probability (default 0.0)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: Optional[RotaryEmbedding] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = rope

        self.norm = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = (
            nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        )

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (B, H, T, D)

    def forward(
        self,
        x: torch.Tensor,
        return_attn_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape
        x = self.norm(x)
        qkv = self.qkv(x)  # (B, T, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)

        # Handle RoPE with flexibility: apply either in model space (D)
        # before head-splitting, or per-head (head_dim) after splitting.
        if self.rope is not None:
            if self.rope.dim == self.d_model:
                cos, sin = self.rope.get_cos_sin(seq_len, x.device, x.dtype)
                q = self.rope.apply_rotary(q, cos, sin)
                k = self.rope.apply_rotary(k, cos, sin)
                q = self._shape(q)
                k = self._shape(k)
                v = self._shape(v)
            elif self.rope.dim == self.head_dim:
                q = self._shape(q)
                k = self._shape(k)
                v = self._shape(v)
                cos, sin = self.rope.get_cos_sin(seq_len, x.device, x.dtype)
                q = self.rope.apply_rotary(q, cos, sin)
                k = self.rope.apply_rotary(k, cos, sin)
            else:
                raise ValueError(
                    f"RoPE dim {self.rope.dim} must equal d_model {self.d_model} or head_dim {self.head_dim}"
                )
        else:
            q = self._shape(q)
            k = self._shape(k)
            v = self._shape(v)

        # scaled dot-product attention with explicit causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)
        causal_mask = torch.ones(
            seq_len, seq_len, device=x.device, dtype=torch.bool
        ).triu(1)
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        y = torch.matmul(attn, v)  # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        y = self.proj(y)

        if return_attn_weights:
            return y, attn
        return y
