from __future__ import annotations

import torch
import torch.nn as nn

from syntrix.nn.layers import RMSNorm, SwiGLU, RotaryEmbedding
from syntrix.nn.attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio: int = 4):
        super().__init__()
        # RoPE per head dimension
        head_dim = d_model // n_head
        rope = RotaryEmbedding(head_dim)
        self.attn = CausalSelfAttention(d_model=d_model, num_heads=n_head, rope=rope)

        hidden_dim = mlp_ratio * d_model
        self.mlp_norm = RMSNorm(d_model)
        self.mlp_up = SwiGLU(d_model, hidden_dim)
        self.mlp_down = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with internal pre-LN
        x = x + self.attn(x)
        # MLP with pre-LN
        y = self.mlp_norm(x)
        y = self.mlp_up(y)
        y = self.mlp_down(y)
        x = x + y
        return x


class GPTMini(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layer: int = 4,
        n_head: int = 4,
        block_size: int = 128,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, n_head=n_head, mlp_ratio=mlp_ratio)
                for _ in range(n_layer)
            ]
        )
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T)
        B, T = idx.shape
        assert (
            T <= self.block_size
        ), f"sequence length {T} exceeds block_size {self.block_size}"
        x = self.tok_emb(idx)  # (B, T, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)  # (B, T, vocab)
        return logits
