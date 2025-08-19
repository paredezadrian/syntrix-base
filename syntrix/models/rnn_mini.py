from __future__ import annotations

import torch
import torch.nn as nn

from syntrix.nn.layers import RMSNorm
from syntrix.nn.rnn import GatedRNN


class RNNMini(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layer: int = 4,
        block_size: int = 128,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [GatedRNN(d_model=d_model, mlp_ratio=mlp_ratio) for _ in range(n_layer)]
        )
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.block_size
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        return self.lm_head(x)
