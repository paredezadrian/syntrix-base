from __future__ import annotations

import torch
import torch.nn as nn

from .layers import RMSNorm, SwiGLU


class GatedRNN(nn.Module):
    """Lightweight GRU-based block with pre-LN and MLP.

    Applies pre-norm, a GRU over the time dimension, and a SwiGLU MLP with residuals.
    """

    def __init__(self, d_model: int, mlp_ratio: int = 4):
        super().__init__()
        self.rnn_norm = RMSNorm(d_model)
        self.gru = nn.GRU(
            input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True
        )
        hidden_dim = mlp_ratio * d_model
        self.mlp_norm = RMSNorm(d_model)
        self.mlp_up = SwiGLU(d_model, hidden_dim)
        self.mlp_down = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        y, _ = self.gru(self.rnn_norm(x))
        x = x + y
        y = self.mlp_down(self.mlp_up(self.mlp_norm(x)))
        x = x + y
        return x
