from __future__ import annotations

import torch
import torch.nn as nn

from .layers import RMSNorm, SwiGLU


class DiagonalSSM(nn.Module):
    """Minimal selective SSM: diagonal state update with conv shortcut.

    This is a toy implementation for CPU-friendly experiments.
    x -> pre-norm -> depthwise conv (shortcut) + diagonal state update, then SwiGLU MLP.
    """

    def __init__(
        self,
        d_model: int,
        state_size: int = 64,
        conv_kernel: int = 3,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=conv_kernel,
            groups=d_model,
            padding=conv_kernel // 2,
        )
        self.state = nn.Parameter(torch.randn(d_model, state_size) * 0.02)
        self.a = nn.Parameter(torch.tanh(torch.randn(d_model)))
        self.b = nn.Parameter(torch.randn(d_model))
        hidden_dim = mlp_ratio * d_model
        self.mlp_norm = RMSNorm(d_model)
        self.mlp_up = SwiGLU(d_model, hidden_dim)
        self.mlp_down = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        y = self.norm(x)  # (B, T, D)
        # depthwise conv shortcut
        y_conv = self.conv(y.transpose(1, 2)).transpose(1, 2)
        # diagonal SSM update (very simplified): rolling filter per channel
        s = torch.tanh(self.a).unsqueeze(0).unsqueeze(0)  # (1,1,D)
        b = self.b.unsqueeze(0).unsqueeze(0)  # (1,1,D)
        y_ssm = []
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        for t in range(T):
            h = s.squeeze(0).squeeze(0) * h + b.squeeze(0).squeeze(0) * y[:, t, :]
            y_ssm.append(h)
        y_ssm = torch.stack(y_ssm, dim=1)
        x = x + (y_conv + y_ssm)

        # MLP
        y = self.mlp_down(self.mlp_up(self.mlp_norm(x)))
        x = x + y
        return x
