from __future__ import annotations

from typing import Iterable
import torch


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains shadow parameters updated as: m <- decay*m + (1-decay)*p
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], decay: float = 0.999):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in params]
        self.num_updates = 0

    @torch.no_grad()
    def update(self, params: Iterable[torch.nn.Parameter]):
        self.num_updates += 1
        d = self.decay
        for s, p in zip(self.shadow, params):
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, params: Iterable[torch.nn.Parameter]):
        for s, p in zip(self.shadow, params):
            p.copy_(s)
