from __future__ import annotations

import math
from typing import Optional


class CosineWithWarmup:
    """Cosine decay with linear warmup.

    step 0..warmup_steps-1: linear from 0 -> base_lr
    step >= warmup_steps: cosine decay from base_lr -> min_lr over (total_steps - warmup_steps)
    """

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ) -> None:
        assert total_steps > 0
        assert warmup_steps >= 0
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.step_num = 0

    def get_lr(self, step: Optional[int] = None) -> float:
        s = self.step_num if step is None else step
        if s < self.warmup_steps and self.warmup_steps > 0:
            return self.base_lr * (s + 1) / self.warmup_steps
        # cosine phase: ensure last step hits min_lr by using denom = steps_cosine - 1
        steps_cosine = self.total_steps - self.warmup_steps
        if steps_cosine <= 1:
            return self.min_lr
        denom = steps_cosine - 1
        s_rel = min(max(s - self.warmup_steps, 0), denom)
        cosine = 0.5 * (1 + math.cos(math.pi * s_rel / denom))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def step(self) -> float:
        lr = self.get_lr(self.step_num)
        self.step_num += 1
        return lr
