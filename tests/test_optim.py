import pytest
import torch
from syntrix.utils.seed import tolerance_for_dtype

from syntrix.optim.schedule import CosineWithWarmup
from syntrix.optim.ema import EMA


def test_cosine_with_warmup_curve():
    base_lr = 1.0
    warmup = 3
    total = 10
    sched = CosineWithWarmup(
        base_lr=base_lr, warmup_steps=warmup, total_steps=total, min_lr=0.0
    )
    lrs = [sched.step() for _ in range(total)]
    # Warmup should increase linearly to ~1.0
    assert lrs[0] == pytest.approx(base_lr * 1 / warmup)
    assert lrs[warmup - 1] == pytest.approx(base_lr)
    # After warmup, cosine decays and stays non-negative
    assert all(lr >= -1e-8 for lr in lrs)
    _, atol = tolerance_for_dtype(torch.get_default_dtype())
    assert lrs[-1] == pytest.approx(0.0, abs=max(atol, 1e-8))


def test_ema_tracks_parameters():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    ema = EMA(model.parameters(), decay=0.5)
    # Modify params slightly
    for p in model.parameters():
        p.data.add_(0.1)
    ema.update(model.parameters())
    # Copy to a clone and ensure values moved towards params
    clone = torch.nn.Linear(4, 4)
    ema.copy_to(clone.parameters())
    diff = 0.0
    for p, q in zip(model.parameters(), clone.parameters()):
        diff += (p - q).abs().mean().item()
    assert diff < 0.5  # sanity: should be reasonably close
