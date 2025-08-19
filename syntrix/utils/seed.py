import os
import random
import numpy as np
import torch
from typing import Tuple


def set_seed(seed: int) -> None:
    """Set all relevant RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def set_threads(num_threads: int) -> None:
    """Pin BLAS/OMP and PyTorch threads for deterministic CPU perf."""
    if num_threads is None or num_threads <= 0:
        return
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    try:
        torch.set_num_threads(num_threads)
    except Exception:
        pass


def get_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "float64":
        return torch.float64
    return torch.float32


def try_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    """Optionally compile the model with torch.compile if available.

    Falls back to identity if not available. Returns the (possibly) compiled model.
    """
    if not enabled:
        return model

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return model
    try:
        return compile_fn(model)
    except Exception:
        return model


def tolerance_for_dtype(dtype: torch.dtype) -> Tuple[float, float]:
    """Return (rtol, atol) suitable for numeric checks under a dtype."""
    if dtype == torch.float64:
        return (1e-5, 1e-8)
    return (1e-4, 1e-6)
