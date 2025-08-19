import os
import random
import numpy as np
import torch


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


