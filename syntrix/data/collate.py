import torch


def collate_microbatches(xs, ys, microbatch: int):
    """Yield microbatches from full arrays."""
    n = xs.shape[0]
    for i in range(0, n, microbatch):
        xb = torch.tensor(xs[i : i + microbatch], dtype=torch.long)
        yb = torch.tensor(ys[i : i + microbatch], dtype=torch.long)
        yield xb, yb
