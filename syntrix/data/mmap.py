from __future__ import annotations

import mmap
from pathlib import Path
from typing import Iterator, Tuple
import torch


class MMapText:
    """Minimal memory-mapped text reader yielding lines."""

    def __init__(self, path: str):
        self.path = Path(path)

    def __iter__(self) -> Iterator[str]:
        with self.path.open("rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for line in iter(mm.readline, b""):
                    yield line.decode("utf-8", errors="ignore").rstrip("\n")


def mmap_random_block_batch(
    path: str,
    batch_size: int,
    block_size: int,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample random contiguous blocks from a mem-mapped text file.

    Converts to byte-level ints for simplicity; for char/BPE tokenization,
    reading into memory and tokenizing is recommended.
    """
    p = Path(path)
    with p.open("rb") as f:
        data = f.read()
    n = max(1, len(data) - block_size - 1)
    if generator is None:
        idx = torch.randint(0, n, (batch_size,))
    else:
        idx = torch.randint(0, n, (batch_size,), generator=generator)
    x = torch.stack(
        [torch.tensor(list(data[i : i + block_size]), dtype=torch.long) for i in idx]
    )
    y = torch.stack(
        [
            torch.tensor(list(data[i + 1 : i + 1 + block_size]), dtype=torch.long)
            for i in idx
        ]
    )
    return x, y
