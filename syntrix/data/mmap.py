from __future__ import annotations

import mmap
from pathlib import Path
from typing import Iterator


class MMapText:
    """Minimal memory-mapped text reader yielding lines."""

    def __init__(self, path: str):
        self.path = Path(path)

    def __iter__(self) -> Iterator[str]:
        with self.path.open("rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for line in iter(mm.readline, b""):
                    yield line.decode("utf-8", errors="ignore").rstrip("\n")


