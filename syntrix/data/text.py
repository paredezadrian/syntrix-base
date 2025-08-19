from pathlib import Path
from typing import Tuple, List
import numpy as np


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def make_char_dataset(text: str, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    data = np.array([ord(c) for c in text], dtype=np.int32)
    n = len(data) - block_size
    xs = np.stack([data[i : i + block_size] for i in range(n)])
    ys = np.stack([data[i + 1 : i + 1 + block_size] for i in range(n)])
    return xs, ys


