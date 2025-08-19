from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import torch


class CharTokenizer:
    def __init__(self, text: Optional[str] = None, chars: Optional[List[str]] = None):
        if chars is not None:
            # Use provided order exactly
            self.chars = list(chars)
        else:
            assert text is not None, "Either text or chars must be provided"
            self.chars = sorted(list(set(text)))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos.get(i, "�") for i in ids)


def load_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def make_char_dataset(
    text: str, block_size: int, tokenizer: Optional[CharTokenizer] = None
) -> Tuple[np.ndarray, np.ndarray]:
    if tokenizer is None:
        tokenizer = CharTokenizer(text)
    data = np.array(tokenizer.encode(text), dtype=np.int32)
    n = len(data) - block_size
    xs = np.stack([data[i : i + block_size] for i in range(n)])
    ys = np.stack([data[i + 1 : i + 1 + block_size] for i in range(n)])
    return xs, ys


def deterministic_split(
    tokens: torch.Tensor,
    val_ratio: float = 0.05,
    seed: int = 1337,
    shuffle: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministically split 1D token tensor into train/val.

    If shuffle is True, uses a seeded permutation; otherwise contiguous split.
    """
    assert tokens.dim() == 1, "tokens must be 1D"
    n = tokens.numel()
    split_idx = int(n * (1 - val_ratio))
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        data = tokens[perm]
    else:
        data = tokens
    return data[:split_idx].clone(), data[split_idx:].clone()


def random_block_batch(
    tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample random contiguous blocks (X, Y) from a 1D token tensor."""
    n = tokens.numel() - block_size - 1
    n = max(n, 1)
    if generator is None:
        idx = torch.randint(0, n, (batch_size,))
    else:
        idx = torch.randint(0, n, (batch_size,), generator=generator)
    x = torch.stack([tokens[i : i + block_size] for i in idx])
    y = torch.stack([tokens[i + 1 : i + 1 + block_size] for i in idx])
    return x, y


class TokenBatchIterator:
    """Simple iterator yielding random block batches with a seeded generator."""

    def __init__(
        self,
        tokens: torch.Tensor,
        block_size: int,
        batch_size: int,
        steps: int,
        seed: int = 0,
    ):
        self.tokens = tokens
        self.block_size = block_size
        self.batch_size = batch_size
        self.steps = steps
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        for _ in range(self.steps):
            yield random_block_batch(
                self.tokens, self.batch_size, self.block_size, self.generator
            )
