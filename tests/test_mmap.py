import tempfile
import os
from syntrix.data.mmap import MMapText, mmap_random_block_batch


def test_mmap_reads_lines():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "f.txt")
        with open(p, "w") as f:
            f.write("a\n\nb\n")
        lines = list(MMapText(p))
        assert lines == ["a", "", "b"]


def test_mmap_random_block_batch_shapes():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "f.txt")
        with open(p, "w") as f:
            f.write("abcdefghijklmnopqrstuvwxyz\n" * 10)
        xb, yb = mmap_random_block_batch(p, batch_size=4, block_size=16)
        assert xb.shape == (4, 16)
        assert yb.shape == (4, 16)
