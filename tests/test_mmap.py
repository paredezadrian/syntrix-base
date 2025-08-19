import tempfile, os
from syntrix.data.mmap import MMapText


def test_mmap_reads_lines():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "f.txt")
        with open(p, "w") as f:
            f.write("a\n\nb\n")
        lines = list(MMapText(p))
        assert lines == ["a", "", "b"]


