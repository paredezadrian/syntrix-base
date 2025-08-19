import torch
from syntrix.utils.seed import tolerance_for_dtype


def test_tolerance_values():
    rtol32, atol32 = tolerance_for_dtype(torch.float32)
    rtol64, atol64 = tolerance_for_dtype(torch.float64)
    assert rtol64 < rtol32
    assert atol64 < atol32
