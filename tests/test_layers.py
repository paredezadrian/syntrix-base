import torch
from syntrix.utils.seed import tolerance_for_dtype

from syntrix.nn.layers import RMSNorm, SwiGLU, RotaryEmbedding


def test_rmsnorm_shapes_and_dtype():
    x = torch.randn(2, 3, 8, dtype=torch.float32)
    norm = RMSNorm(8)
    y = norm(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_swiglu_shapes_and_dtype():
    x = torch.randn(4, 16, dtype=torch.float32)
    m = SwiGLU(16, 32)
    y = m(x)
    assert y.shape == (4, 32)
    assert y.dtype == x.dtype


def test_rope_rotation_toy():
    dim = 8
    T = 4
    rope = RotaryEmbedding(dim)
    x = torch.zeros(1, T, dim)
    # Set a simple basis on even dims to observe rotation into odd dims
    x[0, :, 0] = 1.0
    x[0, :, 2] = 2.0
    cos, sin = rope.get_cos_sin(T, x.device, x.dtype)
    y = rope.apply_rotary(x, cos, sin)
    # Check shape and that odd dims receive non-zero due to rotation (except t=0)
    assert y.shape == x.shape
    rtol, atol = tolerance_for_dtype(y.dtype)
    assert torch.allclose(
        y[0, 0, 1::2],
        torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=y.dtype),
        rtol=rtol,
        atol=atol,
    )
    assert (y[0, 1:, 1::2].abs().sum() > 0).item()
