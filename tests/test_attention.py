import torch
from syntrix.utils.seed import tolerance_for_dtype

from syntrix.nn.layers import RotaryEmbedding
from syntrix.nn.attention import CausalSelfAttention


def test_attention_forward_shapes_cpu():
    torch.manual_seed(0)
    d_model = 16
    num_heads = 4
    T = 8
    B = 2
    rope = RotaryEmbedding(d_model)
    attn = CausalSelfAttention(d_model=d_model, num_heads=num_heads, rope=rope)
    x = torch.randn(B, T, d_model)
    y = attn(x)
    assert y.shape == (B, T, d_model)
    assert y.dtype == x.dtype


def test_causal_masking():
    torch.manual_seed(0)
    d_model = 8
    num_heads = 2
    T = 4
    B = 1
    attn = CausalSelfAttention(d_model=d_model, num_heads=num_heads, rope=None)
    # craft k/v so that the most recent token would have large affinity to future positions
    x = torch.zeros(B, T, d_model)
    # Make queries depend on position via input; but we rely on causal mask to zero future
    x[0, :, 0] = torch.arange(T)
    y, weights = attn(x, return_attn_weights=True)
    # weights: (B, H, T, T); ensure strictly upper triangle is zero after softmax
    upper = torch.triu(weights[0, 0], diagonal=1)
    rtol, atol = tolerance_for_dtype(upper.dtype)
    assert torch.allclose(upper, torch.zeros_like(upper), rtol=rtol, atol=atol)
