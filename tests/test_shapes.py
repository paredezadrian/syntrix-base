import torch

from syntrix.models.gpt_mini import GPTMini


def test_gpt_mini_logits_shape_dtype():
    torch.manual_seed(0)
    B, T, vocab = 2, 8, 128
    model = GPTMini(
        vocab_size=vocab, d_model=32, n_layer=2, n_head=4, block_size=16, mlp_ratio=4
    )
    x = torch.randint(0, vocab, (B, T))
    logits = model(x)
    assert logits.shape == (B, T, vocab)
    assert logits.dtype == torch.float32
