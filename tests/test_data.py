import torch

from syntrix.data.text import CharTokenizer, deterministic_split, TokenBatchIterator


def test_char_tokenizer_round_trip():
    text = "hello world"
    tok = CharTokenizer(text)
    ids = tok.encode(text)
    dec = tok.decode(ids)
    assert dec == text


def test_deterministic_split_reproducible():
    tokens = torch.arange(100)
    a_train, a_val = deterministic_split(tokens, val_ratio=0.2, seed=42, shuffle=True)
    b_train, b_val = deterministic_split(tokens, val_ratio=0.2, seed=42, shuffle=True)
    assert torch.equal(a_train, b_train)
    assert torch.equal(a_val, b_val)


def test_token_batch_iterator_shapes():
    tokens = torch.arange(200)
    it = TokenBatchIterator(tokens, block_size=16, batch_size=4, steps=3, seed=7)
    for xb, yb in it:
        assert xb.shape == (4, 16)
        assert yb.shape == (4, 16)
