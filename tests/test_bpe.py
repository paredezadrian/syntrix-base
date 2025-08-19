from syntrix.data.bpe import BPETokenizer


def test_bpe_round_trip_small_text():
    text = "banana bandana"
    tok = BPETokenizer(text, vocab_size=64)
    ids = tok.encode(text)
    dec = tok.decode(ids)
    assert dec == text
