import torch

from syntrix.utils.seed import set_seed, set_threads, tolerance_for_dtype
from syntrix.data.text import CharTokenizer, random_block_batch
from syntrix.models.gpt_mini import GPTMini


def run_short_training(seed: int = 1234, threads: int = 2):
    set_seed(seed)
    set_threads(threads)

    # Tiny synthetic corpus
    text = "hello world\n" * 50
    tok = CharTokenizer(text)
    tokens = torch.tensor(tok.encode(text), dtype=torch.long)

    # Small model for speed
    model = GPTMini(
        vocab_size=tok.vocab_size, d_model=32, n_layer=2, n_head=4, block_size=16
    )
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))

    losses = []
    for _ in range(5):
        xb, yb = random_block_batch(tokens, batch_size=2, block_size=16)
        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), yb.view(-1)
        )
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        losses.append(loss.item())
    return losses


def test_deterministic_losses_reproduce():
    a = run_short_training()
    b = run_short_training()
    # Identical per-step losses
    assert len(a) == len(b)
    # Use tolerance based on default dtype
    rtol, atol = tolerance_for_dtype(torch.get_default_dtype())
    for la, lb in zip(a, b):
        assert abs(la - lb) <= max(atol, rtol * max(abs(la), abs(lb)))
