import torch

from syntrix.models.rnn_mini import RNNMini
from syntrix.models.ssm_mini import SSMMini
from syntrix.train import Trainer, TrainArgs


def test_rnn_mini_shapes():
    B, T, V = 2, 8, 64
    m = RNNMini(vocab_size=V, d_model=32, n_layer=2, block_size=16)
    x = torch.randint(0, V, (B, T))
    y = m(x)
    assert y.shape == (B, T, V)


def test_trainer_model_selection_smoke():
    # Ensure trainer can instantiate each model variant
    import tempfile
    import os

    text = "hello world\n" * 100
    with tempfile.TemporaryDirectory() as td:
        data_path = os.path.join(td, "data.txt")
        with open(data_path, "w") as f:
            f.write(text)
        for name in ("gpt_mini", "rnn_mini", "ssm_mini"):
            args = TrainArgs(
                data_file=data_path,
                model=name,
                d_model=32,
                n_layer=2,
                n_head=4,
                block_size=16,
                train_steps=1,
                eval_every=1,
                save_every=1,
                out_dir=os.path.join(td, name),
            )
            tr = Trainer(args)
            tr.train()


def test_ssm_mini_shapes():
    B, T, V = 2, 8, 64
    m = SSMMini(vocab_size=V, d_model=32, n_layer=2, block_size=16)
    x = torch.randint(0, V, (B, T))
    y = m(x)
    assert y.shape == (B, T, V)


def test_parameter_counts_reasonable():
    # Sanity checks: parameter counts are positive and scale with number of layers within each model family
    import importlib

    V, D, H, B = 64, 32, 4, 16

    def count(m):
        return sum(p.numel() for p in m.parameters())

    GPTMini = importlib.import_module("syntrix.models.gpt_mini").GPTMini
    gpt1 = GPTMini(vocab_size=V, d_model=D, n_layer=1, n_head=H, block_size=B)
    gpt3 = GPTMini(vocab_size=V, d_model=D, n_layer=3, n_head=H, block_size=B)
    assert count(gpt1) > 0 and count(gpt3) > count(gpt1)

    rnn1 = RNNMini(vocab_size=V, d_model=D, n_layer=1, block_size=B)
    rnn3 = RNNMini(vocab_size=V, d_model=D, n_layer=3, block_size=B)
    assert count(rnn1) > 0 and count(rnn3) > count(rnn1)

    ssm1 = SSMMini(vocab_size=V, d_model=D, n_layer=1, block_size=B)
    ssm3 = SSMMini(vocab_size=V, d_model=D, n_layer=3, block_size=B)
    assert count(ssm1) > 0 and count(ssm3) > count(ssm1)


def test_gradient_flow_one_step():
    import torch

    V, D, L, H, B, T = 64, 32, 2, 4, 2, 8
    gpt = __import__("syntrix.models.gpt_mini", fromlist=["GPTMini"]).GPTMini(
        vocab_size=V, d_model=D, n_layer=L, n_head=H, block_size=T
    )
    optim = torch.optim.AdamW(gpt.parameters(), lr=1e-3)
    x = torch.randint(0, V, (B, T))
    y = torch.randint(0, V, (B, T))
    logits = gpt(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, V), y.view(-1))
    loss.backward()
    # grads exist
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in gpt.parameters()
    )
    assert has_grad
    optim.step()
