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
    import tempfile, os
    text = "hello world\n" * 100
    with tempfile.TemporaryDirectory() as td:
        data_path = os.path.join(td, "data.txt")
        with open(data_path, "w") as f:
            f.write(text)
        for name in ("gpt_mini", "rnn_mini", "ssm_mini"):
            args = TrainArgs(data_file=data_path, model=name, d_model=32, n_layer=2, n_head=4, block_size=16, train_steps=1, eval_every=1, save_every=1, out_dir=os.path.join(td, name))
            tr = Trainer(args)
            tr.train()


def test_ssm_mini_shapes():
    B, T, V = 2, 8, 64
    m = SSMMini(vocab_size=V, d_model=32, n_layer=2, block_size=16)
    x = torch.randint(0, V, (B, T))
    y = m(x)
    assert y.shape == (B, T, V)


