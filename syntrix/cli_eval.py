import argparse
import torch
from .utils.seed import set_seed, get_dtype
from .train import Trainer, TrainArgs


def main(argv=None):
    p = argparse.ArgumentParser(
        "syntrix.eval",
        description="Evaluate a model checkpoint on validation data and report BPC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=str, default=None, help="Base YAML config (optional)"
    )
    p.add_argument(
        "--data.file",
        dest="data_file",
        type=str,
        required=True,
        help="Path to input text file",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        required=False,
        help="Checkpoint path (if omitted, a fresh model is evaluated)",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Default floating point precision",
    )
    p.add_argument("--seed", type=int, default=1337, help="Random seed")
    p.add_argument("--threads", type=int, default=4, help="PyTorch/BLAS threads")
    p.add_argument(
        "--model",
        type=str,
        default="gpt_mini",
        help="Model type if building fresh model",
    )
    p.add_argument(
        "--block_size", type=int, default=128, help="Context length for evaluation"
    )
    args = p.parse_args(argv)

    set_seed(args.seed)
    torch.set_default_dtype(get_dtype(args.dtype))

    train_args = TrainArgs(
        data_file=args.data_file,
        model=args.model,
        block_size=args.block_size,
        train_steps=1,
        eval_every=1,
        save_every=1,
        out_dir="runs/eval",
    )
    tr = Trainer(train_args)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        tr.model.load_state_dict(state["model"])  # best-effort
    bpc = tr._Trainer__class__.__dict__["evaluate_bpc"](
        tr.model, tr.val_tokens, tr.args.block_size, iters=20, batch_size=16
    )  # type: ignore
    print(f"val_bpc: {bpc:.4f}")


if __name__ == "__main__":
    main()
