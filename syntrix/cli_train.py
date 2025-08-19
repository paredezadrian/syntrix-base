import argparse
import torch
from .utils.seed import set_seed, set_threads, get_dtype
from .train import Trainer, TrainArgs


def main(argv=None):
    p = argparse.ArgumentParser("syntrix.train")
    # System
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])

    # Data & IO
    p.add_argument("--data.file", dest="data_file", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="runs/latest")

    # Model
    p.add_argument("--model", type=str, default="gpt_mini")
    p.add_argument("--vocab_size", type=int, default=128)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--mlp_ratio", type=int, default=4)

    # Train
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--microbatch", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=64)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--train_steps", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=150)

    # Optim
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--ema", action="store_true")
    args = p.parse_args(argv)

    set_seed(args.seed)
    set_threads(args.threads)
    dtype = get_dtype(args.dtype)
    torch.set_default_dtype(dtype)

    train_args = TrainArgs(
        data_file=args.data_file,
        model=args.model,
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        mlp_ratio=args.mlp_ratio,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        grad_accum=args.grad_accum,
        grad_clip=args.grad_clip,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        warmup_steps=args.warmup_steps,
        train_steps=args.train_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        seed=args.seed,
        threads=args.threads,
        dtype=args.dtype,
        ema=args.ema,
        out_dir=args.out_dir,
    )

    trainer = Trainer(train_args)
    trainer.train()


if __name__ == "__main__":
    main()


