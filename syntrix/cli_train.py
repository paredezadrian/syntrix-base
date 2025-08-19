import argparse
import torch
from .utils.seed import set_seed, set_threads, get_dtype
from .utils.config import load_yaml_config
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
    p.add_argument("--config", type=str, default=None)

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

    # Load base config if provided, then override with CLI flags
    if args.config:
        cfg = load_yaml_config(args.config)
        model = cfg.model
        train_cfg = cfg.train
        optim = cfg.optim
    else:
        cfg = None
        model = None
        train_cfg = None
        optim = None

    train_args = TrainArgs(
        data_file=args.data_file,
        model=(model.model if model else args.model),
        vocab_size=(model.vocab_size if model else args.vocab_size),
        block_size=(model.block_size if model else args.block_size),
        d_model=(model.d_model if model else args.d_model),
        n_layer=(model.n_layer if model else args.n_layer),
        n_head=(model.n_head if model else args.n_head),
        mlp_ratio=(model.mlp_ratio if model else args.mlp_ratio),
        batch_size=(train_cfg.batch_size if train_cfg else args.batch_size),
        microbatch=(train_cfg.microbatch if train_cfg else args.microbatch),
        grad_accum=(train_cfg.grad_accum if train_cfg else args.grad_accum),
        grad_clip=(train_cfg.grad_clip if train_cfg else args.grad_clip),
        lr=(optim.lr if optim else args.lr),
        weight_decay=(optim.weight_decay if optim else args.weight_decay),
        betas=(optim.betas if optim else (args.beta1, args.beta2)),
        warmup_steps=(cfg.schedule.warmup_steps if cfg else args.warmup_steps),
        train_steps=(train_cfg.train_steps if train_cfg and hasattr(train_cfg, 'train_steps') else args.train_steps),
        eval_every=(train_cfg.eval_every if train_cfg else args.eval_every),
        save_every=(train_cfg.save_every if train_cfg else args.save_every),
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


