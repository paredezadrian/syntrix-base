import argparse
import torch
from .utils.seed import set_seed, set_threads, get_dtype
from .utils.config import load_yaml_config
from .data.download import download_text8_mini
from .train import Trainer, TrainArgs


def main(argv=None):
    p = argparse.ArgumentParser("syntrix.train")
    # System
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--compile", action="store_true")

    # Data & IO
    p.add_argument("--data.file", dest="data_file", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="runs/latest")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--tokenizer", type=str, default="char", choices=["char", "bpe"])
    p.add_argument("--bpe_vocab_size", type=int, default=256)
    p.add_argument("--download.text8_mini", dest="dl_text8", action="store_true")

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

    if args.dl_text8:
        # download minimal text8 if requested and override data_file
        args.data_file = download_text8_mini()

    train_args = TrainArgs(
        data_file=args.data_file,
        model=(args.model if args.model is not None else (model.model if model else "gpt_mini")),
        vocab_size=(args.vocab_size if args.vocab_size is not None else (model.vocab_size if model else 128)),
        block_size=(args.block_size if args.block_size is not None else (model.block_size if model else 128)),
        d_model=(args.d_model if args.d_model is not None else (model.d_model if model else 256)),
        n_layer=(args.n_layer if args.n_layer is not None else (model.n_layer if model else 4)),
        n_head=(args.n_head if args.n_head is not None else (model.n_head if model else 4)),
        mlp_ratio=(args.mlp_ratio if args.mlp_ratio is not None else (model.mlp_ratio if model else 4)),
        batch_size=(args.batch_size if args.batch_size is not None else (train_cfg.batch_size if train_cfg else 32)),
        microbatch=(args.microbatch if args.microbatch is not None else (train_cfg.microbatch if train_cfg else 1)),
        grad_accum=(args.grad_accum if args.grad_accum is not None else (train_cfg.grad_accum if train_cfg else 64)),
        grad_clip=(args.grad_clip if args.grad_clip is not None else (train_cfg.grad_clip if train_cfg else 1.0)),
        lr=(args.lr if args.lr is not None else (optim.lr if optim else 3e-3)),
        weight_decay=(args.weight_decay if args.weight_decay is not None else (optim.weight_decay if optim else 0.1)),
        betas=((args.beta1, args.beta2) if (args.beta1 is not None and args.beta2 is not None) else (optim.betas if optim else (0.9, 0.95))),
        warmup_steps=(args.warmup_steps if args.warmup_steps is not None else (cfg.schedule.warmup_steps if cfg else 50)),
        train_steps=(args.train_steps if args.train_steps is not None else (train_cfg.train_steps if train_cfg else 300)),
        eval_every=(args.eval_every if args.eval_every is not None else (train_cfg.eval_every if train_cfg else 100)),
        save_every=(args.save_every if args.save_every is not None else (train_cfg.save_every if train_cfg else 200)),
        seed=args.seed,
        threads=args.threads,
        dtype=args.dtype,
        tokenizer=args.tokenizer,
        bpe_vocab_size=args.bpe_vocab_size,
        # compile flag is recognized but not used directly in TrainArgs; Trainer can read from env/args if extended
        ema=args.ema,
        out_dir=args.out_dir,
    )

    trainer = Trainer(train_args)
    trainer.train()


if __name__ == "__main__":
    main()


