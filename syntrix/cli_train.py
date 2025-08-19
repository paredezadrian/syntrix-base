import argparse
import os
import torch
from .utils.seed import set_seed, set_threads, get_dtype
from .utils.config import load_yaml_config
from .data.download import download_text8_mini
from .train import Trainer, TrainArgs


def main(argv=None):
    p = argparse.ArgumentParser(
        "syntrix.train",
        description="Train small models on CPU with deterministic behavior and reproducible logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # System
    p.add_argument("--threads", type=int, default=4, help="Number of PyTorch/BLAS threads to use")
    p.add_argument("--seed", type=int, default=1337, help="Random seed for determinism")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"], help="Default floating point precision")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase console verbosity (-v, -vv)")
    p.add_argument("--compile", action="store_true", help="Enable torch.compile if available")
    p.add_argument("--compile.validate", dest="compile_validate", action="store_true", help="Benchmark forward throughput to validate compile speedup")
    p.add_argument("--compile.auto", dest="compile_auto", action="store_true", help="Auto-enable compile only if validation shows improvement")
    p.add_argument("--compile.min_improvement", dest="compile_min_improvement", type=float, default=1.05, help="Minimum throughput improvement ratio to accept compile in auto mode")

    # Data & IO
    p.add_argument("--data.file", dest="data_file", type=str, required=True, help="Path to input text file")
    p.add_argument("--out_dir", type=str, default="runs/latest", help="Output directory for checkpoints and logs")
    p.add_argument("--config", type=str, default=None, help="YAML config to load as base")
    p.add_argument("--tokenizer", type=str, default="char", choices=["char", "bpe"], help="Tokenizer type")
    p.add_argument("--bpe_vocab_size", type=int, default=256, help="BPE vocabulary size if tokenizer=bpe")
    p.add_argument("--download.text8_mini", dest="dl_text8", action="store_true", help="Download a tiny text8 sample and override --data.file")
    p.add_argument("--data.use_mmap", dest="use_mmap", action="store_true", help="Use memory-mapped block sampler for large files")

    # Model
    p.add_argument("--model", type=str, default="gpt_mini", help="Model type: gpt_mini | rnn_mini | ssm_mini")
    p.add_argument("--vocab_size", type=int, default=128, help="Model vocabulary size (min of tokenizer and this value is used)")
    p.add_argument("--block_size", type=int, default=128, help="Context length / block size")
    p.add_argument("--d_model", type=int, default=256, help="Model hidden dimension")
    p.add_argument("--n_layer", type=int, default=4, help="Number of layers")
    p.add_argument("--n_head", type=int, default=4, help="Number of attention heads (GPT only)")
    p.add_argument("--mlp_ratio", type=int, default=4, help="MLP expansion ratio")

    # Train
    p.add_argument("--batch_size", type=int, default=32, help="Global batch size (may be simulated via grad_accum)")
    p.add_argument("--microbatch", type=int, default=1, help="Per-step microbatch size")
    p.add_argument("--grad_accum", type=int, default=64, help="Gradient accumulation steps")
    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm (0 or negative disables)")
    p.add_argument("--train_steps", type=int, default=300, help="Number of training steps")
    p.add_argument("--eval_every", type=int, default=100, help="Evaluate validation BPC every N steps")
    p.add_argument("--save_every", type=int, default=150, help="Save checkpoint every N steps")

    # Optim
    p.add_argument("--lr", type=float, default=3e-3, help="Base learning rate")
    p.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    p.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    p.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")
    p.add_argument("--warmup_steps", type=int, default=50, help="Cosine schedule warmup steps")
    p.add_argument("--ema", action="store_true", help="Enable Exponential Moving Average of parameters")
    args = p.parse_args(argv)

    set_seed(args.seed)
    set_threads(args.threads)
    dtype = get_dtype(args.dtype)
    torch.set_default_dtype(dtype)

    if args.compile:
        os.environ["SYNTRIX_COMPILE"] = "1"

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
        use_mmap=args.use_mmap,
        verbosity=(1 + int(args.verbose)),
        compile=args.compile,
        compile_validate=args.compile_validate,
        compile_auto=args.compile_auto,
        compile_min_improvement=args.compile_min_improvement,
        ema=args.ema,
        out_dir=args.out_dir,
    )

    trainer = Trainer(train_args)
    trainer.train()


if __name__ == "__main__":
    main()


