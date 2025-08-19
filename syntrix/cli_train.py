import argparse
import sys
import os
import torch
from .utils.seed import set_seed, set_threads, get_dtype
from .utils.config import load_yaml_config, Config
from .data.download import download_text8_mini
from .train import Trainer, TrainArgs


def main(argv=None):
    p = argparse.ArgumentParser(
        "syntrix.train",
        description="Train small models on CPU with deterministic behavior and reproducible logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # System
    p.add_argument(
        "--threads", type=int, default=4, help="Number of PyTorch/BLAS threads to use"
    )
    p.add_argument("--seed", type=int, default=1337, help="Random seed for determinism")
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Default floating point precision",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase console verbosity (-v, -vv)",
    )
    p.add_argument(
        "--compile", action="store_true", help="Enable torch.compile if available"
    )
    p.add_argument(
        "--compile.validate",
        dest="compile_validate",
        action="store_true",
        help="Benchmark forward throughput to validate compile speedup",
    )
    p.add_argument(
        "--compile.auto",
        dest="compile_auto",
        action="store_true",
        help="Auto-enable compile only if validation shows improvement",
    )
    p.add_argument(
        "--compile.min_improvement",
        dest="compile_min_improvement",
        type=float,
        default=1.05,
        help="Minimum throughput improvement ratio to accept compile in auto mode",
    )

    # Data & IO
    p.add_argument(
        "--data.file",
        dest="data_file",
        type=str,
        required=True,
        help="Path to input text file",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="runs/latest",
        help="Output directory for checkpoints and logs",
    )
    p.add_argument(
        "--config", type=str, default=None, help="YAML config to load as base"
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default="char",
        choices=["char", "bpe"],
        help="Tokenizer type",
    )
    p.add_argument(
        "--bpe_vocab_size",
        type=int,
        default=256,
        help="BPE vocabulary size if tokenizer=bpe",
    )
    p.add_argument(
        "--download.text8_mini",
        dest="dl_text8",
        action="store_true",
        help="Download a tiny text8 sample and override --data.file",
    )
    p.add_argument(
        "--data.use_mmap",
        dest="use_mmap",
        action="store_true",
        help="Use memory-mapped block sampler for large files",
    )

    # Model
    p.add_argument(
        "--model",
        type=str,
        default="gpt_mini",
        help="Model type: gpt_mini | rnn_mini | ssm_mini",
    )
    p.add_argument(
        "--vocab_size",
        type=int,
        default=128,
        help="Model vocabulary size (min of tokenizer and this value is used)",
    )
    p.add_argument(
        "--block_size", type=int, default=128, help="Context length / block size"
    )
    p.add_argument("--d_model", type=int, default=256, help="Model hidden dimension")
    p.add_argument("--n_layer", type=int, default=4, help="Number of layers")
    p.add_argument(
        "--n_head", type=int, default=4, help="Number of attention heads (GPT only)"
    )
    p.add_argument("--mlp_ratio", type=int, default=4, help="MLP expansion ratio")

    # Train
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Global batch size (may be simulated via grad_accum)",
    )
    p.add_argument("--microbatch", type=int, default=1, help="Per-step microbatch size")
    p.add_argument(
        "--grad_accum", type=int, default=64, help="Gradient accumulation steps"
    )
    p.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm (0 or negative disables)",
    )
    p.add_argument(
        "--train_steps", type=int, default=300, help="Number of training steps"
    )
    p.add_argument(
        "--eval_every",
        type=int,
        default=100,
        help="Evaluate validation BPC every N steps",
    )
    p.add_argument(
        "--save_every", type=int, default=150, help="Save checkpoint every N steps"
    )

    # Optim
    p.add_argument("--lr", type=float, default=3e-3, help="Base learning rate")
    p.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    p.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    p.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")
    p.add_argument(
        "--warmup_steps", type=int, default=50, help="Cosine schedule warmup steps"
    )
    p.add_argument(
        "--ema",
        action="store_true",
        help="Enable Exponential Moving Average of parameters",
    )
    args, unknown = p.parse_known_args(argv)

    set_seed(args.seed)
    set_threads(args.threads)
    dtype = get_dtype(args.dtype)
    torch.set_default_dtype(dtype)

    if args.compile:
        os.environ["SYNTRIX_COMPILE"] = "1"

    # Load base config or defaults
    cfg = load_yaml_config(args.config) if args.config else Config()
    model = cfg.model
    train_cfg = cfg.train
    optim = cfg.optim

    # Apply dot-notation overrides from unknown args (e.g., --model.n_layer 6, --train.batch_size 16)
    def _infer_type(value: str):
        vl = value.lower()
        if vl in ("true", "false"):
            return vl == "true"
        try:
            if any(ch in value for ch in (".", "e", "E")):
                return float(value)
            return int(value)
        except ValueError:
            return value

    i = 0
    while i < len(unknown):
        token = unknown[i]
        if token.startswith("--") and "." in token:
            key = token[2:]
            if "=" in key:
                dotted, val = key.split("=", 1)
            else:
                # value may be next token
                if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                    dotted, val = key, unknown[i + 1]
                    i += 1
                else:
                    dotted, val = key, "true"
            parts = dotted.split(".", 1)
            if len(parts) == 2:
                section, name = parts
                tgt = getattr(cfg, section, None)
                if tgt is not None and hasattr(tgt, name):
                    setattr(tgt, name, _infer_type(val))
        i += 1

    if args.dl_text8:
        # download minimal text8 if requested and override data_file
        args.data_file = download_text8_mini()

    train_args = TrainArgs(
        data_file=args.data_file,
        model=(args.model if args.model is not None else model.model),
        vocab_size=(
            args.vocab_size if args.vocab_size is not None else model.vocab_size
        ),
        block_size=(
            args.block_size if args.block_size is not None else model.block_size
        ),
        d_model=(args.d_model if args.d_model is not None else model.d_model),
        n_layer=(args.n_layer if args.n_layer is not None else model.n_layer),
        n_head=(
            args.n_head if args.n_head is not None else getattr(model, "n_head", 4)
        ),
        mlp_ratio=(args.mlp_ratio if args.mlp_ratio is not None else model.mlp_ratio),
        batch_size=(
            args.batch_size if args.batch_size is not None else train_cfg.batch_size
        ),
        microbatch=(
            args.microbatch if args.microbatch is not None else train_cfg.microbatch
        ),
        grad_accum=(
            args.grad_accum if args.grad_accum is not None else train_cfg.grad_accum
        ),
        grad_clip=(
            args.grad_clip if args.grad_clip is not None else train_cfg.grad_clip
        ),
        lr=(args.lr if args.lr is not None else optim.lr),
        weight_decay=(
            args.weight_decay if args.weight_decay is not None else optim.weight_decay
        ),
        betas=(
            (args.beta1, args.beta2)
            if (args.beta1 is not None and args.beta2 is not None)
            else optim.betas
        ),
        warmup_steps=(
            args.warmup_steps
            if args.warmup_steps is not None
            else cfg.schedule.warmup_steps
        ),
        train_steps=(
            args.train_steps if args.train_steps is not None else train_cfg.train_steps
        ),
        eval_every=(
            args.eval_every if args.eval_every is not None else train_cfg.eval_every
        ),
        save_every=(
            args.save_every if args.save_every is not None else train_cfg.save_every
        ),
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

    # Validation with actionable errors
    errors = []
    if train_args.batch_size <= 0:
        errors.append("--batch_size must be > 0")
    if train_args.microbatch <= 0:
        errors.append("--microbatch must be > 0")
    if train_args.grad_accum <= 0:
        errors.append("--grad_accum must be > 0")
    if train_args.block_size <= 0:
        errors.append("--block_size must be > 0")
    if train_args.n_layer <= 0:
        errors.append("--n_layer must be > 0")
    if train_args.d_model <= 0:
        errors.append("--d_model must be > 0")
    if train_args.lr <= 0:
        errors.append("--lr must be > 0")
    if train_args.warmup_steps < 0:
        errors.append("--warmup_steps cannot be negative")
    if train_args.eval_every <= 0 or train_args.save_every <= 0:
        errors.append("--eval_every and --save_every must be > 0")
    if train_args.train_steps <= 0:
        errors.append("--train_steps must be > 0")
    if train_args.dtype not in ("float32", "float64"):
        errors.append("--dtype must be float32 or float64")
    if errors:
        for e in errors:
            print(f"Config error: {e}")
        sys.exit(2)

    trainer = Trainer(train_args)
    trainer.train()


if __name__ == "__main__":
    main()
