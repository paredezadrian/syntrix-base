from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .models.gpt_mini import GPTMini
from .models.rnn_mini import RNNMini
from .models.ssm_mini import SSMMini
from .optim.schedule import CosineWithWarmup
from .optim.ema import EMA
from .utils.seed import set_seed, set_threads, get_dtype
from .data.text import CharTokenizer, load_text_file
import platform


@dataclass
class TrainArgs:
    data_file: str
    model: str = "gpt_mini"
    vocab_size: int = 128
    block_size: int = 128
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    mlp_ratio: int = 4

    batch_size: int = 32
    microbatch: int = 1
    grad_accum: int = 64
    grad_clip: float = 1.0

    lr: float = 3e-3
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 50
    train_steps: int = 300
    eval_every: int = 100
    save_every: int = 150

    seed: int = 1337
    threads: int = 4
    dtype: str = "float32"
    ema: bool = False
    out_dir: str = "runs/latest"


def tokens_from_text(text: str, tokenizer: CharTokenizer) -> torch.Tensor:
    ids = tokenizer.encode(text)
    return torch.tensor(ids, dtype=torch.long)


def split_train_val(tokens: torch.Tensor, val_ratio: float = 0.05, seed: int = 1337) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    n = tokens.numel()
    split = int(n * (1 - val_ratio))
    return tokens[:split], tokens[split:]


def sample_batch(tokens: torch.Tensor, batch_size: int, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Random contiguous blocks
    n = tokens.numel() - block_size - 1
    idx = torch.randint(0, max(n, 1), (batch_size,))
    x = torch.stack([tokens[i : i + block_size] for i in idx])
    y = torch.stack([tokens[i + 1 : i + 1 + block_size] for i in idx])
    return x, y


def evaluate_bpc(model: torch.nn.Module, tokens: torch.Tensor, block_size: int, iters: int = 20, batch_size: int = 32) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(iters):
            xb, yb = sample_batch(tokens, batch_size, block_size)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            losses.append(loss.item())
    model.train()
    mean_loss = sum(losses) / max(1, len(losses))
    return mean_loss / math.log(2.0)


class Trainer:
    def __init__(self, args: TrainArgs):
        self.args = args

        set_seed(args.seed)
        set_threads(args.threads)
        torch.set_default_dtype(get_dtype(args.dtype))

        raw_text = load_text_file(args.data_file)
        self.tokenizer = CharTokenizer(raw_text)
        tokens = tokens_from_text(raw_text, self.tokenizer)
        self.train_tokens, self.val_tokens = split_train_val(tokens, val_ratio=0.05, seed=args.seed)

        vocab_size = max(self.tokenizer.vocab_size, args.vocab_size)
        if args.model == "gpt_mini":
            self.model = GPTMini(
                vocab_size=vocab_size,
                d_model=args.d_model,
                n_layer=args.n_layer,
                n_head=args.n_head,
                block_size=args.block_size,
                mlp_ratio=args.mlp_ratio,
            )
        elif args.model == "rnn_mini":
            self.model = RNNMini(
                vocab_size=vocab_size,
                d_model=args.d_model,
                n_layer=args.n_layer,
                block_size=args.block_size,
                mlp_ratio=args.mlp_ratio,
            )
        elif args.model == "ssm_mini":
            self.model = SSMMini(
                vocab_size=vocab_size,
                d_model=args.d_model,
                n_layer=args.n_layer,
                block_size=args.block_size,
                mlp_ratio=args.mlp_ratio,
            )
        else:
            raise ValueError(f"Unknown model {args.model}")

        self.optimizer = AdamW(self.model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
        self.scheduler = CosineWithWarmup(
            base_lr=args.lr, warmup_steps=args.warmup_steps, total_steps=args.train_steps, min_lr=0.0
        )
        self.ema = EMA(self.model.parameters()) if args.ema else None

        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        self.log_file = Path(args.out_dir) / "log.jsonl"

    def _log(self, data: dict) -> None:
        with self.log_file.open("a") as f:
            f.write(json.dumps(data) + "\n")

    def train(self) -> None:
        args = self.args
        model = self.model
        model.train()

        global_step = 0
        t0 = time.time()
        # Log environment/system info once at start
        self._log(
            {
                "event": "env",
                "python": platform.python_version(),
                "torch": torch.__version__,
                "threads": {
                    "torch_num_threads": torch.get_num_threads(),
                    "omp": os.environ.get("OMP_NUM_THREADS"),
                    "mkl": os.environ.get("MKL_NUM_THREADS"),
                },
                "dtype": str(torch.get_default_dtype()),
            }
        )
        for step in range(1, args.train_steps + 1):
            self.optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for i in range(args.grad_accum):
                xb, yb = sample_batch(self.train_tokens, args.microbatch, args.block_size)
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
                (loss / args.grad_accum).backward()
                loss_accum += loss.item()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            self.optimizer.step()
            lr = self.scheduler.step()
            if self.ema is not None:
                self.ema.update(model.parameters())

            global_step += 1

            if step % args.eval_every == 0 or step == 1:
                bpc = evaluate_bpc(model, self.val_tokens, args.block_size, iters=10, batch_size=min(16, args.batch_size))
                tok_s = (args.grad_accum * args.microbatch * args.block_size) / max(1e-6, (time.time() - t0))
                print(f"step {step} | loss {loss_accum/args.grad_accum:.3f} | val bpc {bpc:.3f} | lr {lr:.2e} | tok/s {tok_s:.0f}")
                self._log({
                    "step": step,
                    "loss": loss_accum/args.grad_accum,
                    "val_bpc": bpc,
                    "lr": lr,
                    "tokens_per_s": tok_s,
                })
                t0 = time.time()

            if step % args.save_every == 0 or step == args.train_steps:
                ckpt_path = Path(args.out_dir) / "ckpt.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "step": step,
                        "meta": {
                            "vocab_size": self.model.vocab_size,
                            "d_model": self.args.d_model,
                            "n_layer": self.args.n_layer,
                            "n_head": self.args.n_head,
                            "block_size": self.args.block_size,
                            "mlp_ratio": self.args.mlp_ratio,
                            "chars": self.tokenizer.chars,
                        },
                    },
                    ckpt_path,
                )


