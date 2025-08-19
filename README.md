# Syntrix‑Base — A Low‑Resource (CPU‑First) Machine Learning Framework

[![CI](https://github.com/paredezadrian/syntrix-base/actions/workflows/ci.yaml/badge.svg)](https://github.com/paredezadrian/syntrix-base/actions/workflows/ci.yaml)
[![CodeQL](https://github.com/paredezadrian/syntrix-base/actions/workflows/codeql.yml/badge.svg)](https://github.com/paredezadrian/syntrix-base/actions/workflows/codeql.yml)
[![License: MIT+Commons Clause](https://img.shields.io/badge/License-MIT%20%2B%20Commons%20Clause-orange.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/syntrix.svg)](https://pypi.org/project/syntrix/)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/syntrix/)

Train and run modern small models fast on everyday CPUs — simple, transparent, and reproducible.

## Table of Contents

- Why Syntrix‑Base?
- Highlights
- Requirements
- Quickstart (pip and from source)
- Usage (Train, Sample, Eval, Config)
- Configuration overrides
- Reproducibility & Determinism
- Benchmarks
- Troubleshooting
- Contributing & Governance
- License

## Why Syntrix‑Base?

Syntrix‑Base is a CPU‑first, deterministic learning toolkit for tiny but modern models. It emphasizes clarity over complexity: clean PyTorch code, reproducible logs, and practical CLIs that work well on everyday hardware.

## Highlights

- CPU‑first ergonomics: pinned threads, deterministic seeds, dtype control
- Tiny but modern models: GPT‑mini, SSM‑mini, RNN‑mini
- Reproducible logging: JSONL logs with tokens/sec and environment
- Optional `torch.compile` with CLI toggle and auto validation

## Requirements

- Python >= 3.9
- Linux/macOS (Windows may work via WSL)
- PyTorch (installed automatically via `pip install -e .`)

## Quickstart

### Install (official, from PyPI)
```bash
pip install syntrix
```

### Alternative: From source (dev install)
```bash
git clone https://github.com/paredezadrian/syntrix-base.git
cd syntrix-base
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### 2) Get sample data (TinyShakespeare)
```bash
mkdir -p data
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data/tinyshakespeare.txt
```

Or download a mini text8 sample via CLI when training: add `--download.text8_mini`.

### 3) Train (YAML + CLI overrides)
```bash
syntrix.train \
  --config configs/gpt-mini.yaml \
  --data.file data/tinyshakespeare.txt \
  --train_steps 300 --eval_every 100 --save_every 150 \
  --threads 4 \
  --out_dir runs/gpt-mini_base
```

Enable `torch.compile` and auto‑validate throughput (accept only if >= 5% faster):
```bash
syntrix.train \
  --config configs/gpt-mini.yaml \
  --data.file data/tinyshakespeare.txt \
  --threads 4 \
  --train_steps 300 --eval_every 100 --save_every 150 \
  --compile --compile.validate --compile.auto --compile.min_improvement 1.05 \
  --out_dir runs/gpt-mini_compile_auto
```

Outputs:
- Checkpoints: `runs/<name>/ckpt.pt`
- Logs (JSONL): `runs/<name>/log.jsonl` with `step`, `val_bpc`, `tokens_per_s`, `lr`, and an initial `env` record (Python, PyTorch, threads, dtype, compiled flag).

### 4) Sample from a checkpoint
```bash
syntrix.sample \
  --ckpt runs/gpt-mini_base/ckpt.pt \
  --data.file data/tinyshakespeare.txt \
  --max_new_tokens 200 --temp 0.9
```

### 5) Evaluate or validate config
```bash
# Evaluate a checkpoint (reports validation BPC)
syntrix.eval --data.file data/tinyshakespeare.txt --ckpt runs/gpt-mini_base/ckpt.pt

# Validate and inspect a YAML config
syntrix.config --config configs/gpt-mini.yaml
```

## Configuration

- Base configs live in `configs/` (e.g., `configs/gpt-mini.yaml`).
- You can override most settings via CLI flags. Some use dot notation (e.g., `--data.file`, `--download.text8_mini`).
- Examples:
```bash
# Increase layers and reduce batch using dot-notation overrides
syntrix.train --config configs/gpt-mini.yaml --data.file data/tinyshakespeare.txt \
  --model.n_layer 6 --train.batch_size 16
```
- Precision: switch default dtype with `--dtype float32|float64`. Numeric tests use dtype‑aware tolerances.

## CLI Reference

List options:
```bash
python -m syntrix.cli_train --help
python -m syntrix.cli_sample --help
```

Notable flags:
- `--threads <int>`: sets `torch.set_num_threads` and pins MKL/OMP threads
- `--compile`: enable `torch.compile` if available
- `--compile.validate --compile.auto --compile.min_improvement 1.05`: benchmark forward throughput and auto‑enable compile only if faster
- `--tokenizer <char|bpe>` and `--bpe_vocab_size <int>`
- `--use_mmap`: use memory‑mapped data loader for large files

## Reproducibility & Determinism

- Seeds and threads are initialized consistently via `syntrix.utils.seed`
- Logs record environment details (threads, Python, PyTorch, dtype)
- Tests cover determinism and tolerance‑aware numeric checks

## Benchmarks

For reproducible commands and example results tables, see `docs/benchmarks.md` and architecture/FAQ in `docs/architecture.md`.

## Troubleshooting

- Non‑deterministic results:
  - Ensure `--seed` and `--threads` are set; check `OMP_NUM_THREADS` and `MKL_NUM_THREADS`.
- Slow throughput:
  - Use smaller `--block_size`, small `--microbatch` with higher `--grad_accum`, and try `--compile --compile.validate --compile.auto`.
- Memory constraints:
  - Use `--data.use_mmap` for memory‑mapped random block sampling on large files.

## Contributing

We welcome contributions of all kinds: bug fixes, features, docs, and benchmarks.

- Please read `CONTRIBUTING.md` for our contribution process, standards, and PR guidelines
- All participants are expected to follow our `CODE_OF_CONDUCT.md`

## Governance & Support

- Issues: Use GitHub Issues for bug reports and feature requests. Include OS, Python, and PyTorch versions, steps to reproduce, and expected vs. actual behavior.
- CI: Pull requests must pass GitHub Actions (pytest on Python 3.10/3.11/3.12).

## License

MIT with Commons Clause (non‑commercial). See `LICENSE` for details.

 
