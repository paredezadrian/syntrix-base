# Syntrix‑Base — A Low‑Resource (CPU‑First) Machine Learning Framework

[![CI](https://github.com/paredezadrian/syntrix-base/actions/workflows/ci.yaml/badge.svg)](https://github.com/paredezadrian/syntrix-base/actions/workflows/ci.yaml)
[![CodeQL](https://github.com/paredezadrian/syntrix-base/actions/workflows/codeql.yml/badge.svg)](https://github.com/paredezadrian/syntrix-base/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/syntrix.svg)](https://pypi.org/project/syntrix/)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/syntrix/)

Train and run modern small models fast on everyday CPUs — simple, transparent, and reproducible.

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

### 1) Clone and install
```bash
git clone https://github.com/paredezadrian/syntrix-base.git
cd syntrix-base
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Install from PyPI (after publish):
```bash
pip install syntrix
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

## Configuration

- Base configs live in `configs/` (e.g., `configs/gpt-mini.yaml`).
- You can override most settings via CLI flags. Some use dot notation (e.g., `--data.file`, `--download.text8_mini`).
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

For reproducible commands and example results tables, see `docs/benchmarks.md`.

## Contributing

We welcome contributions of all kinds: bug fixes, features, docs, and benchmarks.

- Please read `CONTRIBUTING.md` for our contribution process, standards, and PR guidelines
- All participants are expected to follow our `CODE_OF_CONDUCT.md`

## Governance & Support

- Issues: Use GitHub Issues for bug reports and feature requests. Include OS, Python, and PyTorch versions, steps to reproduce, and expected vs. actual behavior.
- CI: Pull requests must pass GitHub Actions (pytest on Python 3.10/3.11/3.12).

## License

MIT — see `LICENSE`.

## Packaging & Publishing

### Versioning Policy

We follow Semantic Versioning (SemVer): `MAJOR.MINOR.PATCH`.
- Increase MAJOR for incompatible API changes.
- Increase MINOR for added functionality in a backward-compatible manner.
- Increase PATCH for backward-compatible bug fixes.

### Release Checklist

1. Ensure all tests pass locally and in CI.
2. Update `CHANGELOG.md` with a new section for the release.
3. Bump the version in `pyproject.toml`.
4. Commit changes and tag the release:
   - `git commit -m "chore(release): bump version to X.Y.Z"`
   - `git tag vX.Y.Z && git push origin main --tags`
5. Build and upload to PyPI:
   - `pip install build twine`
   - `python -m build`
   - `twine upload dist/*`
6. Create a GitHub Release referencing the tag and the corresponding changelog notes.
7. Verify README badges (CI, CodeQL, PyPI) render correctly.

### Installing from PyPI (post‑publish)

```bash
pip install syntrix
# verify CLI entry points
syntrix.train --help
syntrix.sample --help
```
