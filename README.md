# Syntrix‑Base — A Low‑Resource (CPU‑First) Machine Learning Framework

Train and run modern small models fast on everyday CPUs — simple, transparent, and reproducible.

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -e .
mkdir -p data
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data/tinyshakespeare.txt

# Train (YAML + overrides)
syntrix.train \
  --config configs/gpt-mini.yaml \
  --data.file data/tinyshakespeare.txt \
  --train_steps 300 --eval_every 100 --save_every 150 \
  --threads 4 --out_dir runs/gpt-mini

# Sample
syntrix.sample \
  --ckpt runs/gpt-mini/ckpt.pt \
  --data.file data/tinyshakespeare.txt \
  --max_new_tokens 200 --temp 0.9
```

See `configs/` for example configs and `examples/tiny_char_model.md`.
For CPU performance guidance and reproducible runs, see `docs/benchmarks.md`.

## Contributing

We welcome contributions of all kinds: bug fixes, new features, documentation, and benchmarks.

- Please read `CONTRIBUTING.md` for our contribution process, coding/testing/documentation standards, and PR guidelines.
- All participants are expected to follow our `CODE_OF_CONDUCT.md`.

## Governance & Support

- Issues: Use GitHub Issues for bug reports and feature requests. Include OS, Python, and PyTorch versions, steps to reproduce, and expected vs. actual behavior.
- CI: Pull requests must pass the GitHub Actions CI (pytest on Python 3.10/3.11/3.12).
