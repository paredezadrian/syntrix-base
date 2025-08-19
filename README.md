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
