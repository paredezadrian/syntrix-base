# Tiny Char Model (CPU)

This example trains a small GPT-style model on TinyShakespeare using only CPU.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e .
mkdir -p data
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data/tinyshakespeare.txt
```

## Train (YAML + overrides)

```bash
syntrix.train \
  --config configs/gpt-mini.yaml \
  --data.file data/tinyshakespeare.txt \
  --train_steps 300 --eval_every 100 --save_every 150 \
  --threads 4 --out_dir runs/gpt-mini
```

## Sample

```bash
syntrix.sample \
  --ckpt runs/gpt-mini/ckpt.pt \
  --data.file data/tinyshakespeare.txt \
  --max_new_tokens 200 --temp 0.9
```
