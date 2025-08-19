# Benchmarks (CPU)

Hardware: i5-11320H (4c/8t), 16.5 GB RAM, Ubuntu 24.04, Python 3.12

## TinyShakespeare (char)

Command:

```bash
syntrix.train \
  --config configs/gpt-mini.yaml \
  --data.file data/tinyshakespeare.txt \
  --threads 4 \
  --train_steps 300 --eval_every 100 --save_every 150 \
  --out_dir runs/gpt-mini
```

Seed: 1337

Target (MVP):

- gpt_mini (d_model=256, n_layer=4, n_head=4, block=128)
  - <= 1.0 BPC within 10–15 min
  - ~5–15k tok/s

Notes:

- Ensure `torch.set_num_threads(4)`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`.
- Deterministic: fixed seed + identical loss curves across runs.

## Measured Stats (example)

Hardware: i5-11320H, 4 threads

Command:

```bash
syntrix.train \
  --config configs/gpt-mini.yaml \
  --data.file data/tinyshakespeare.txt \
  --threads 4 \
  --train_steps 300 --eval_every 100 --save_every 150 \
  --out_dir runs/gpt-mini
```

Observed (example run):

- Step 100: val bpc ~2.9; tok/s ~30–50
- Step 300: val bpc ~2.4; final checkpoint saved
