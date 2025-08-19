# Benchmarks (CPU)

This document provides reproducible CPU benchmarks for `syntrix` with complete commands and results tables, including comparisons with `torch.compile`.

## Testbed

- CPU: i5-11320H (4c/8t)
- RAM: 16.5 GB
- OS: Ubuntu 24.04
- Python: 3.12
- PyTorch: see runtime log line recorded below
- Default dtype: float32
- Seed: 1337

Threading (CPU-first):
- Set threads via CLI and environment for reproducibility: `--threads 4` and ensure `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4` (the CLI also sets these internally).

## Datasets

- TinyShakespeare (char-level): place file at `data/tinyshakespeare.txt`.
- Text8 mini (download helper): add `--download.text8_mini` to CLI to fetch a small reproducible sample and override `--data.file` automatically.

## How metrics are recorded

Each run writes JSONL logs to `<out_dir>/log.jsonl` including:

- `step`, `loss`, `val_bpc`, `lr`, `tokens_per_s`, and at start an `env` record that includes `torch` version, thread counts, dtype, and whether a compiled model was used (`compiled`).

You can parse the file with `jq`:

```bash
jq -r 'select(.step!=null) | [.step, .val_bpc, .tokens_per_s] | @tsv' runs/<name>/log.jsonl
```

## Reproducible commands

All commands below run for 300 steps, evaluate every 100, and save at 150. Adjust `--out_dir` per run to avoid overwriting.

### GPT-Mini on TinyShakespeare

Baseline (compile off):
```bash
syntrix.train \
  --config configs/gpt-mini.yaml \
  --data.file data/tinyshakespeare.txt \
  --threads 4 \
  --train_steps 300 --eval_every 100 --save_every 150 \
  --out_dir runs/gpt-mini_base
```

Compile forced on:
```bash
syntrix.train \
  --config configs/gpt-mini.yaml \
  --data.file data/tinyshakespeare.txt \
  --threads 4 \
  --train_steps 300 --eval_every 100 --save_every 150 \
  --compile \
  --out_dir runs/gpt-mini_compile
```

Compile auto with validation (accepts only if >=5% faster):
```bash
syntrix.train \
  --config configs/gpt-mini.yaml \
  --data.file data/tinyshakespeare.txt \
  --threads 4 \
  --train_steps 300 --eval_every 100 --save_every 150 \
  --compile --compile.validate --compile.auto --compile.min_improvement 1.05 \
  --out_dir runs/gpt-mini_compile_auto
```

### RNN-Mini (Text8 mini)

Baseline:
```bash
syntrix.train \
  --config configs/rnn-mini.yaml \
  --download.text8_mini \
  --threads 4 \
  --train_steps 300 --eval_every 100 --save_every 150 \
  --out_dir runs/rnn-mini_base
```

Compile forced on:
```bash
syntrix.train \
  --config configs/rnn-mini.yaml \
  --download.text8_mini \
  --threads 4 \
  --train_steps 300 --eval_every 100 --save_every 150 \
  --compile \
  --out_dir runs/rnn-mini_compile
```

### SSM-Mini (Text8 mini)

Baseline:
```bash
syntrix.train \
  --config configs/ssm-mini.yaml \
  --download.text8_mini \
  --threads 4 \
  --train_steps 300 --eval_every 100 --save_every 150 \
  --out_dir runs/ssm-mini_base
```

Compile forced on:
```bash
syntrix.train \
  --config configs/ssm-mini.yaml \
  --download.text8_mini \
  --threads 4 \
  --train_steps 300 --eval_every 100 --save_every 150 \
  --compile \
  --out_dir runs/ssm-mini_compile
```

### Thread scaling (GPT-Mini, TinyShakespeare)

```bash
for t in 1 2 4; do
  syntrix.train \
    --config configs/gpt-mini.yaml \
    --data.file data/tinyshakespeare.txt \
    --threads "$t" \
    --train_steps 300 --eval_every 100 --save_every 150 \
    --out_dir runs/gpt-mini_t${t}
done
```

## Results tables (example)

The following are example results from the testbed above. Re-run the commands on your machine to regenerate.

### GPT-Mini @ 4 threads (TinyShakespeare)

| Mode        | Compiled | Step | Val BPC | Tokens/sec | Speedup vs. baseline |
|-------------|---------:|-----:|--------:|-----------:|---------------------:|
| Baseline    |     no   |  100 |    ~2.9 |       30–50|                 1.00x |
| Baseline    |     no   |  300 |    ~2.4 |       30–50|                 1.00x |
| Compile ON  |     yes  |  100 |    ~2.9 |       32–55|             ~1.05–1.10x |
| Compile ON  |     yes  |  300 |    ~2.4 |       32–55|             ~1.05–1.10x |

### RNN-Mini @ 4 threads (Text8 mini)

| Mode        | Compiled | Step | Val BPC | Tokens/sec | Speedup vs. baseline |
|-------------|---------:|-----:|--------:|-----------:|---------------------:|
| Baseline    |     no   |  100 |     n/a |       35–60|                 1.00x |
| Compile ON  |     yes  |  100 |     n/a |       35–60|                 ~1.00x |

Note: for smaller RNNs, `torch.compile` often shows negligible speedup on CPU.

### SSM-Mini @ 4 threads (Text8 mini)

| Mode        | Compiled | Step | Val BPC | Tokens/sec | Speedup vs. baseline |
|-------------|---------:|-----:|--------:|-----------:|---------------------:|
| Baseline    |     no   |  100 |     n/a |       25–45|                 1.00x |
| Compile ON  |     yes  |  100 |     n/a |       26–48|             ~1.02–1.07x |

### Thread scaling (GPT-Mini, TinyShakespeare)

| Threads | Compile | Tokens/sec (Step 300) |
|--------:|:--------|----------------------:|
| 1       | off     |                10–18  |
| 2       | off     |                18–30  |
| 4       | off     |                30–50  |

## Tips

- Cold-start compile overhead is amortized; throughput samples are measured after warmup during training, and an explicit forward-only warmup is used for validation in auto mode.
- Ensure power/perf mode is consistent across runs.
- Use unique `--out_dir` per run; logs are appended.
- To export a summary CSV from logs:

```bash
jq -r 'select(.step!=null) | [.step, .val_bpc, .tokens_per_s] | @csv' runs/gpt-mini_base/log.jsonl > runs/gpt-mini_base/metrics.csv
```

## Recreate this page on your machine

Run the commands above, then paste your measured values into the tables. Optionally, add your hardware block at the top for clarity.

