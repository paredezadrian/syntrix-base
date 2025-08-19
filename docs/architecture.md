# Syntrix-Base Architecture Notes

This document summarizes the core model architectures and implementation notes with equations and references. Models are designed for CPU-first execution with clarity and determinism.

## GPT Mini (Decoder-Only Transformer)

- Pre-LayerNorm residual blocks with attention and MLP (SwiGLU)
- Rotary positional embeddings (RoPE) applied to Q/K

Let X ∈ ℝ^{B×T×D} be token embeddings. For each block:

1) Pre-norm: X̃ = LN(X)

2) Multi-Head Self-Attention:
   - Head dimension: d_h = D / H
   - Projections: Q = X̃ W_Q, K = X̃ W_K, V = X̃ W_V, with W_Q, W_K, W_V ∈ ℝ^{D×D}
   - RoPE on each head: (Q, K) ← rope(Q, K), angles θ_i = base^{−2i/D}
   - Causal attention: A = softmax((Q Kᵀ) / √d_h + M_causal)
   - Context: C = A V, output O = C W_O, W_O ∈ ℝ^{D×D}
   - Residual: X ← X + O

3) SwiGLU MLP:
   - Hidden H = m · D, with expansion ratio m (e.g., 4)
   - Gate: G = σ(X W_g), Up: U = X W_u
   - Out: Y = (U ⊙ G) W_2, Residual: X ← X + Y

Loss uses cross-entropy over vocabulary; logits ∈ ℝ^{B×T×V}.

References:
- RoPE: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- SwiGLU: Shazeer, "GLU Variants Improve Transformer" (2020)
- Pre-LN Transformers: Xiong et al., (2020)

## RNN Mini (Gated RNN)

- Pre-norm + gated recurrent cell (GRU-like) across time
- Followed by SwiGLU MLP + residual

Recurrence (schematic):

h_t = GRU(h_{t−1}, x_t); y_t = h_t; X ← X + MLP(LN(X))

## SSM Mini (Selective SSM)

- Pre-norm + depthwise convolutional shortcut
- Diagonal (per-channel) state update:

h_t = α ⊙ h_{t−1} + β ⊙ x_t; y_t = h_t

This is a toy SSM variant optimized for CPU simplicity; then Residual + SwiGLU MLP.

References:
- Gu et al., "State Space Models for Sequence Modeling" (S4)
- Smith et al., "Simplified SSMs" (various minimalist variants)

## Determinism & Reproducibility

- Seeds fixed via `set_seed()`; threads pinned via `set_threads()` (OMP/MKL/Torch)
- Environment logged at run start: Python, PyTorch, threads, dtype, git commit, compiled flag
- Dtype-aware tolerances used in tests (float32/float64)

## Tokenizers

- Character tokenizer: bijection between chars and integer ids
- Tiny BPE: greedy merges to target vocabulary size (e.g., 256)

## Training

- Microbatching and gradient accumulation to simulate large batch sizes on CPUs
- Gradient clipping; cosine LR with warmup; optional EMA
- Optional `torch.compile` with CLI `--compile` and validation/auto-selection

---

## Troubleshooting & FAQ

### Common Errors

- "sequence length exceeds block_size":
  - Ensure your `--block_size` matches or exceeds the longest context window used during training/eval.

- Non-deterministic results across runs:
  - Confirm seeds and thread counts are set (see `--seed`, `--threads`).
  - Check that `OMP_NUM_THREADS` and `MKL_NUM_THREADS` are consistent.

- Slow throughput on CPU:
  - Reduce `--block_size`, increase `--grad_accum` with small `--microbatch`.
  - Try `--compile --compile.validate --compile.auto` to auto-enable `torch.compile` if faster.
  - Ensure power/performance mode is set consistently on your system.

- Memory issues with large datasets:
  - Use `--data.use_mmap` to enable memory-mapped random block sampling.

### Performance Tips

- Thread Tuning: Start with `--threads 4` on 4c/8t CPUs; adjust based on observed tokens/sec.
- Microbatching: Keep `--microbatch` small (1–4) and increase `--grad_accum`.
- Evaluations: Increase `--eval_every` to reduce overhead during benchmarking.

### Dataset Notes

- TinyShakespeare: One-file plain text; good for quick validation and demos.
- Text8 mini: Use `--download.text8_mini` to fetch a small subset; reproducible across runs.

### Logging & Reproducibility

- JSONL logs at `<out_dir>/log.jsonl` include `step`, `val_bpc`, `tokens_per_s`, `elapsed_s`.
- The initial `env` record captures the toolchain and thread settings; include it when reporting results.
