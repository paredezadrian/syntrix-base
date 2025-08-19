# Syntrix-Base Architecture Notes

## GPT Mini (Decoder-Only Transformer)

- Pre-LayerNorm blocks with attention and MLP (SwiGLU)
- Rotary positional embeddings (RoPE) on Q/K

Given token embeddings X ∈ R^{B×T×D}:

1. Pre-norm: X̃ = LN(X)
2. Attention: Q = X̃ W_Q, K = X̃ W_K, V = X̃ W_V
3. RoPE: (Q, K) ← rope(Q, K)
4. Causal scores: A = softmax((Q Kᵀ)/√d_h + M_causal)
5. Context: C = A V
6. Residual: X ← X + C W_O
7. MLP: X ← X + W₂ (silu(W_u X) ⊙ W_g X)

## RNN Mini (Gated RNN)

- Pre-norm + GRU cell over time, followed by SwiGLU MLP with residuals.

## SSM Mini (Selective SSM)

- Pre-norm + depthwise conv shortcut
- Diagonal state update (toy): h_t = α ⊙ h_{t−1} + β ⊙ x_t
- Residual + SwiGLU MLP

## Determinism

- Seeds fixed via `set_seed()`; threads pinned via `set_threads()`
- Log environment (MKL/OMP threads, torch, Python)

## Tokenizers

- Char tokenizer (simple bijection)
- Tiny BPE (greedy merges to target vocab)

## Training

- Microbatching, gradient accumulation, gradient clipping
- Cosine schedule with warmup; optional EMA
