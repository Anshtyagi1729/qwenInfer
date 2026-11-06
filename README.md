# QwenInfer

**A from-scratch LLM inference engine for memory-constrained GPUs.**

QwenInfer runs Qwen2.5-0.5B on a 2 GB laptop GPU (NVIDIA MX450) with a
forward pass, KV cache and decode loop written from scratch in PyTorch. It is
verified against HuggingFace `transformers` by exact logit parity, and decodes
**1.7–1.8× faster than HuggingFace** on the same hardware, reaching **84% of the
GPU's memory-bandwidth limit**.

```
                    decode latency, ms/token (lower is better)
context   QwenInfer   HF DynamicCache   HF StaticCache   speedup vs best HF
   16       24.50          42.05             56.67            1.72x
  256       26.34          44.95             56.64            1.71x
 1024       31.12          57.04             56.68            1.82x
```

## Highlights

- **Complete Qwen2 implementation**: RMSNorm, rotary position embeddings,
  grouped-query attention, SwiGLU, tied embeddings, KV cache, with no model code
  from `transformers`.
- **Verified, not eyeballed.** Per-layer hidden states, final logits and greedy
  decoding all match HuggingFace. In fp16 QwenInfer deviates from the fp32 ground
  truth *less* than HuggingFace's own fp16 path (max 0.13 vs 0.31).
- **Profiling-driven 1.7× decode speedup** from a single change in weight
  layout; see [the main finding](#the-main-finding-a-kernel-built-for-hardware-this-gpu-does-not-have).
- **A KV cache that stays flat as context grows**, where HuggingFace's two caches
  each fail in a different way.

## The main finding: a kernel built for hardware this GPU does not have

Batch-1 decode is memory-bound: every token requires reading all 942 MiB of
weights. At the measured 48.2 GB/s that sets a floor of **20.5 ms/token**. The
first working version took 42.3 ms, so roughly half the time was going
somewhere other than physics.

Profiling one decode step showed **32.5 of 40.8 ms of GPU time** in a single
kernel family, `cutlass_70_wmma_tensorop_f16_s161616`: a tensor-core kernel that
computes in 16x16x16 tiles. During decode each projection multiplies **one row**
(one token), so 15/16 of every tile is padding. And the MX450's TU117 die ships
without tensor cores.

A one-row matmul is a GEMV. Benchmarking mathematically identical formulations of
one projection ([`scripts/gemv_layout.py`](scripts/gemv_layout.py)):

```
F.linear(x, W)     weight stored (out, in), as nn.Linear does    16.6 GB/s
x @ W_t            weight stored (in, out)                       43.9 GB/s  (91% of peak)
```

Same math, same bytes, 2.6× the bandwidth: the transposed layout gets coalesced
reads and is dispatched to a GEMV kernel. QwenInfer therefore **repacks weights
once at load time**: projections transposed, Q/K/V fused into one matrix,
gate/up fused into one. That cuts 72 kernel launches per token, changes no
outputs (all parity tests run through the repacked path), and took decode from
42.3 to 24.5 ms/token. Prefill also got slightly faster rather than regressing.

## The KV cache

HuggingFace's two caches fail in opposite directions:

| cache | behaviour | measured cost |
|---|---|---|
| `DynamicCache` | grows with `torch.cat` each step | +15 ms/step from 16 to 1024 tokens: allocator churn, versus 0.26 ms of actual KV bandwidth |
| `StaticCache` | preallocated to max length | flat ~56.7 ms: always attends over the whole reserved window |
| **QwenInfer** | preallocated once, attention reads a view sliced to the live length | neither cost |

The attention mask is derived from **absolute positions** (`key j is visible to
the query at position p iff j <= p`) rather than tensor shapes. That one rule
covers causal prefill, decode with no special case, and batches of sequences
with different lengths. The last is tested directly: a ragged batch must produce
exactly the logits each sequence gets alone.

## Hardware characterisation

Measured on the test machine (NVIDIA MX450, Turing sm_75, 14 SMs):

```
addressable VRAM     1681 MiB   (board is "2 GB"; ~367 MiB reserved)
memory bandwidth     48.2 GB/s  (56 GB/s theoretical, 64-bit GDDR5)
fp32 matmul          1.39 TFLOP/s
fp16 matmul          0.32 TFLOP/s   <- 4x slower than fp32
```

fp16 matmul being slower than fp32 is not an ALU limit: an elementwise fp16 FMA
runs at exactly 2x the fp32 rate. It is cuBLAS kernel selection on a die without
tensor cores, the same root cause as the main finding above. fp32 weights
(1886 MiB) do not fit, so fp16 storage is required regardless.

## Correctness

HuggingFace `transformers` is the oracle; no reference math is hand-written in
the tests.

| test | checks |
|---|---|
| `test_rmsnorm`, `test_rope`, `test_mlp` | each component against HF's own module |
| `test_hidden_states_per_layer` | all 24 layers; reports the first layer that diverges |
| `test_logit_parity` | final logits within 2e-3 (fp32) |
| `test_greedy_sequence_identical` | identical greedy tokens |
| `test_incremental_decode_matches_full_recompute` | cached decode == recomputing from scratch, every step |
| `test_generate_matches_hf_greedy` | end-to-end cached generation == HF |
| `test_ragged_batch_matches_individual` | different-length sequences batched == each alone |

Two early failures turned out to be HuggingFace behaviour rather than model bugs:
HF stores the final-normed stream in its last `hidden_states` entry, and
`generate()` applies the checkpoint's `repetition_penalty: 1.1` even in greedy
mode. Each was verified independently before the harness was changed.

## Benchmark methodology

Laptop GPUs change clock speed under load, and naive timing on them is
unreliable: an un-warmed run measured 49 ms/token for a model that runs at
41.8 ms warm. Every benchmark here therefore:

- warms the GPU before timing and synchronises before reading the clock;
- interleaves conditions across rounds, so clock drift cannot pose as a trend;
- reports medians over repeated rounds (spread is under 1%);
- runs each implementation in its own process, twice (A B C A B C), so
  cross-process drift would show up as disagreement between passes.

## Usage

```bash
uv sync --extra dev
# weights: Qwen/Qwen2.5-0.5B-Instruct -> weights/qwen2.5-0.5b-instruct/

.venv/bin/python scripts/chat.py                              # chat, streamed
.venv/bin/python -m pytest -q                                 # correctness suite
.venv/bin/python -m qweninfer.bench.decode_compare --driver   # decode vs HF
.venv/bin/python -m qweninfer.bench.prefill_compare           # prefill vs HF
.venv/bin/python scripts/roofline.py                          # hardware constants
```

```python
from qweninfer.models.qwen2 import Qwen2
from qweninfer.engine.generate import generate

model = Qwen2.from_pretrained("weights/qwen2.5-0.5b-instruct")   # fp16, CUDA
tokens = generate(model, prompt_ids, max_new_tokens=100)
```

## Layout

```
qweninfer/
  models/config.py      model configuration
  models/qwen2.py       the forward pass, weight loading and repacking
  engine/kv_cache.py    preallocated, length-sliced KV cache
  engine/generate.py    prefill + decode loop
  bench/                decode and prefill benchmarks against HuggingFace
tests/                  parity and cache-correctness suites
scripts/                chat demo, hardware characterisation, layout microbenchmark
results/                raw benchmark output (JSON)
```

## Limitations and next steps

- **Continuous batching**: in progress. The cache and the position-based mask
  already support ragged batches.
- **Attention cost still grows with context** (24.5 → 31.1 ms from 16 to 1024
  tokens). The attention matmuls hit the same poor cuBLAS kernel selection, and
  PyTorch's `scaled_dot_product_attention` was measured 44% *slower* on this GPU.
  A custom Triton decode-attention kernel is the planned fix.
- **Prefill runs at the fp16 matmul ceiling** (~0.37 TFLOP/s). Since fp32 matmul
  is ~4x faster on this GPU, computing prefill in fp32 from fp16 storage is a
  promising hardware-specific optimisation.
