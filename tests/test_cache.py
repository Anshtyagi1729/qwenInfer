"""KV cache correctness.

The cache is an optimisation, so its contract is exact: anything computed
through it must equal what recomputing from scratch gives. Off-by-one errors in
cache positions, the mask or RoPE offsets all break that equality, so these
tests catch the classic cache bugs directly.

CPU, fp32, so equality can be tested tightly.
"""

from __future__ import annotations

import pytest
import torch

from qweninfer.engine.generate import generate
from qweninfer.engine.kv_cache import KVCache
from qweninfer.models.qwen2 import Qwen2

TOL = 1e-3


@pytest.fixture(scope="module")
def model(weights_dir):
    return Qwen2.from_pretrained(weights_dir, dtype=torch.float32, device="cpu")


def _full_last_logits(model, ids):
    with torch.no_grad():
        return model(ids)[:, -1]


def test_incremental_decode_matches_full_recompute(model, prompt_ids):
    """Prefill, then feed tokens one at a time through the cache. At every step
    the logits must equal a from-scratch forward over the whole sequence."""
    P, n = prompt_ids.shape[1], 8
    cache = KVCache(model.cfg, 1, P + n, dtype=torch.float32, device="cpu")
    seq = prompt_ids.clone()

    with torch.no_grad():
        cache.window = P
        logits = model(seq, positions=torch.arange(P), cache=cache)[:, -1]
        for step in range(n):
            ref = _full_last_logits(model, seq)
            d = (logits - ref).abs().max().item()
            assert d < TOL, f"step {step}: cached vs full recompute differ by {d:.3e}"
            nxt = logits.argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
            pos = P + step
            cache.window = pos + 1
            logits = model(nxt, positions=torch.tensor([[pos]]), cache=cache)[:, -1]


def test_generate_matches_hf_greedy(model, hf_model, prompt_ids):
    """End to end: our cached greedy generation vs HF generate (no penalty)."""
    n = 16
    ours = generate(model, prompt_ids, n)
    with torch.no_grad():
        ref = hf_model.generate(prompt_ids, max_new_tokens=n, do_sample=False,
                                repetition_penalty=1.0)[0, prompt_ids.shape[1]:].tolist()
    assert ours == ref, f"\n  ours: {ours}\n  HF:   {ref}"


def test_ragged_batch_matches_individual(model, tokenizer):
    """Two sequences of different lengths decode in ONE batch; each must get the
    logits it would get alone. This is what the position-based mask is for: the
    shorter sequence's stale cache slots must be invisible to it."""
    a = tokenizer("Water boils at", return_tensors="pt").input_ids
    b = tokenizer("The largest planet in the solar system is", return_tensors="pt").input_ids
    La, Lb = a.shape[1], b.shape[1]
    assert La != Lb, "test needs different lengths"

    cache = KVCache(model.cfg, 2, 64, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        # Prefill each request into its own row, one at a time.
        firsts = []
        for row, ids in enumerate((a, b)):
            cache.row_start, cache.window = row, ids.shape[1]
            out = model(ids, positions=torch.arange(ids.shape[1]), cache=cache)
            firsts.append(out[:, -1].argmax(-1))
        # Batched decode: both rows at once, each at its own position.
        cache.row_start = 0
        cache.window = max(La, Lb) + 1
        nxt = torch.stack(firsts)                                     # (2, 1)
        batched = model(nxt, positions=torch.tensor([[La], [Lb]]), cache=cache)[:, -1]

    for i, ids in enumerate((a, b)):
        alone = _full_last_logits(model, torch.cat([ids, nxt[i : i + 1]], dim=1))
        d = (batched[i] - alone[0]).abs().max().item()
        assert d < TOL, f"sequence {i}: batched vs alone differ by {d:.3e}"
