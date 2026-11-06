"""Single-request generation: prefill once, then decode one token at a time.

The two phases are separate because they are different workloads:
prefill pushes the whole prompt through in one compute-bound pass; decode pushes
one token per pass and is bound by reading the weights from memory.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from qweninfer.engine.kv_cache import KVCache
from qweninfer.models.qwen2 import Qwen2


@torch.inference_mode()
def generate(
    model: Qwen2, prompt_ids: torch.Tensor, max_new_tokens: int,
    eos_ids: set[int] | None = None, cache: KVCache | None = None,
    on_token: Callable[[int], None] | None = None,
) -> list[int]:
    """Greedy decoding with a KV cache. prompt_ids: (1, P) on the model's device."""
    P = prompt_ids.shape[1]
    device = prompt_ids.device
    if cache is None:
        cache = KVCache(model.cfg, 1, P + max_new_tokens,
                        dtype=model.dtype, device=device)

    # Prefill: every prompt token at positions 0..P-1, one pass. Only the last
    # position's logits are needed to choose the first new token.
    cache.window = P
    logits = model(prompt_ids, positions=torch.arange(P, device=device),
                   cache=cache, last_only=True)
    nxt = logits[:, -1].argmax(-1, keepdim=True)                     # (1, 1)

    out: list[int] = []
    for step in range(max_new_tokens):
        tok = nxt.item()          # one sync per token: we need it to stop on EOS
        if eos_ids and tok in eos_ids:
            break
        out.append(tok)
        if on_token is not None:
            on_token(tok)
        pos = P + step            # absolute position of the token being fed in
        cache.window = pos + 1
        logits = model(nxt, positions=torch.tensor([[pos]], device=device),
                       cache=cache)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
    return out
