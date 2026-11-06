"""Decode throughput vs batch size, fp16 and bf16.

Batch-1 decode is memory-bound, so a batch of B reads the same weights once and
should give ~B x the tokens until compute catches up. On this GPU the knee and
the kernels cuBLAS picks differ sharply by dtype.

    python -m qweninfer.bench.batch_scaling
"""

from __future__ import annotations

import gc
import json
import statistics as st
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "weights" / "qwen2.5-0.5b-instruct"
BATCHES = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64]
CTX, NDEC, REPS = 128, 16, 3


@torch.inference_mode()
def step_ms(model, B: int) -> float:
    from qweninfer.engine.kv_cache import KVCache
    cache = KVCache(model.cfg, B, CTX + NDEC + 8, dtype=model.dtype)
    cache.window = CTX
    model(torch.randint(0, 150000, (B, CTX), device="cuda"),
          positions=torch.arange(CTX, device="cuda"), cache=cache, last_only=True)
    nxt = torch.randint(0, 150000, (B, 1), device="cuda")
    times = []
    for _ in range(REPS):
        for i in range(NDEC + 2):
            if i == 2:
                torch.cuda.synchronize(); t = time.perf_counter()
            cache.window = CTX + i % NDEC + 1
            model(nxt, positions=torch.full((B, 1), CTX + i % NDEC, device="cuda"), cache=cache)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1e3 / NDEC)
    del cache; torch.cuda.empty_cache()
    return st.median(times)


def main() -> None:
    from qweninfer.models.qwen2 import Qwen2
    out = {}
    for dt in (torch.float16, torch.bfloat16):
        name = str(dt).removeprefix("torch.")
        m = Qwen2.from_pretrained(W, dtype=dt)
        step_ms(m, 4)                                    # warm up
        out[name] = {}
        for B in BATCHES:
            ms = step_ms(m, B)
            out[name][B] = {"ms_per_step": ms, "tok_per_s": B * 1e3 / ms}
        del m; gc.collect(); torch.cuda.empty_cache()

    print(f"{'batch':>5} | {'fp16 ms':>8} {'tok/s':>7} | {'bf16 ms':>8} {'tok/s':>7}")
    for B in BATCHES:
        a, b = out["float16"][B], out["bfloat16"][B]
        print(f"{B:>5} | {a['ms_per_step']:8.1f} {a['tok_per_s']:7.0f} | "
              f"{b['ms_per_step']:8.1f} {b['tok_per_s']:7.0f}")
    (ROOT / "results" / "batch_scaling.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
