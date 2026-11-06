"""Serving workload: sequential vs static vs continuous batching.

96 requests arrive together. Prompt lengths are uniform in [32, 256]; output
lengths are long-tailed (exponential, mean ~100, clipped to [8, 256]) because
that is what real traffic looks like and what static batching handles worst.
Requests generate exactly their assigned length (EOS ignored), as is standard in
serving benchmarks, so every mode does identical work.

    python -m qweninfer.bench.serving
"""

from __future__ import annotations

import gc
import json
import statistics as st
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "weights" / "qwen2.5-0.5b-instruct"
N_REQ, MAX_LEN, SEED = 96, 512, 0


def workload():
    from qweninfer.engine.engine import Request
    rng = np.random.default_rng(SEED)
    plens = rng.integers(32, 257, N_REQ)
    olens = np.clip(rng.exponential(100, N_REQ), 8, 256).astype(int)
    return [Request(i, rng.integers(1000, 100000, int(p)).tolist(), int(o))
            for i, (p, o) in enumerate(zip(plens, olens))]


def run_mode(model, name, **engine_kw):
    from qweninfer.engine.engine import Engine
    reqs = workload()
    engine = Engine(model, max_len=MAX_LEN, **engine_kw)
    t0 = time.perf_counter()
    for r in reqs:
        r.arrival = t0
    done = engine.run(reqs)
    wall = time.perf_counter() - t0
    del engine; torch.cuda.empty_cache()
    out_tok = sum(len(r.output) for r in done)
    ttft = sorted(r.first_token - r.arrival for r in done)
    lat = [r.finished - r.arrival for r in done]
    res = {"mode": name, "wall_s": wall, "output_tokens": out_tok,
           "tok_per_s": out_tok / wall, "ttft_mean_s": st.mean(ttft),
           "ttft_p90_s": ttft[int(0.9 * len(ttft))], "latency_mean_s": st.mean(lat)}
    print(f"{name:34} {wall:7.1f}s {res['tok_per_s']:7.1f} tok/s  "
          f"TTFT mean {res['ttft_mean_s']:6.1f}s p90 {res['ttft_p90_s']:6.1f}s  "
          f"latency mean {res['latency_mean_s']:6.1f}s", flush=True)
    return res


def main() -> None:
    from qweninfer.engine.engine import autotune_padding
    from qweninfer.models.qwen2 import Qwen2

    reqs = workload()
    print(f"{N_REQ} requests, {sum(len(r.prompt) for r in reqs)} prompt tokens, "
          f"{sum(r.max_new_tokens for r in reqs)} output tokens "
          f"(output length min {min(r.max_new_tokens for r in reqs)}, "
          f"median {int(np.median([r.max_new_tokens for r in reqs]))}, "
          f"max {max(r.max_new_tokens for r in reqs)})\n", flush=True)

    model = Qwen2.from_pretrained(W, dtype=torch.bfloat16)
    pad32, ms32 = autotune_padding(model, 32)
    pad64, ms64 = autotune_padding(model, 64)
    print("autotuned step times (ms): " + ", ".join(f"{c}:{v:.0f}" for c, v in ms64.items()))
    print(f"pad_to for n=1..32: {pad32[1:]}\n", flush=True)

    results = [
        run_mode(model, "sequential (1 slot)", max_slots=1),
        run_mode(model, "static batching (32 slots)", max_slots=32, static=True),
        run_mode(model, "continuous (32 slots)", max_slots=32),
        run_mode(model, "continuous + padding (32 slots)", max_slots=32, pad_to=pad32),
        run_mode(model, "continuous + padding (64 slots)", max_slots=64, pad_to=pad64),
    ]
    del model; gc.collect(); torch.cuda.empty_cache()
    (ROOT / "results" / "serving.json").write_text(json.dumps(
        {"workload": {"n_requests": N_REQ, "seed": SEED}, "autotune_ms": ms64,
         "results": results}, indent=1))


if __name__ == "__main__":
    main()
