"""Prefill latency (time to first token): ours repacked vs ours unrepacked vs HF.

Checks that repacking weights for decode did not regress prefill, which is a
large matmul rather than a GEMV and could prefer the other layout. Models are
loaded one at a time (only one fits in 1681 MiB); two passes expose drift.

    python -m qweninfer.bench.prefill_compare
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
LENS, REPS = [64, 256, 512, 1024], 5


def load(name):
    if name == "hf":
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(W, dtype=torch.float16).cuda().eval()
        return lambda ids: m(ids, use_cache=False, logits_to_keep=1).logits, m
    from qweninfer.models.qwen2 import Qwen2
    m = Qwen2.from_pretrained(W, dtype=torch.float16, device="cuda", fuse=(name == "ours"))
    return lambda ids: m(ids, last_only=True), m


@torch.inference_mode()
def main() -> None:
    from transformers import AutoTokenizer
    ids = AutoTokenizer.from_pretrained(W)("The history of computing is a long story. " * 400,
                                           return_tensors="pt").input_ids.cuda()
    impls = ["ours", "ours-unfused", "hf"]
    res = {i: {n: [] for n in LENS} for i in impls}
    for _ in range(2):
        for impl in impls:
            fn, m = load(impl)
            for _ in range(3):
                fn(ids[:, :64])
            for _ in range(REPS):
                for n in LENS:
                    torch.cuda.synchronize(); t = time.perf_counter()
                    fn(ids[:, :n])
                    torch.cuda.synchronize()
                    res[impl][n].append((time.perf_counter() - t) * 1e3)
            del fn, m; gc.collect(); torch.cuda.empty_cache()

    print(f"{'prompt':>7} | " + " | ".join(f"{i:>13}" for i in impls))
    print("-" * 56)
    out = {}
    for n in LENS:
        med = {i: st.median(res[i][n]) for i in impls}
        out[n] = med
        print(f"{n:>7} | " + " | ".join(f"{med[i]:10.1f} ms" for i in impls))
    (ROOT / "results" / "prefill_compare.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
