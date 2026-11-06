"""Decode latency vs context length: ours vs HF DynamicCache vs HF StaticCache.

Methodology: warm up first, interleave context lengths
across rounds so clock drift cannot look like a trend, report min/median/max.
Only one model fits in 1681 MiB, so each implementation runs in its own process;
`--driver` runs them all twice (A B C A B C) so cross-process drift is visible.

    python -m qweninfer.bench.decode_compare --driver
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "weights" / "qwen2.5-0.5b-instruct"
CTX = [16, 256, 1024]
ROUNDS, NDEC, WARM = 5, 24, 3


def sync_now() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def make_ours():
    from qweninfer.engine.kv_cache import KVCache
    from qweninfer.models.qwen2 import Qwen2

    model = Qwen2.from_pretrained(W, dtype=torch.float16, device="cuda")
    cache = KVCache(model.cfg, 1, max(CTX) + NDEC + WARM + 8)

    def run(ids, S):
        cache.window = S
        logits = model(ids[:, :S], positions=torch.arange(S, device="cuda"),
                       cache=cache, last_only=True)
        nxt = logits[:, -1].argmax(-1, keepdim=True)

        def step(i):
            nonlocal nxt
            cache.window = S + i + 1
            logits = model(nxt, positions=torch.tensor([[S + i]], device="cuda"),
                           cache=cache)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
        return step
    return run


def make_hf(static: bool):
    from transformers import AutoModelForCausalLM, StaticCache

    model = AutoModelForCausalLM.from_pretrained(W, dtype=torch.float16).cuda().eval()
    maxlen = max(CTX) + NDEC + WARM + 8

    def run(ids, S):
        if static:
            past = StaticCache(config=model.config, max_cache_len=maxlen)
            out = model(ids[:, :S], past_key_values=past,
                        cache_position=torch.arange(S, device="cuda"), use_cache=True)
        else:
            out = model(ids[:, :S], use_cache=True)
            past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        state = {"past": past, "nxt": nxt}

        def step(i):
            kw = {"cache_position": torch.tensor([S + i], device="cuda")} if static else {}
            o = model(state["nxt"], past_key_values=state["past"], use_cache=True, **kw)
            if not static:
                state["past"] = o.past_key_values
            state["nxt"] = o.logits[:, -1:].argmax(-1)
        return step
    return run


@torch.inference_mode()
def measure(impl: str) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(W)
    ids = tok("The history of computing is a long story. " * 400,
              return_tensors="pt").input_ids.cuda()
    run = {"ours": make_ours, "hf-dynamic": lambda: make_hf(False),
           "hf-static": lambda: make_hf(True)}[impl]()

    for _ in range(3):                                  # ramp the clocks
        step = run(ids, 64)
        for i in range(4):
            step(i)

    samples = {S: [] for S in CTX}
    for _ in range(ROUNDS):
        for S in CTX:                                   # interleaved
            step = run(ids, S)
            for i in range(WARM):
                step(i)
            t = sync_now()
            for i in range(WARM, WARM + NDEC):
                step(i)
            samples[S].append((sync_now() - t) * 1e3 / NDEC)
    return {"impl": impl, "ms_per_tok": {
        S: {"min": min(v), "median": st.median(v), "max": max(v)} for S, v in samples.items()}}


def driver() -> None:
    impls = ["ours", "hf-dynamic", "hf-static"]
    runs = []
    for p in (1, 2):
        for impl in impls:
            r = subprocess.run([sys.executable, "-m", "qweninfer.bench.decode_compare",
                                "--impl", impl], capture_output=True, text=True, cwd=ROOT)
            line = [l for l in r.stdout.splitlines() if l.startswith("{")]
            if not line:
                print(r.stderr[-2000:]); raise SystemExit(f"{impl} failed")
            res = json.loads(line[-1]); res["pass"] = p
            runs.append(res)
            print(f"pass {p} {impl:>10}: " + "  ".join(
                f"ctx {S}={v['median']:.2f}ms" for S, v in res["ms_per_tok"].items()), flush=True)

    out = ROOT / "results" / "decode_compare.json"
    out.write_text(json.dumps(runs, indent=1))

    print(f"\n{'ctx':>6} | " + " | ".join(f"{i:>12}" for i in impls) + " | ours vs best HF")
    print("-" * 70)
    for S in map(str, CTX):
        med = {i: st.median(r["ms_per_tok"][S]["median"] for r in runs if r["impl"] == i)
               for i in impls}
        best_hf = min(med["hf-dynamic"], med["hf-static"])
        print(f"{S:>6} | " + " | ".join(f"{med[i]:9.2f} ms" for i in impls)
              + f" | {(best_hf / med['ours'] - 1) * 100:+.0f}%")
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["ours", "hf-dynamic", "hf-static"])
    ap.add_argument("--driver", action="store_true")
    a = ap.parse_args()
    if a.driver:
        driver()
    else:
        res = measure(a.impl)
        print(json.dumps({"impl": res["impl"],
                          "ms_per_tok": {str(k): v for k, v in res["ms_per_tok"].items()}}))
