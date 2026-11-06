"""Numerical accuracy against the fp32 ground truth, on a fixed prompt set.

Compares HF and QwenInfer in fp16 and bf16. Reports logit error (mean, max) and
how many prompts produce a greedy continuation identical to fp32. bf16 should be
judged against HF's own bf16: the question is whether we add error beyond what
the format itself costs.

    python -m qweninfer.bench.accuracy
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "weights" / "qwen2.5-0.5b-instruct"
N_NEW = 32
PROMPTS = [
    "The capital of France is Paris, and the capital of Japan is",
    "def fibonacci(n):\n    ",
    "Photosynthesis is the process by which",
    "The three laws of thermodynamics state that",
    "Once upon a time, in a small village,",
    "To compute the derivative of x^3, we",
    "The main difference between TCP and UDP is",
    "In 1969, the Apollo 11 mission",
]


@torch.inference_mode()
def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from qweninfer.engine.generate import generate
    from qweninfer.models.qwen2 import Qwen2

    tok = AutoTokenizer.from_pretrained(W)
    ids = [tok(p, return_tensors="pt").input_ids for p in PROMPTS]

    ref = AutoModelForCausalLM.from_pretrained(W, dtype=torch.float32).eval()
    truth = [ref(i).logits for i in ids]
    truth_tok = [ref.generate(i, max_new_tokens=N_NEW, do_sample=False,
                              repetition_penalty=1.0)[0, i.shape[1]:].tolist() for i in ids]
    del ref; gc.collect()

    results = {}
    for dt in (torch.float16, torch.bfloat16):
        name = str(dt).removeprefix("torch.")
        hf = AutoModelForCausalLM.from_pretrained(W, dtype=dt).cuda().eval()
        results[f"hf-{name}"] = (
            [hf(i.cuda()).logits.float().cpu() for i in ids],
            [hf.generate(i.cuda(), max_new_tokens=N_NEW, do_sample=False,
                         repetition_penalty=1.0)[0, i.shape[1]:].tolist() for i in ids])
        del hf; gc.collect(); torch.cuda.empty_cache()

        me = Qwen2.from_pretrained(W, dtype=dt)
        results[f"qweninfer-{name}"] = (
            [me(i.cuda()).float().cpu() for i in ids],
            [generate(me, i.cuda(), N_NEW) for i in ids])
        del me; gc.collect(); torch.cuda.empty_cache()

    out = {}
    print(f"{len(PROMPTS)} prompts, error vs fp32 logits, greedy {N_NEW}-token match vs fp32\n")
    print(f"{'config':18} {'mean err':>9} {'max err':>8} {'greedy = fp32':>14}")
    for name, (logits, toks) in results.items():
        errs = [(l - t).abs() for l, t in zip(logits, truth)]
        mean = sum(e.mean().item() for e in errs) / len(errs)
        mx = max(e.max().item() for e in errs)
        agree = sum(a == b for a, b in zip(toks, truth_tok))
        out[name] = {"mean_err": mean, "max_err": mx, "greedy_match": agree, "n": len(PROMPTS)}
        print(f"{name:18} {mean:9.4f} {mx:8.3f} {agree:>10}/{len(PROMPTS)}")
    (ROOT / "results" / "accuracy.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
