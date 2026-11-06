#!/usr/bin/env python
"""Chat with Qwen2.5-0.5B running on QwenInfer. Streams tokens and reports speed.

    .venv/bin/python scripts/chat.py
    .venv/bin/python scripts/chat.py --prompt "Explain a KV cache in two sentences."
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer   # tokenizer only; the model is ours

from qweninfer.engine.generate import generate
from qweninfer.models.qwen2 import Qwen2

W = Path(__file__).resolve().parent.parent / "weights" / "qwen2.5-0.5b-instruct"
EOS = {151645, 151643}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None, help="one-shot; omit for interactive")
    ap.add_argument("--max-new", type=int, default=200)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(W)
    t0 = time.perf_counter()
    model = Qwen2.from_pretrained(W, dtype=torch.float16, device="cuda")
    print(f"QwenInfer: loaded in {time.perf_counter() - t0:.1f}s\n")

    def run(text: str) -> None:
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True)["input_ids"].cuda()
        produced: list[int] = []
        printed = 0
        first: list[float] = []

        def stream(t: int) -> None:
            nonlocal printed
            if not first:
                first.append(time.perf_counter())
            produced.append(t)
            # Decode the whole tail each time: single tokens can split multi-byte
            # characters, so print only what is newly stable.
            text_so_far = tok.decode(produced)
            print(text_so_far[printed:], end="", flush=True)
            printed = len(text_so_far)

        start = time.perf_counter()
        generate(model, ids, args.max_new, eos_ids=EOS, on_token=stream)
        end = time.perf_counter()
        n = len(produced)
        ttft = (first[0] - start) if first else float("nan")
        rate = (n - 1) / (end - first[0]) if n > 1 else float("nan")
        print(f"\n\n  [{ids.shape[1]} prompt tokens | TTFT {ttft * 1e3:.0f} ms | "
              f"{n} tokens at {rate:.1f} tok/s]\n")

    if args.prompt:
        run(args.prompt)
        return
    print("interactive: empty line or ctrl-c to quit\n")
    while True:
        try:
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not q:
            break
        run(q)


if __name__ == "__main__":
    main()
