"""Batch-padding ablation at 16 request slots.

At 16 active rows bf16 decode takes ~70 ms per step on this GPU, while 20 rows
take ~47 ms (cuBLAS kernel choice). With padding the engine runs 20 rows: 16 real
requests plus 4 dummy rows in extra cache rows. Admission stays capped at 16, so
the only difference between the two configurations is padding. Two interleaved
passes rule out thermal drift.

    python -m qweninfer.bench.padding
"""

from __future__ import annotations

import json

import torch

from qweninfer.bench.serving import ROOT, W, run_mode


def main() -> None:
    from qweninfer.engine.engine import autotune_padding
    from qweninfer.models.qwen2 import Qwen2

    model = Qwen2.from_pretrained(W, dtype=torch.bfloat16)
    pad_to, ms = autotune_padding(model, max_slots=16, max_rows=24)
    print("step ms:", {c: round(v) for c, v in ms.items()})
    print("pad_to n=1..16:", pad_to[1:], "\n", flush=True)
    results = []
    for _ in range(2):
        results.append(run_mode(model, "continuous, 16 slots", max_slots=16))
        results.append(run_mode(model, "continuous + padding, 16 slots", max_slots=16,
                                pad_to=pad_to))
    (ROOT / "results" / "padding.json").write_text(json.dumps(
        {"autotune_ms": ms, "pad_to": pad_to, "results": results}, indent=1))


if __name__ == "__main__":
    main()
