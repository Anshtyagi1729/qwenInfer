#!/usr/bin/env python
"""Decode-shaped matmul (ONE row x weight) in several mathematically identical
formulations. Shows that weight layout alone changes achieved bandwidth ~2.6x on
the MX450: nn.Linear's (out, in) layout lands on a padded 16x16 tensor-op kernel,
the transposed (in, out) layout on a coalesced GEMV near peak bandwidth."""
import time

import torch
import torch.nn.functional as F


def bench(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters


W = torch.randn(4864, 896, device="cuda", dtype=torch.float16) * 0.02  # MLP up_proj
Wt = W.t().contiguous()
x = torch.randn(1, 1, 896, device="cuda", dtype=torch.float16)
xv = x.view(-1)
nbytes = W.numel() * W.element_size()
for name, fn in {
    "F.linear(x, W)   nn.Linear layout": lambda: F.linear(x, W),
    "torch.mv(W, x)                   ": lambda: torch.mv(W, xv),
    "x @ Wt           transposed      ": lambda: x @ Wt,
}.items():
    t = bench(fn)
    print(f"{name}  {t*1e6:7.1f} us  {nbytes/t/1e9:5.1f} GB/s")
