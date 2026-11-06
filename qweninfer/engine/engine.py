"""Continuous-batching inference engine.

Static batching runs a group of requests until the longest one finishes, so
slots whose requests are done sit idle and utilisation falls to roughly
mean_length / max_length. Continuous batching (iteration-level scheduling)
re-forms the batch every step: a finished request's slot goes to the next
waiting request immediately, so the batch stays full.

Each step:
  1. admit   prefill waiting requests into free cache rows
  2. decode  one forward pass for every active request at once; each sits at
             its own position, which the position-based mask handles
  3. retire  remove finished requests, keeping live ones packed in rows 0..n-1

Packing matters: decode reads the cache as rows[0:n], a free view. Holes would
force a gather that copies the whole live cache every step. Moving the last row
into a hole costs one sequence's copy, once per finished request.

All bookkeeping (lengths, positions, slot assignment) lives on the CPU. The one
device-to-host sync per step is reading back the new tokens, which is needed to
detect stopping.

Batch padding: on this GPU, decode step time is not monotonic in batch size
(bf16: 16 rows take 70.7 ms, 24 rows take 48.2 ms) because cuBLAS picks kernels
per shape. `autotune_padding` measures a set of candidate sizes once and maps
every active count n to the fastest size >= n. Dummy rows sit in free cache rows
at position 0, so they never affect real requests.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import torch

from qweninfer.engine.kv_cache import KVCache
from qweninfer.models.qwen2 import Qwen2


@dataclass
class Request:
    rid: int
    prompt: list[int]
    max_new_tokens: int
    output: list[int] = field(default_factory=list)
    # timestamps (perf_counter seconds)
    arrival: float = 0.0
    first_token: float | None = None
    finished: float | None = None
    # engine state
    length: int = 0            # tokens currently in this request's cache row
    last: int = 0              # next token to feed

    @property
    def done(self) -> bool:
        return self.finished is not None


class Engine:
    def __init__(
        self, model: Qwen2, max_slots: int, max_len: int,
        eos_ids: set[int] | None = None, static: bool = False,
        max_prefills_per_step: int | None = None,
        pad_to: list[int] | None = None,
    ) -> None:
        """
        max_slots              concurrent requests (rows of the KV cache)
        max_len                capacity per request: prompt + generated tokens
        static                 admit only when the batch is empty, i.e. classic
                               static batching, kept for comparison
        max_prefills_per_step  cap on admissions per step; prefill stalls decode
                               for everyone, so this trades time-to-first-token
                               against inter-token latency (None = no cap)
        pad_to                 pad_to[n] = batch size to run when n requests are
                               active (see autotune_padding); None = no padding.
                               Sizes may exceed max_slots: the extra cache rows
                               hold dummy rows only, never requests
        """
        if pad_to is not None:
            if len(pad_to) <= max_slots or any(pad_to[n] < n for n in range(max_slots + 1)):
                raise ValueError("pad_to needs an entry n -> size >= n for n = 0..max_slots")
        # Request slots (how many requests may run) and cache rows (physical
        # memory) are separate: padding rows are extra rows, never extra slots,
        # so padding cannot quietly raise concurrency.
        rows = max([max_slots] + (pad_to[: max_slots + 1] if pad_to else []))
        self.model = model
        self.device = model.norm.weight.device
        self.cache = KVCache(model.cfg, rows, max_len, dtype=model.dtype,
                             device=self.device)
        self.max_slots, self.max_len = max_slots, max_len
        self.eos_ids = eos_ids or set()
        self.static = static
        self.max_prefills = max_prefills_per_step
        self.pad_to = pad_to
        self.waiting: deque[Request] = deque()
        self.active: list[Request] = []       # active[i] owns cache row i

    def submit(self, req: Request) -> None:
        if len(req.prompt) + req.max_new_tokens > self.max_len:
            raise ValueError(f"request {req.rid} needs {len(req.prompt) + req.max_new_tokens}"
                             f" tokens, capacity is {self.max_len}")
        req.arrival = req.arrival or time.perf_counter()
        self.waiting.append(req)

    # ---------------------------------------------------------------- phases

    def _finish_check(self, req: Request, tok: int) -> bool:
        if tok in self.eos_ids:
            return True
        req.output.append(tok)
        return len(req.output) >= req.max_new_tokens

    @torch.inference_mode()
    def _admit(self) -> list[Request]:
        done = []
        if self.static and self.active:
            return done
        admitted = 0
        while (self.waiting and len(self.active) < self.max_slots
               and (self.max_prefills is None or admitted < self.max_prefills)):
            req = self.waiting.popleft()
            row, P = len(self.active), len(req.prompt)
            self.cache.row_start, self.cache.window = row, P
            ids = torch.tensor([req.prompt], device=self.device)
            logits = self.model(ids, positions=torch.arange(P, device=self.device),
                                cache=self.cache, last_only=True)
            tok = int(logits[0, -1].argmax())
            req.first_token = time.perf_counter()
            admitted += 1
            if self._finish_check(req, tok):
                req.finished = req.first_token
                done.append(req)                # never needed a decode slot
                continue
            req.length, req.last = P, tok
            self.active.append(req)
        return done

    @torch.inference_mode()
    def _decode(self) -> list[int]:
        self.cache.row_start = 0
        self.cache.window = max(r.length for r in self.active) + 1
        toks = [[r.last] for r in self.active]
        pos = [[r.length] for r in self.active]
        if self.pad_to is not None:
            # Dummy rows: free cache rows at position 0. Their writes land in
            # slots the next prefill overwrites, and the mask hides them.
            extra = self.pad_to[len(self.active)] - len(self.active)
            toks += [[0]] * extra
            pos += [[0]] * extra
        logits = self.model(torch.tensor(toks, device=self.device),
                            positions=torch.tensor(pos, device=self.device),
                            cache=self.cache)
        nxt = logits[: len(self.active), -1].argmax(-1).tolist()   # one sync per step
        finished_rows = []
        now = time.perf_counter()
        for i, (req, tok) in enumerate(zip(self.active, nxt)):
            req.length += 1
            req.last = tok
            if self._finish_check(req, tok):
                req.finished = now
                finished_rows.append(i)
        return finished_rows

    def _retire(self, rows: list[int]) -> list[Request]:
        done = []
        # Highest row first, so moving the last row never disturbs a row we
        # still have to process.
        for row in sorted(rows, reverse=True):
            req = self.active[row]
            last = len(self.active) - 1
            if row != last:
                mover = self.active[last]
                self.cache.move_row(src=last, dst=row, length=mover.length)
                self.active[row] = mover
            self.active.pop()
            done.append(req)
        return done

    # ------------------------------------------------------------------ API

    def step(self) -> list[Request]:
        """One scheduler iteration. Returns requests that finished in it."""
        done = self._admit()
        if self.active:
            done += self._retire(self._decode())
        return done

    def run(self, requests: list[Request]) -> list[Request]:
        for r in requests:
            self.submit(r)
        finished: list[Request] = []
        while self.waiting or self.active:
            finished += self.step()
        return sorted(finished, key=lambda r: r.rid)


@torch.inference_mode()
def autotune_padding(
    model: Qwen2, max_slots: int, max_rows: int | None = None,
    candidates: list[int] | None = None, context: int = 128, iters: int = 8,
) -> tuple[list[int], dict[int, float]]:
    """Measure decode step time at candidate batch sizes; return pad_to, where
    pad_to[n] (n = 0..max_slots) is the fastest candidate >= n, plus the measured
    step times (ms). Candidates go up to max_rows (default max_slots), so padding
    may use cache rows beyond the request slots.

    Measured rather than hardcoded because the fast and slow sizes come from
    cuBLAS kernel heuristics, which change with GPU and library version.
    """
    if candidates is None:
        candidates = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]
    max_rows = max_rows or max_slots
    candidates = sorted({c for c in candidates if c <= max_rows} | {max_slots, max_rows})
    device = model.norm.weight.device
    cache = KVCache(model.cfg, max_rows, context + 2, dtype=model.dtype, device=device)
    cache.window = context + 1
    ms: dict[int, float] = {}
    for c in candidates:
        toks = torch.zeros(c, 1, dtype=torch.long, device=device)
        pos = torch.full((c, 1), context, device=device)
        for i in range(iters + 2):
            if i == 2:
                torch.cuda.synchronize(); t = time.perf_counter()
            model(toks, positions=pos, cache=cache)
        torch.cuda.synchronize()
        ms[c] = (time.perf_counter() - t) * 1e3 / iters
    del cache
    torch.cuda.empty_cache()
    pad_to = [0] + [min((c for c in candidates if c >= n), key=lambda c: ms[c])
                    for n in range(1, max_slots + 1)]
    return pad_to, ms
