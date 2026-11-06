"""Preallocated KV cache, sliced to the live length on every read.

Measured against HuggingFace:
  DynamicCache grows by torch.cat each step. The tensors change size every
      step, so the caching allocator cannot reuse blocks: +15 ms/step at 1024
      tokens of context, against 0.26 ms of actual KV bandwidth.
  StaticCache preallocates, but attends over the whole reserved window every
      step: a flat ~56 ms regardless of how much context is real.

This cache allocates exactly once and returns a *view* truncated to the longest
live sequence, so it pays neither cost.

Layout invariant the model relies on: cache slot j of a sequence holds the token
at absolute position j. That is what lets attention mask with `j <= position`.
"""

from __future__ import annotations

import torch

from qweninfer.models.config import Qwen2Config


class KVCache:
    def __init__(
        self, cfg: Qwen2Config, max_batch: int, max_len: int,
        dtype: torch.dtype = torch.float16, device: str | torch.device = "cuda",
    ) -> None:
        shape = (cfg.num_hidden_layers, max_batch, cfg.num_key_value_heads,
                 max_len, cfg.head_dim)
        # One allocation for the life of the cache. A server reserves memory when
        # it admits a request; it must never OOM halfway through generating.
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_batch = max_batch
        self.max_len = max_len
        # Set by the caller before each forward pass: the number of positions
        # attention must read (longest live sequence after this step's writes).
        # Kept on the CPU on purpose: deriving it from `positions` on the GPU
        # would force a device->host sync in every layer of every step.
        self.window = 0
        # The forward pass operates on cache rows [row_start, row_start + B).
        # Prefilling a new request into slot s sets row_start = s with B = 1;
        # batched decode uses row_start = 0. A contiguous range keeps every read a
        # view rather than a gather.
        self.row_start = 0

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write this step's k/v (B, Hkv, S, hd) at `positions` (B, S), then
        return the cached history (B, Hkv, window, hd) for the same rows."""
        B = k.shape[0]
        r = slice(self.row_start, self.row_start + B)
        rows = torch.arange(B, device=k.device).unsqueeze(1)          # (B, 1)
        # Advanced indexing with rows (B,1) and positions (B,S) addresses an
        # (B, S) grid of cache slots; the indexed dims come first, so the
        # assigned values must be (B, S, Hkv, hd).
        self.k[layer, r][rows, :, positions] = k.transpose(1, 2)
        self.v[layer, r][rows, :, positions] = v.transpose(1, 2)
        # A slice is a view: no copy, and attention only does work proportional
        # to real context rather than reserved capacity.
        return self.k[layer, r, :, : self.window], self.v[layer, r, :, : self.window]

    def move_row(self, src: int, dst: int, length: int) -> None:
        """Copy one sequence's cache from row src to row dst (all layers).

        Used to keep live sequences packed into rows 0..n-1, so every decode step
        can read a contiguous slice instead of gathering scattered rows. The copy
        happens once per finished request, not once per step.
        """
        self.k[:, dst, :, :length] = self.k[:, src, :, :length]
        self.v[:, dst, :, :length] = self.v[:, src, :, :length]

    def memory_bytes(self) -> int:
        return 2 * self.k.numel() * self.k.element_size()
