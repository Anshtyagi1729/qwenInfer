"""Model configuration, kept deliberately explicit.

We do not reuse HuggingFace's `PretrainedConfig` because part of the point of
this project is that nothing about the forward pass is hidden behind a library.
Fields are named to match the HF checkpoint JSON so that loading is a direct
mapping rather than a translation layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Qwen2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def n_kv_groups(self) -> int:
        """How many query heads share each key/value head."""
        return self.num_attention_heads // self.num_key_value_heads

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """Exact KV-cache cost of one token, across all layers.

        This is the number that governs how many sequences fit in memory, so it
        is worth being able to compute it without running anything.
        """
        per_layer = 2 * self.num_key_value_heads * self.head_dim * dtype_bytes
        return per_layer * self.num_hidden_layers

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen2Config":
        raw = json.loads(Path(path).read_text())
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            max_position_embeddings=raw["max_position_embeddings"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
        )
