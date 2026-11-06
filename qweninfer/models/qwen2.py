"""Qwen2 (Qwen2.5-0.5B) inference, written from scratch.

Every operation of the forward pass is explicit here: nothing about attention,
positions or caching is hidden inside a library call. Correctness is defined as
logit parity with HuggingFace's Qwen2ForCausalLM (tests/test_parity.py).

Design decisions, each explained where it is implemented:
  1. Module names mirror the checkpoint, so loading is a direct mapping.
  2. The model is built on the meta device, so no weights are initialised only
     to be overwritten.
  3. Attention masks come from absolute positions, never from tensor shapes.
     One rule then covers prefill, decode and ragged batches.
  4. GQA folds query groups into the row dimension instead of copying K/V.
  5. Logits can be computed for the last position only.
  6. For inference, weights are repacked once at load time (`fuse`): transposed
     to (in, out) and q/k/v and gate/up concatenated. Decode multiplies ONE row
     by each weight matrix, and on this GPU the transposed layout runs at 91% of
     memory bandwidth versus 34% for nn.Linear's layout (scripts/gemv_layout.py).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from qweninfer.models.config import Qwen2Config


#instead of layernorm because the scaling mattered so we skipped the mean in the norm and went with RMSNorm
class RMSNorm(nn.Module):
    def __init__(self,hidden_dim:int,eps:float=1e-6):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(hidden_dim))
    def forward(self,x:torch.Tensor):
        org_dtype=x.dtype
        sq=x.to(torch.float32).pow(2).mean(-1,keepdim=True) #to avoid the overfloow
        rms_inv=torch.rsqrt(sq+self.eps)
        return x*rms_inv.to(org_dtype)*self.weight


# --------------------------------------------------------------------- RoPE

def build_rope_cache(
    head_dim: int, max_seq_len: int, theta: float, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables of shape (max_seq_len, head_dim), in fp32.

    The head vector is treated as head_dim/2 independent 2-D planes. Plane i is
    rotated by position * theta^(-2i/head_dim): low planes spin fast and encode
    fine position, high planes spin slowly and encode long range.

    Kept in fp32 regardless of model dtype: at position 30,000 the angle is large,
    and fp16 cannot represent it precisely enough for cos/sin to be right.
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )                                                          # (head_dim/2,)
    pos = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(pos, inv_freq)                        # (max_seq, head_dim/2)
    # HF pairs dimension i with i + head_dim/2 ("halves"), so both members of a
    # pair need the same angle: duplicate the table side by side.
    emb = torch.cat([angles, angles], dim=-1)                  # (max_seq, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """For pairs (x[i], x[i+half]), return (-x[i+half], x[i]).

    From the 2-D rotation x' = x cos - y sin, y' = x sin + y cos: the sin term
    needs the partner coordinate, negated for the first member of each pair.
    """
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q (B,H,S,hd) and k (B,Hkv,S,hd) by their token positions.

    `positions` is (S,) when the whole batch shares positions, or (B,S) when each
    sequence has its own, which is the normal case once sequences of different
    lengths decode together. Indexing gives (S,hd) or (B,S,hd); inserting a head
    axis lets one table broadcast over both 14 query heads and 2 KV heads.
    """
    c = cos[positions].unsqueeze(-3).to(q.dtype)
    s = sin[positions].unsqueeze(-3).to(q.dtype)
    return q * c + _rotate_half(q) * s, k * c + _rotate_half(k) * s


# ---------------------------------------------------------------------- MLP

class MLP(nn.Module):
    """SwiGLU: down( silu(gate(x)) * up(x) ). No biases anywhere in Qwen2's MLP."""

    def __init__(self, cfg: Qwen2Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "w_gate_up"):
            gate, up = (x @ self.w_gate_up).chunk(2, dim=-1)
            return (F.silu(gate) * up) @ self.w_down
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    @torch.no_grad()
    def fuse(self) -> None:
        """Repack for inference: one (d, 2I) matrix for gate+up, transposed down.
        Same function; 3 matmuls become 2, in the layout decode GEMV is fast in."""
        gate_up = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
        self.register_buffer("w_gate_up", gate_up.t().contiguous())   # (896, 9728)
        self.register_buffer("w_down", self.down_proj.weight.t().contiguous())  # (4864, 896)
        del self.gate_proj, self.up_proj, self.down_proj


# ---------------------------------------------------------------- attention

class Attention(nn.Module):
    def __init__(self, cfg: Qwen2Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = cfg.num_attention_heads          # 14
        self.n_kv_heads = cfg.num_key_value_heads       # 2
        self.head_dim = cfg.head_dim                    # 64
        self.n_rep = self.n_heads // self.n_kv_heads    # 7 query heads per KV head
        self.scale = 1.0 / math.sqrt(self.head_dim)     # the HEAD dim, not 896

        d = cfg.hidden_size
        # The 896 -> 128 shapes of k/v *are* GQA. Qwen2 quirk: q/k/v have biases,
        # o_proj does not.
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        positions: torch.Tensor, cache=None,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        H, Hkv, hd, R = self.n_heads, self.n_kv_heads, self.head_dim, self.n_rep

        # Projection output stores head 0 in features 0..63, head 1 in 64..127...
        # so view() splits heads correctly, then transpose puts heads before
        # sequence. Reshaping directly to (B,H,S,hd) would scramble heads with
        # positions.
        if hasattr(self, "w_qkv"):
            q, k, v = (x @ self.w_qkv + self.b_qkv).split(
                [H * hd, Hkv * hd, Hkv * hd], dim=-1)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(B, S, H, hd).transpose(1, 2)                   # (B,14,S,64)
        k = k.view(B, S, Hkv, hd).transpose(1, 2)                 # (B, 2,S,64)
        v = v.view(B, S, Hkv, hd).transpose(1, 2)                 # (B, 2,S,64)

        # Rotate before caching, so the cache holds already-rotated keys and
        # past keys never need re-rotating.
        q, k = apply_rope(q, k, cos, sin, positions)

        # With a cache, k/v become the full history: (B, Hkv, T, hd), where slot
        # j of the cache holds the token at absolute position j. Without one,
        # T == S and the same invariant holds for positions 0..S-1.
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v, positions)
        T = k.shape[-2]

        # GQA without copying K/V: the 7 query heads sharing a KV head are folded
        # into the row dimension, so each KV head does one matmul against 7*S
        # query rows. Query head h = g*7 + r belongs to KV head g, which is
        # exactly the contiguous split view() gives here.
        q = q.reshape(B, Hkv, R * S, hd)
        scores = (q @ k.transpose(-2, -1)) * self.scale            # (B,2,7S,T)

        # The mask comes from absolute positions: key j is visible to a query at
        # position p iff j <= p. That one rule is causal masking during prefill,
        # needs no special case during decode, and hides stale cache slots of
        # shorter sequences in a ragged batch.
        key_pos = torch.arange(T, device=x.device)
        visible = key_pos.view(1, 1, T) <= positions.unsqueeze(-1)  # (B,S,T)
        scores = scores.view(B, Hkv, R, S, T).masked_fill(
            ~visible[:, None, None], float("-inf")
        )

        # Softmax in fp32: in fp16, long rows of tiny probabilities lose accuracy.
        probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        out = probs.view(B, Hkv, R * S, T) @ v                     # (B,2,7S,64)

        # Undo the fold: back to (B,14,S,64), then merge heads to (B,S,896).
        out = out.view(B, H, S, hd).transpose(1, 2).reshape(B, S, H * hd)
        if hasattr(self, "w_o"):
            return out @ self.w_o
        return self.o_proj(out)

    @torch.no_grad()
    def fuse(self) -> None:
        """Repack for inference: q/k/v as one (896, 1152) matrix plus one bias,
        o_proj transposed. One matmul instead of three per layer."""
        w = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], 0)
        b = torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], 0)
        self.register_buffer("w_qkv", w.t().contiguous())         # (896, 1152)
        self.register_buffer("b_qkv", b.contiguous())             # (1152,)
        self.register_buffer("w_o", self.o_proj.weight.t().contiguous())  # (896, 896)
        del self.q_proj, self.k_proj, self.v_proj, self.o_proj


# ------------------------------------------------------------- block, model

class Block(nn.Module):
    """Pre-norm block: each sublayer reads a normalised copy of the residual
    stream and adds its result back; the stream itself is never normalised."""

    def __init__(self, cfg: Qwen2Config, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, layer_idx)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, positions, cache=None) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, positions, cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen2(nn.Module):
    def __init__(self, cfg: Qwen2Config, max_seq_len: int = 4096) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_seq_len = max_seq_len
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Block(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # Buffers, not parameters: derived constants that should follow .to(),
        # but are not trained and are not in the checkpoint (persistent=False).
        cos, sin = build_rope_cache(cfg.head_dim, max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        return self.norm.weight.dtype

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor | None = None,
        cache=None, last_only: bool = False,
    ) -> torch.Tensor:
        B, S = input_ids.shape
        if positions is None:
            positions = torch.arange(S, device=input_ids.device).expand(B, S)
        elif positions.dim() == 1:
            positions = positions.expand(B, S)

        if hasattr(self, "embed_t"):
            # Lookup from the transposed table: column gather, (d,B,S) -> (B,S,d).
            x = self.embed_t[:, input_ids].permute(1, 2, 0).contiguous()
        else:
            x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin, positions, cache)
        x = self.norm(x)

        # The output head is 896 x 151936 per token. Generation only needs the
        # last position; a 1024-token prompt would otherwise produce 311 MB of
        # fp16 logits just to throw 1023 rows away.
        if last_only:
            x = x[:, -1:]
        # Tied embeddings: the output head IS the input embedding matrix.
        if hasattr(self, "embed_t"):
            return x @ self.embed_t
        return F.linear(x, self.embed_tokens.weight)

    @torch.no_grad()
    def fuse(self) -> None:
        """Repack every layer for inference. Call on CPU before moving to the GPU,
        so the original and repacked weights never coexist in VRAM.

        The tied embedding is used twice with opposite layout needs: as a lookup
        table (one 1.8 KB row per input token) and as the output head (a full
        272 MB read per generated token, the largest matmul in the model). Keeping
        both layouts would cost 272 MB of a ~630 MB budget, so it is stored the way
        the expensive op wants, and the cheap lookup does a strided gather."""
        for layer in self.layers:
            layer.self_attn.fuse()
            layer.mlp.fuse()
        self.register_buffer("embed_t", self.embed_tokens.weight.t().contiguous())
        del self.embed_tokens

    @classmethod
    def from_pretrained(
        cls, path: str | Path, dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda", max_seq_len: int = 4096,
        fuse: bool = True,
    ) -> "Qwen2":
        path = Path(path)
        cfg = Qwen2Config.from_json(path / "config.json")

        # Meta tensors have shape and dtype but no storage: building 494M
        # parameters this way costs nothing, instead of random-initialising
        # weights we are about to overwrite.
        with torch.device("meta"):
            model = cls(cfg, max_seq_len)

        state = load_file(path / "model.safetensors", device="cpu")
        state = {k.removeprefix("model."): t.to(dtype) for k, t in state.items()}
        # strict=True fails loudly on any missing or unexpected tensor. A weight
        # left unloaded would otherwise sit at init and produce silent garbage.
        model.load_state_dict(state, strict=True, assign=True)

        # The RoPE buffers were built on meta too; give them real storage.
        model.rope_cos, model.rope_sin = build_rope_cache(
            cfg.head_dim, max_seq_len, cfg.rope_theta
        )
        if fuse:
            model.fuse()          # on CPU, before .to(device): never two copies in VRAM
        return model.to(device).eval()
