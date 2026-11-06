"""Parity harness: your Qwen2 implementation vs HuggingFace's.

HuggingFace is the oracle. No reference math is written here: each test either
instantiates HF's equivalent module or runs HF's full model, and compares.

=============================================================================
API this harness assumes from `qweninfer/models/qwen2.py`
=============================================================================

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6)
        # exposes self.weight of shape (hidden_size,)
        # forward(x: (..., hidden_size)) -> same shape

    def build_rope_cache(head_dim: int, max_seq_len: int, theta: float)
        -> tuple[cos, sin]              # layout is YOUR choice

    def apply_rope(q, k, cos, sin, positions) -> tuple[q_out, k_out]
        # q: (B, H, S, hd)   k: (B, Hkv, S, hd)   positions: (S,) int64

    class MLP(nn.Module):
        def __init__(self, cfg: Qwen2Config)
        # exposes .gate_proj, .up_proj, .down_proj  (checkpoint naming)

    class Qwen2(nn.Module):
        @classmethod
        def from_pretrained(cls, path, dtype=torch.float32, device="cpu") -> Qwen2
        def forward(self, input_ids: (B, S)) -> logits: (B, S, vocab)

"""

from __future__ import annotations

import inspect

import pytest
import torch

from conftest import need, report

HEAD_DIM = 64
HIDDEN = 896
TOL_F32 = 2e-5          # component tests: fp32, so demand real precision
TOL_LOGITS_F32 = 2e-3   # 24 layers of accumulated fp32 drift


# ---------------------------------------------------------------- components

def test_rmsnorm(student, cfg):
    """Your RMSNorm vs HF's Qwen2RMSNorm on random input."""
    (RMSNorm,) = need(student, "RMSNorm")
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    torch.manual_seed(0)
    w = torch.randn(HIDDEN) * 0.1 + 1.0
    x = torch.randn(2, 7, HIDDEN)

    ref = Qwen2RMSNorm(HIDDEN, eps=cfg.rms_norm_eps)
    with torch.no_grad():
        ref.weight.copy_(w)

    mine = RMSNorm(HIDDEN, cfg.rms_norm_eps)
    with torch.no_grad():
        mine.weight.copy_(w)

    with torch.no_grad():
        got, exp = mine(x), ref(x)

    if got.shape != exp.shape or not torch.allclose(got, exp, atol=TOL_F32):
        pytest.fail(report("rmsnorm", got, exp))


def test_rope(student, cfg, hf_config):
    """Your build_rope_cache + apply_rope vs HF's rotary embedding + apply.

    This tests the *composition*, so your cos/sin layout can differ from HF's as
    long as the resulting rotation matches. The common bug lives here: HF rotates across
    halves of the head dim, not between adjacent pairs. A wrong convention
    produces fluent-but-wrong text later, so pin it now.
    """
    build_rope_cache, apply_rope = need(student, "build_rope_cache", "apply_rope")
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2RotaryEmbedding,
        apply_rotary_pos_emb,
    )

    torch.manual_seed(0)
    B, S, H, Hkv = 1, 11, cfg.num_attention_heads, cfg.num_key_value_heads
    q = torch.randn(B, H, S, HEAD_DIM)
    k = torch.randn(B, Hkv, S, HEAD_DIM)
    positions = torch.arange(S)

    # --- reference
    rot = Qwen2RotaryEmbedding(config=hf_config)
    cos_ref, sin_ref = rot(q, positions.unsqueeze(0))
    q_exp, k_exp = apply_rotary_pos_emb(q, k, cos_ref, sin_ref)

    # --- yours (tolerate optional device/dtype kwargs)
    kwargs = {}
    params = inspect.signature(build_rope_cache).parameters
    if "device" in params:
        kwargs["device"] = torch.device("cpu")
    if "dtype" in params:
        kwargs["dtype"] = torch.float32
    cos, sin = build_rope_cache(
        HEAD_DIM, cfg.max_position_embeddings, cfg.rope_theta, **kwargs
    )
    q_got, k_got = apply_rope(q, k, cos, sin, positions)

    if q_got.shape != q_exp.shape:
        pytest.fail(report("rope q", q_got, q_exp))
    if k_got.shape != k_exp.shape:
        pytest.fail(report("rope k", k_got, k_exp))

    # Diagnose the classic failure rather than just failing.
    if not torch.allclose(q_got, q_exp, atol=TOL_F32):
        from transformers.models.qwen2.modeling_qwen2 import rotate_half

        half = HEAD_DIM // 2
        interleaved = q.clone()
        x1, x2 = q[..., 0::2], q[..., 1::2]
        c = cos_ref[..., :half].unsqueeze(1)
        s = sin_ref[..., :half].unsqueeze(1)
        interleaved[..., 0::2] = x1 * c - x2 * s
        interleaved[..., 1::2] = x1 * s + x2 * c
        hint = ""
        if torch.allclose(q_got, interleaved, atol=1e-4):
            hint = (
                "\n  >> Your rotation matches the INTERLEAVED (pair-adjacent) "
                "convention. HF/Qwen2 rotates across HALVES of the head dim: "
                "split into x[:32] and x[32:]. The pretrained weights assume "
                "HF's choice, so this must change."
            )
        elif torch.allclose(q_got.abs(), q_exp.abs(), atol=1e-4):
            hint = "\n  >> Magnitudes match but signs differ; check the sign in rotate_half."
        pytest.fail(report("rope q", q_got, q_exp) + hint)

    if not torch.allclose(k_got, k_exp, atol=TOL_F32):
        pytest.fail(report("rope k", k_got, k_exp))


def test_mlp(student, cfg, hf_config):
    """Your SwiGLU MLP vs HF's Qwen2MLP, same random weights."""
    (MLP,) = need(student, "MLP")
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    torch.manual_seed(0)
    ref = Qwen2MLP(hf_config)
    mine = MLP(cfg)

    try:
        with torch.no_grad():
            for name in ("gate_proj", "up_proj", "down_proj"):
                getattr(mine, name).weight.copy_(getattr(ref, name).weight)
    except AttributeError as e:
        pytest.skip(f"MLP needs .gate_proj/.up_proj/.down_proj for this test: {e}")

    x = torch.randn(2, 5, HIDDEN)
    with torch.no_grad():
        got, exp = mine(x), ref(x)

    if got.shape != exp.shape or not torch.allclose(got, exp, atol=TOL_F32):
        pytest.fail(report("mlp", got, exp))


# ---------------------------------------------------------------- full model

def _find_blocks(model):
    """Locate your per-layer module list without dictating what you called it."""
    for attr in ("layers", "blocks", "h"):
        obj = model
        for part in ([attr], ["model", attr]):
            cur = obj
            ok = True
            for p in part:
                if not hasattr(cur, p):
                    ok = False
                    break
                cur = getattr(cur, p)
            if ok and isinstance(cur, torch.nn.ModuleList):
                return cur
    return None


@pytest.fixture(scope="module")
def my_model(student, weights_dir):
    (Qwen2,) = need(student, "Qwen2")
    if not hasattr(Qwen2, "from_pretrained"):
        pytest.skip("Qwen2.from_pretrained not implemented yet")
    m = Qwen2.from_pretrained(weights_dir, dtype=torch.float32, device="cpu")
    m.eval()
    return m


@pytest.mark.slow
def test_hidden_states_per_layer(my_model, hf_model, prompt_ids):
    """Compare every layer's output and report the FIRST one that diverges.

    The most useful diagnostic in the suite: instead of "logits wrong", you
    learn "layer 0 is fine, layer 1 diverges", which localises the bug to one
    component. If layer 0 already diverges, the problem is in embedding,
    attention, RoPE or the MLP; if all layers are fine but logits are wrong, the
    problem is the final norm or the tied output head.
    """
    blocks = _find_blocks(my_model)
    if blocks is None:
        pytest.skip("could not find your block ModuleList (tried .layers/.blocks/.h)")

    captured: list[torch.Tensor] = []
    handles = [
        b.register_forward_hook(
            lambda _m, _i, out: captured.append(
                (out[0] if isinstance(out, tuple) else out).detach()
            )
        )
        for b in blocks
    ]
    try:
        with torch.no_grad():
            my_model(prompt_ids)
    finally:
        for h in handles:
            h.remove()

    with torch.no_grad():
        ref = hf_model(prompt_ids, output_hidden_states=True).hidden_states
        # hidden_states[0] is the embedding and [i+1] is the output of block i,
        # EXCEPT the last entry: HF stores the final-normed stream there, not
        # block 23's raw output. Normalise ours the same way before comparing
        # (verified: raw differs by 1.7e2, normed by 2e-4).
        if hasattr(my_model, "norm"):
            captured[-1] = my_model.norm(captured[-1])
    ref_blocks = ref[1:]

    assert len(captured) == len(ref_blocks), (
        f"captured {len(captured)} block outputs, HF has {len(ref_blocks)}"
    )

    lines = []
    first_bad = None
    for i, (got, exp) in enumerate(zip(captured, ref_blocks)):
        if got.shape != exp.shape:
            pytest.fail(report(f"layer {i}", got, exp))
        d = (got.float() - exp.float()).abs().max().item()
        lines.append(f"  layer {i:>2}: max_abs={d:.3e}")
        if first_bad is None and d > TOL_LOGITS_F32:
            first_bad = i

    if first_bad is not None:
        ctx = "\n".join(lines[max(0, first_bad - 1) : first_bad + 2])
        where = (
            "embedding / attention / RoPE / MLP inside the block"
            if first_bad == 0
            else f"something layer-dependent: layers 0..{first_bad - 1} are correct, "
            "so the per-layer math is right and this is likely position handling "
            "or a weight-loading offset"
        )
        pytest.fail(
            f"first divergence at layer {first_bad} (tol {TOL_LOGITS_F32:.0e}):\n"
            f"{ctx}\n  >> look at: {where}"
        )


@pytest.mark.slow
def test_logit_parity(my_model, hf_model, prompt_ids):
    """The headline test: your logits vs HF's logits."""
    with torch.no_grad():
        got = my_model(prompt_ids)
        exp = hf_model(prompt_ids).logits

    got = got[0] if isinstance(got, tuple) else got
    if got.shape != exp.shape:
        pytest.fail(report("logits", got, exp))

    d = (got.float() - exp.float()).abs().max().item()
    if d >= TOL_LOGITS_F32:
        pytest.fail(
            report("logits", got, exp)
            + "\n  >> run test_hidden_states_per_layer to find which layer diverges"
        )


@pytest.mark.slow
def test_greedy_sequence_identical(my_model, hf_model, prompt_ids, tokenizer):
    """Argmax agreement is the test that actually matters for generation.

    Logits can be slightly off and still argmax identically; they can also be
    'close' and pick a different token at a near-tie. This checks the thing a
    user would notice.
    """
    n_new = 12
    ids = prompt_ids.clone()
    with torch.no_grad():
        for _ in range(n_new):
            out = my_model(ids)
            out = out[0] if isinstance(out, tuple) else out
            ids = torch.cat([ids, out[:, -1:].argmax(-1)], dim=1)

        # generation_config.json sets repetition_penalty=1.1, which generate()
        # applies even when greedy. Disable it: we are comparing models, not
        # decoding policies.
        ref_ids = hf_model.generate(
            prompt_ids, max_new_tokens=n_new, do_sample=False, use_cache=True,
            repetition_penalty=1.0,
        )

    mine_txt = tokenizer.decode(ids[0, prompt_ids.shape[1] :])
    ref_txt = tokenizer.decode(ref_ids[0, prompt_ids.shape[1] :])
    if not torch.equal(ids, ref_ids):
        pytest.fail(
            f"greedy sequences differ\n  yours: {mine_txt!r}\n  HF:    {ref_txt!r}\n"
            f"  your ids: {ids[0, prompt_ids.shape[1]:].tolist()}\n"
            f"  HF ids:   {ref_ids[0, prompt_ids.shape[1]:].tolist()}"
        )
