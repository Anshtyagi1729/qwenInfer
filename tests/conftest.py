"""Shared fixtures for the parity harness.

The oracle for every test is HuggingFace's own Qwen2 implementation, loaded from
the same checkpoint. We never hand-write reference math here: a test either
compares your module against HF's equivalent module, or it compares your logits
against HF's logits.

Component tests run on CPU in float32 on purpose. fp16 noise would force loose
tolerances, and loose tolerances hide exactly the bugs these tests exist to
catch (a transposed GQA expand can still land within 1e-2). End-to-end tests
additionally run in the real serving dtype.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "qwen2.5-0.5b-instruct"

# A prompt with enough tokens that position-dependent bugs (RoPE, masking)
# actually show up. A 2-token prompt would pass with broken RoPE.
PROMPT = "The capital of France is Paris, and the capital of Japan is"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads the full model")


@pytest.fixture(scope="session")
def weights_dir() -> Path:
    if not (WEIGHTS / "model.safetensors").exists():
        pytest.skip(f"checkpoint not downloaded yet: {WEIGHTS}")
    return WEIGHTS


@pytest.fixture(scope="session")
def cfg(weights_dir):
    from qweninfer.models.config import Qwen2Config

    return Qwen2Config.from_json(weights_dir / "config.json")


@pytest.fixture(scope="session")
def hf_config(weights_dir):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(weights_dir)


@pytest.fixture(scope="session")
def tokenizer(weights_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(weights_dir)


@pytest.fixture(scope="session")
def hf_model(weights_dir):
    """Reference model, CPU float32. This is the oracle."""
    from transformers import AutoModelForCausalLM

    # No device_map= (needs `accelerate`) and no torch_dtype= (deprecated in
    # transformers 5). Plain load, then eval; CPU is already the default.
    m = AutoModelForCausalLM.from_pretrained(weights_dir, dtype=torch.float32)
    m.eval()
    return m


@pytest.fixture(scope="session")
def prompt_ids(tokenizer):
    return tokenizer(PROMPT, return_tensors="pt").input_ids


@pytest.fixture(scope="session")
def student():
    """The implementation under test, or a skip if the module is missing.

    Missing pieces report as skips rather than collection errors, so the suite
    always runs.
    """
    try:
        import qweninfer.models.qwen2 as mod
    except ModuleNotFoundError:
        pytest.skip("qweninfer/models/qwen2.py not found")
    return mod


def need(mod, *names):
    """Skip unless every named symbol exists in your module."""
    missing = [n for n in names if not hasattr(mod, n)]
    if missing:
        pytest.skip(f"not implemented yet: {', '.join(missing)}")
    return tuple(getattr(mod, n) for n in names)


def report(name: str, mine: torch.Tensor, ref: torch.Tensor) -> str:
    """Uniform diff summary, so failures are readable instead of a wall of tensor."""
    if mine.shape != ref.shape:
        return f"{name}: SHAPE MISMATCH yours={tuple(mine.shape)} ref={tuple(ref.shape)}"
    d = (mine.float() - ref.float()).abs()
    denom = ref.float().abs().mean().clamp_min(1e-9)
    return (
        f"{name}: max_abs={d.max():.3e}  mean_abs={d.mean():.3e}  "
        f"rel={d.mean()/denom:.3e}  shape={tuple(ref.shape)}"
    )
