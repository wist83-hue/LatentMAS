"""Shared pytest fixtures.

Some tests need a real (tiny) HF model. We cache one ModelWrapper across the
session to keep test wall time reasonable.
"""
import os
import sys

import pytest
import torch

# Make the repo root importable from tests/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# Some heavyweight imports are guarded by a flag so the cheap tests can run
# even if torch isn't installed in a minimal CI environment.
HAS_TORCH = True
try:
    import torch  # noqa: F401
except ImportError:
    HAS_TORCH = False


TINY_MODEL_ID = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="session")
def tiny_args():
    """Minimal argparse-Namespace-like object for tests that touch ModelWrapper."""
    import argparse
    return argparse.Namespace(
        device="cpu",
        device2="cpu",
        task="gsm8k",
        method="latent_mas",
        prompt="sequential",
        max_new_tokens=64,
        seed=42,
        latent_steps=2,
        latent_steps_map=None,
        latent_space_realign=False,
        latent_halt_threshold=0.0,
        latent_halt_entropy_nats=0.0,
        latent_halt_argmax_steps=0,
        latent_halt_kl_nats=0.0,
        latent_halt_min_steps=3,
        inter_persona_anchor_tokens=0,
        pipeline=None,
        think=False,
        use_vllm=False,
        use_second_HF_model=False,
        enable_prefix_caching=False,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        latent_only=False,
        sequential_info_only=False,
        # The prompts.py builders assert model_name contains "qwen". Tests
        # actually load TINY_MODEL_ID (tiny-gpt2) via the fixture, but pass
        # a qwen-flavored args.model_name so the assertions pass.
        model_name="Qwen/qwen-test",
        generate_bs=2,
        temperature=0.7,
        top_p=0.95,
        text_mas_context_length=-1,
        latent_norm_mode="scalar_mean",
        latent_ablation="none",
        latent_decode_debug=False,
        latent_ood_debug=False,
        latent_feedback_mode="w_a",
        latent_soft_embed_temperature=2.0,
        text_mas_nonjudger_max_tokens=0,
        concise_nonjudger_prompt=False,
        latent_halt_on_eos=False,
        disable_thinking=False,
        minimal_persona_prompts=False,
    )


@pytest.fixture(scope="session")
def tiny_model_wrapper(tiny_args):
    """A ModelWrapper with sshleifer/tiny-gpt2 on CPU. ~5MB; loads in seconds."""
    from models import ModelWrapper
    mw = ModelWrapper(TINY_MODEL_ID, torch.device("cpu"), use_vllm=False, args=tiny_args)
    return mw


@pytest.fixture(autouse=True)
def _reset_tiny_model_args(request, tiny_args):
    """Tests that mutate `mw.args` (e.g. halt-criteria tests) shouldn't leak
    state into other tests via the session-scoped tiny_model_wrapper fixture.
    We snapshot args before each test and restore after.
    """
    if "tiny_model_wrapper" not in request.fixturenames:
        yield
        return
    mw = request.getfixturevalue("tiny_model_wrapper")
    saved = mw.args
    try:
        yield
    finally:
        mw.args = saved
        # Also clear any cached realignment matrices that depended on the args
        mw._latent_realign_matrices.clear()
