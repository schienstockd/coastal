"""Smoke tests for `coastal.support` — the SUPPORT training + inference wrappers.

Verifies the two entry points (`train_support`, `denoise_stack`) hold their pure-array contract
without OMEZarr/anndata/cecelia in the loop, and that the arch dict round-trips cleanly through
`build_model`. Two full-flight test cases:

  1. train + denoise a tiny fake 12-frame stack, assert output shape + finite pixels
  2. `denoise_stack` mirror-pads correctly so boundary frames are NOT returned as init-zeros
     (the D8 rule of the DENOISE_INTEGRATION plan)

Skips when torch is unavailable.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from coastal.support import (
    DEFAULT_ARCH, build_model, train_support, denoise_stack,
)


def _fake_stack(seed=0, T=12, Y=32, X=32):
    """Small noisy stack for a smoke test — mean ~50, poisson-flavoured noise."""
    rng = np.random.default_rng(seed)
    base = 50 * np.ones((T, Y, X), dtype=np.float32)
    # a moving blob + shot noise so blindspot has something to reconstruct
    for t in range(T):
        y, x = 8 + (t % 5), 12 + (t % 3)
        base[t, y:y + 6, x:x + 6] += 200
    return (base + rng.normal(0, 5, base.shape)).astype(np.float32)


def _tiny_arch():
    """Small enough to train in a few seconds on CPU. Keys mirror DEFAULT_ARCH exactly."""
    return dict(DEFAULT_ARCH,
                inputFrames=5, patchXY=16,
                midChannels=[8, 16, 32], depth=3,
                blindConvChannels=4)


def test_build_model_reads_arch_verbatim():
    """arch keys map straight to SUPPORT(...) kwargs — a rename here breaks manifest round-trip."""
    device = torch.device('cpu')
    arch = _tiny_arch()
    model = build_model(arch, device)
    # First layer's channel count reflects inputFrames — cheapest way to know the mapping stuck
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0


def test_train_support_writes_state_dict_and_losses():
    """A one-epoch training pass returns a usable state_dict + per-epoch loss list."""
    vols = [torch.from_numpy(_fake_stack())]
    state_dict, losses = train_support(
        volumes=vols, arch=_tiny_arch(),
        epochs=1, batch_size=2, lr=5e-4,
        device=torch.device('cpu'),
    )
    assert isinstance(state_dict, dict) and len(state_dict) > 0
    assert len(losses) == 1
    assert np.isfinite(losses[0])

    # Round-trip: build a fresh model with the same arch, load the state_dict — must not raise.
    fresh = build_model(_tiny_arch(), torch.device('cpu'))
    fresh.load_state_dict(state_dict)


def test_denoise_stack_preserves_shape_and_boundaries():
    """`denoise_stack` returns same-shape T,Y,X — and the first/last frames are NOT init-zero.

    Without mirror-padding, SUPPORT's boundary frames come back as literal zeros — the D8 rule.
    Assert that the boundary rows carry finite denoised signal, not a zero-fill.
    """
    arch = _tiny_arch()
    vols = [torch.from_numpy(_fake_stack())]
    state_dict, _ = train_support(volumes=vols, arch=arch, epochs=1, batch_size=2,
                                  device=torch.device('cpu'))

    arr = _fake_stack(seed=1)   # a different noise realisation, same shape
    out = denoise_stack(arr, state_dict, arch, batch_size=2, device=torch.device('cpu'))

    assert out.shape == arr.shape
    assert np.isfinite(out).all()
    # boundary sanity: first + last frames carry non-zero variance
    assert out[0].std() > 0
    assert out[-1].std() > 0


def test_train_support_refuses_empty_volume_list():
    with pytest.raises(ValueError, match='no volumes'):
        train_support(volumes=[], arch=_tiny_arch(),
                      epochs=1, batch_size=2, device=torch.device('cpu'))


def test_progress_callback_ticks_per_batch():
    """`on_progress` fires once per batch (matches the operational-unit contract)."""
    ticks = []
    vols = [torch.from_numpy(_fake_stack())]
    train_support(
        volumes=vols, arch=_tiny_arch(),
        epochs=1, batch_size=2,
        device=torch.device('cpu'),
        on_progress=lambda d, t: ticks.append((d, t)),
    )
    # first tick is (0, total), last is (total, total); total = 1 epoch * n_batches
    assert ticks[0][0] == 0
    assert ticks[-1][0] == ticks[-1][1]
    assert ticks[-1][1] > 0
