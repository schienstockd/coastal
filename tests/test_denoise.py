"""Golden-value parity test for coastal.denoise.

coastal.denoise reimplements Cellpose 3's CPnet + restoration harness so it can load Cellpose's
public weights WITHOUT importing cellpose at runtime (see coastal/denoise.py, THIRD_PARTY.md,
docs/todo/DENOISE_PLAN.md). This test pins that reimplementation to the reference: on CPU, coastal
and cellpose must produce the same restored image from the same weights.

Skips cleanly when cellpose (test-only dev extra) or the pretrained weights are unavailable.
"""

import numpy as np
import pytest

from coastal.denoise import (
    DenoiseModel, normalize99, _get_pad_yx, ratio_preserving_gain, denoise_preserving_ratio,
    gaussian_restorer, cellpose_restorer, temporal_mean_restorer,
)


def _synthetic_image(seed=0, n=256):
    """A few Gaussian blobs ('cells') + shot-like noise — grayscale uint16."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.zeros((n, n), np.float32)
    for _ in range(12):
        cy, cx = rng.integers(20, n - 20, size=2)
        r = rng.uniform(6, 14)
        img += 800 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
    img = img + rng.normal(0, 40, img.shape) + rng.poisson(np.clip(img, 0, None) * 0.05)
    return np.clip(img, 0, 65535).astype(np.uint16)


# ---- unit-level checks that need no weights -------------------------------------------------

def test_normalize99_maps_percentiles():
    x = np.linspace(0, 100, 10000).astype(np.float32)
    y = normalize99(x, 1, 99)
    assert abs(np.percentile(y, 1) - 0.0) < 1e-3
    assert abs(np.percentile(y, 99) - 1.0) < 1e-3


def test_pad_is_divisible_by_16():
    ypad1, ypad2, xpad1, xpad2 = _get_pad_yx(250, 300)
    assert (250 + ypad1 + ypad2) % 16 == 0
    assert (300 + xpad1 + xpad2) % 16 == 0


# ---- golden parity vs cellpose --------------------------------------------------------------

def test_matches_cellpose_denoise():
    cellpose_denoise = pytest.importorskip(
        "cellpose.denoise", reason="cellpose is a test-only extra; install .[dev] to run parity")
    try:
        ref = cellpose_denoise.DenoiseModel(model_type="denoise_cyto3", gpu=False)
    except Exception as e:  # weights not downloadable in this env
        pytest.skip(f"cellpose weights unavailable: {e}")

    img = _synthetic_image()

    # cellpose path — exactly how cecelia calls it (grayscale, channels=[0,0], per-plane).
    ref_out = ref.eval([img], channels=[0, 0], diameter=None)[0][..., 0]

    # coastal path — CPU to match cellpose's fp32 CPU inference deterministically.
    got = DenoiseModel(model_type="denoise_cyto3", device="cpu").eval(img, diameter=None)

    assert got.shape == ref_out.shape
    max_abs = np.abs(got - ref_out).max()
    corr = np.corrcoef(got.ravel(), ref_out.ravel())[0, 1]
    # Same weights + same math → near bit-identical; allow float op-ordering slack.
    assert corr > 0.9999, f"correlation {corr:.5f} too low"
    assert max_abs < 1e-2, f"max abs diff {max_abs:.4g} too high"


# ---- ratio-preserving restoration (model-free algebra) --------------------------------------
#
# Confetti identity is the ratio between channels; the whole reason this path exists is that
# restoring channels independently destroys it. These pin the property, not the pixel values.

def _confetti_stack(seed=0, n=64, nc=3):
    """(C, Y, X) where each 'cell' is bright in exactly one channel — a confetti stand-in."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    a = rng.uniform(4, 8, size=(nc, n, n)).astype(np.float32)     # background
    for c in range(nc):
        for _ in range(3):
            cy, cx = rng.integers(10, n - 10, size=2)
            a[c] += 90 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 5.0 ** 2))
    return a


def test_gain_preserves_channel_ratios():
    a = _confetti_stack()
    # Any restored projection will do — the property must not depend on what the net produced.
    restored = a.mean(0) * 0.4 + 3.0
    out = ratio_preserving_gain(a, restored, cap=1e9)
    ref, got = a[0] / a[1], out[0] / out[1]
    assert np.abs(ref - got).max() < 1e-4
    assert (a.argmax(0) == out.argmax(0)).all(), "dominant channel must be identical everywhere"


def test_gain_projection_equals_restored_projection():
    """Where the cap is slack, the output's projection IS the restored projection."""
    a = _confetti_stack(seed=1)
    restored = a.mean(0) * 0.4 + 3.0
    out = ratio_preserving_gain(a, restored, cap=1e9)
    assert np.abs(out.mean(0) - restored).max() < 1e-3


def test_gain_cap_bounds_amplification():
    a = _confetti_stack(seed=2)
    restored = a.mean(0) * 50.0            # absurd amplification the cap must contain
    out = ratio_preserving_gain(a, restored, cap=2.0)
    assert (out <= a * 2.0 + 1e-4).all()


def test_gain_preserves_ratios_even_when_cap_binds():
    """Clipping rescales the shared factor; it must not tilt the ratio between channels.

    On real movies the cap binds on ~0.3% of pixels, some of them inside cells, so this is the
    case that decides whether identity really is preserved everywhere or only in the easy region.
    """
    a = _confetti_stack(seed=6)
    out = ratio_preserving_gain(a, a.mean(0) * 50.0, cap=2.0)   # cap binds on every pixel
    assert np.abs(a[0] / a[1] - out[0] / out[1]).max() < 1e-4
    assert (a.argmax(0) == out.argmax(0)).all()


def test_gain_unselected_channels_pass_through():
    """A non-confetti channel (e.g. second harmonic) must come back untouched."""
    a = _confetti_stack(nc=4)
    out = ratio_preserving_gain(a, a[[1, 2, 3]].mean(0) * 0.5, channels=[1, 2, 3])
    assert np.array_equal(out[0], a[0])
    assert not np.allclose(out[1], a[1])


def test_gain_handles_zero_pixels():
    a = np.zeros((3, 8, 8), np.float32)
    a[0, 4, 4] = 10.0
    out = ratio_preserving_gain(a, np.ones((8, 8), np.float32))
    assert np.isfinite(out).all()


def test_gain_channel_axis_and_leading_dims():
    """(T, C, Y, X) with channel_axis=1 must equal per-timepoint (C, Y, X) calls."""
    a = np.stack([_confetti_stack(seed=s) for s in (3, 4)])       # (T, C, Y, X)
    restored = a[:, :].mean(1) * 0.6
    got = ratio_preserving_gain(a, restored, channel_axis=1)
    for t in range(a.shape[0]):
        assert np.abs(got[t] - ratio_preserving_gain(a[t], restored[t])).max() < 1e-5


def test_gain_rejects_mismatched_restored_shape():
    with pytest.raises(ValueError):
        ratio_preserving_gain(_confetti_stack(), np.zeros((8, 8), np.float32))


def test_denoise_preserving_ratio_needs_no_weights():
    """The DEFAULT path must run with no Cellpose weights at all — that is the point of it.

    Getting off Cellpose restoration is the goal of docs/todo/DENOISE_PLAN.md; a default that
    silently downloads CPnet weights would quietly undo that, so pin it.
    """
    a = _confetti_stack(seed=5, n=128)
    out = denoise_preserving_ratio(a)
    assert out.shape == a.shape
    assert (a.argmax(0) == out.argmax(0)).all()
    assert np.isfinite(out).all()
    # Input units — not the Cellpose restoration range (~[-1, 10]).
    assert 0.2 < np.percentile(out, 25) / np.percentile(a, 25) < 5.0


def test_denoise_preserving_ratio_passes_flat_planes_through():
    """A constant plane must survive untouched — drift correction pads real stacks with them."""
    a = np.full((3, 64, 64), 7.0, np.float32)
    assert np.allclose(denoise_preserving_ratio(a), a)


def test_gaussian_restorer_smooths_without_shifting_level():
    r = gaussian_restorer(sigma=1.0)
    rng = np.random.default_rng(0)
    plane = (50 + rng.normal(0, 5, (64, 64))).astype(np.float32)
    out = r(plane)
    assert out.std() < plane.std()                      # it did smooth
    assert abs(out.mean() - plane.mean()) < 0.5         # without moving the level


def test_temporal_mean_restorer_averages_along_time_only():
    """It must mix frames, never neighbouring pixels — that is what separates it from a blur."""
    t = np.zeros((5, 16, 16), np.float32)
    t[2, 8, 8] = 10.0                                   # one bright pixel, one frame
    out = temporal_mean_restorer(window=3)(t)
    assert out[2, 8, 8] == pytest.approx(10 / 3, rel=1e-3)   # spread over 3 frames
    assert out[1, 8, 8] == pytest.approx(10 / 3, rel=1e-3)
    assert out[2, 8, 9] == 0.0                          # neighbours untouched: no spatial mixing


def test_temporal_mean_restorer_rejects_a_single_plane():
    """A (Y, X) input has no time axis; averaging it would silently return the input."""
    with pytest.raises(ValueError):
        temporal_mean_restorer()(np.zeros((16, 16), np.float32))


def test_temporal_restorer_in_the_wrapper_keeps_identity():
    """The reason to wrap it: averaging channels directly costs identity, the gain does not.

    Measured on real movies — averaging the channels gives 97.6% dominant-channel agreement,
    the same average applied through the gain gives 99.5%, at identical segmentation.
    """
    rng = np.random.default_rng(3)
    a = np.stack([_confetti_stack(seed=s, n=48) for s in range(6)])      # (T, C, Y, X)
    a = a + rng.normal(0, 2, a.shape).astype(np.float32)
    out = denoise_preserving_ratio(a, channel_axis=1,
                                   restorer=temporal_mean_restorer(3), cap=1e9)
    assert out.shape == a.shape
    assert np.abs(a[:, 0] / a[:, 1] - out[:, 0] / out[:, 1]).max() < 1e-3


def test_cellpose_restorer_is_opt_in_and_still_preserves_ratios():
    """The net remains available as an explicit choice, with the same identity guarantee."""
    try:
        m = DenoiseModel(model_type="denoise_cyto3", device="cpu")
    except Exception as e:
        pytest.skip(f"weights unavailable: {e}")
    a = _confetti_stack(seed=7, n=128)
    out = denoise_preserving_ratio(a, restorer=cellpose_restorer(_model=m), cap=1e9)
    assert (a.argmax(0) == out.argmax(0)).all()
    assert np.abs(a[0] / a[1] - out[0] / out[1]).max() < 1e-4


def test_stack_matches_per_plane():
    """A (Z,Y,X) stack call must equal per-plane calls (batching must not change results)."""
    cellpose_denoise = pytest.importorskip("cellpose.denoise")
    try:
        DenoiseModel(model_type="denoise_cyto3", device="cpu")
    except Exception as e:
        pytest.skip(f"weights unavailable: {e}")
    m = DenoiseModel(model_type="denoise_cyto3", device="cpu")
    stack = np.stack([_synthetic_image(1), _synthetic_image(2), _synthetic_image(3)])
    batched = m.eval(stack, diameter=None)
    per_plane = np.stack([m.eval(stack[z], diameter=None) for z in range(3)])
    assert np.abs(batched - per_plane).max() < 1e-3
