"""Golden-value parity test for coastal.denoise.

coastal.denoise reimplements Cellpose 3's CPnet + restoration harness so it can load Cellpose's
public weights WITHOUT importing cellpose at runtime (see coastal/denoise.py, THIRD_PARTY.md,
docs/todo/DENOISE_PLAN.md). This test pins that reimplementation to the reference: on CPU, coastal
and cellpose must produce the same restored image from the same weights.

Skips cleanly when cellpose (test-only dev extra) or the pretrained weights are unavailable.
"""

import numpy as np
import pytest

from coastal.denoise import DenoiseModel, normalize99, _get_pad_yx


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


def _parity(model_type, diameter):
    """coastal vs cellpose on one image → (got, ref_out). Skips if cellpose/weights unavailable."""
    cellpose_denoise = pytest.importorskip("cellpose.denoise")
    try:
        ref = cellpose_denoise.DenoiseModel(model_type=model_type, gpu=False)
    except Exception as e:
        pytest.skip(f"cellpose weights unavailable: {e}")
    img = _synthetic_image()
    ref_out = ref.eval([img], channels=[0, 0], diameter=diameter)[0][..., 0]
    got = DenoiseModel(model_type=model_type, device="cpu").eval(img, diameter=diameter)
    return got, ref_out


def test_matches_cellpose_deblur():
    """Deblur model (+ the diameter/rescale path) must match cellpose too — not just denoise."""
    got, ref_out = _parity("deblur_cyto3", diameter=10.)
    assert got.shape == ref_out.shape
    corr = np.corrcoef(got.ravel(), ref_out.ravel())[0, 1]
    assert corr > 0.9999, f"deblur correlation {corr:.5f} too low"
    assert np.abs(got - ref_out).max() < 1e-2


def test_matches_cellpose_upsample():
    """Upsample model: output is upscaled by diam_mean/diameter AND matches cellpose pixel-wise."""
    got, ref_out = _parity("upsample_cyto3", diameter=10.)   # diam_mean 30 → 3x upscale
    assert got.shape == ref_out.shape, f"{got.shape} != cellpose {ref_out.shape}"
    assert got.shape[0] > 256 and got.shape[1] > 256, "upsample output should be larger than input"
    corr = np.corrcoef(got.ravel(), ref_out.ravel())[0, 1]
    assert corr > 0.9999, f"upsample correlation {corr:.5f} too low"
    assert np.abs(got - ref_out).max() < 1e-2
