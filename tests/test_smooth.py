"""`coastal.smooth` — the model-free smoothing primitives.

The invariants here are not stylistic. Each one was established by measurement on photon-limited
resonance data (`zolIMa/fXgbTl`), and each has a failure mode that is silent:

* **one shared kernel across channels** — the consumer is a cross-channel ratio, and a per-channel
  transform corrupts it without erroring (this is what disqualified the learned net);
* **spatial before temporal** — reversed, a temporal median on single-digit photon counts sees only
  zeros and keeps LESS signal than no smoothing at all;
* **a centred odd window** — the R predecessor's `slice(i-sw, i+sw)` was half-open, so its window
  was `2*sw` frames and off-centre, and at `sw=1` its "median" of 2 samples was a mean;
* **no spatial median** — a median rejects sparse outliers, and here the signal IS sparse counts.

Record: `cecelia-feijoa/docs/todo/SMOOTHING_PLAN.md`.
"""
import numpy as np
import pytest

from coastal.smooth import (
    spatial_smooth, temporal_smooth, smooth_channels,
    gaussian_restorer, temporal_mean_restorer, temporal_median_restorer,
)


def _movie(T=9, C=3, Y=16, X=18, seed=0):
    """A (T,C,Y,X) stack with per-channel scale differences, like real multi-channel data."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 4, size=(T, C, Y, X)).astype(np.float32)
    for c in range(C):
        a[:, c, 4:9, 5:11] += 40.0 * (c + 1)          # channel-dependent brightness
    return a


# ── spatial ───────────────────────────────────────────────────────────────────────────────────

def test_spatial_smooth_only_touches_the_spatial_axes():
    a = _movie()
    out = spatial_smooth(a, sigma=1.0)
    assert out.shape == a.shape and out.dtype == np.float32
    # a constant-in-time input must stay constant in time: no leakage across T or C
    flat = np.repeat(a[:1], a.shape[0], axis=0)
    fout = spatial_smooth(flat, sigma=1.0)
    assert np.allclose(fout, fout[0][None], atol=1e-5)


def test_spatial_smooth_zero_sigma_is_a_passthrough():
    a = _movie()
    assert np.array_equal(spatial_smooth(a, sigma=0), a.astype(np.float32))
    assert np.array_equal(spatial_smooth(a, sigma=None), a.astype(np.float32))


# ── temporal ──────────────────────────────────────────────────────────────────────────────────

def test_temporal_window_is_centred_and_odd():
    """The R predecessor's off-by-one: an even width must not silently shift the window."""
    a = np.zeros((9, 4, 4), np.float32)
    a[4] = 100.0                                       # one bright frame, dead centre
    out = temporal_smooth(a, frames=3, stat="mean", time_axis=0)
    # a centred 3-window spreads it symmetrically over 3,4,5 — and NOWHERE else
    assert out[3, 0, 0] == pytest.approx(out[5, 0, 0]), "window is off-centre"
    assert out[4, 0, 0] > 0 and out[2, 0, 0] == 0 and out[6, 0, 0] == 0
    # an even width is promoted to odd rather than shifting the centre
    assert np.allclose(temporal_smooth(a, 4, "mean", 0), temporal_smooth(a, 5, "mean", 0))


def test_temporal_median_rejects_a_transient_where_the_mean_averages_it_in():
    """The whole reason median is the default: a cell that moved through one frame."""
    a = np.zeros((9, 4, 4), np.float32)
    a[4] = 100.0
    mean3 = temporal_smooth(a, 3, "mean", 0)
    med3 = temporal_smooth(a, 3, "median", 0)
    assert mean3[4, 0, 0] == pytest.approx(100.0 / 3)   # smeared in at partial weight
    assert med3[4, 0, 0] == 0.0                         # rejected as transient
    # ...and a PERSISTENT feature survives the median untouched
    b = np.zeros((9, 4, 4), np.float32)
    b[3:7] = 100.0
    assert temporal_smooth(b, 3, "median", 0)[4, 0, 0] == pytest.approx(100.0)


def test_temporal_smooth_noops_where_there_is_no_time():
    a = _movie(T=1)
    assert np.array_equal(temporal_smooth(a, 3, "median", 0), a.astype(np.float32))
    b = _movie()
    assert np.array_equal(temporal_smooth(b, 1, "median", 0), b.astype(np.float32))
    assert np.array_equal(temporal_smooth(b, 0, "median", 0), b.astype(np.float32))


def test_temporal_smooth_rejects_an_unknown_stat():
    with pytest.raises(ValueError, match="median.*mean|mean.*median"):
        temporal_smooth(_movie(), 3, "mode", 0)


# ── the composite, and its invariants ─────────────────────────────────────────────────────────

def test_every_channel_gets_the_SAME_kernel():
    """THE invariant. A per-channel kernel would corrupt the cross-channel ratio its consumers
    read, silently. Verified by smoothing channels jointly vs one at a time."""
    a = _movie()
    joint = smooth_channels(a, sigma=1.0, frames=3, channel_axis=1, time_axis=0)
    for c in range(a.shape[1]):
        alone = smooth_channels(a, sigma=1.0, frames=3, channel_axis=1, time_axis=0, channels=[c])
        assert np.allclose(joint[:, c], alone[:, c], atol=1e-5), f"channel {c} treated differently"


def test_a_uniform_channel_scaling_passes_straight_through():
    """A shared linear-then-rank pipeline commutes with a global per-channel SCALE, which is the
    formal statement of 'the kernel does not care which channel it is looking at'."""
    a = _movie()
    b = a.copy(); b[:, 1] *= 7.0
    sa = smooth_channels(a, sigma=1.0, frames=3, channel_axis=1, time_axis=0)
    sb = smooth_channels(b, sigma=1.0, frames=3, channel_axis=1, time_axis=0)
    assert np.allclose(sb[:, 1], sa[:, 1] * 7.0, rtol=1e-4, atol=1e-3)


def test_unselected_channels_are_untouched():
    a = _movie()
    out = smooth_channels(a, sigma=1.0, frames=3, channel_axis=1, time_axis=0, channels=[0])
    assert np.array_equal(out[:, 1], a[:, 1])
    assert not np.array_equal(out[:, 0], a[:, 0])


def test_time_axis_is_opt_in_so_a_stack_is_never_blurred_across_z():
    """`time_axis=None` must disable the temporal term. Guessing it wrong smooths across Z and
    silently flattens a stack into a slab."""
    a = _movie()
    no_t = smooth_channels(a, sigma=1.0, frames=3, channel_axis=1, time_axis=None)
    spatial_only = smooth_channels(a, sigma=1.0, frames=0, channel_axis=1, time_axis=0)
    assert np.allclose(no_t, spatial_only, atol=1e-5)


def test_channel_and_time_axis_cannot_collide():
    with pytest.raises(ValueError, match="cannot be the same axis"):
        smooth_channels(_movie(), channel_axis=1, time_axis=1, frames=3)


def test_order_is_spatial_then_temporal():
    """Not commutative, and the wrong order is the documented failure on sparse data."""
    a = _movie()
    got = smooth_channels(a, sigma=1.0, frames=3, stat="median", channel_axis=1, time_axis=0)
    want = temporal_smooth(spatial_smooth(a, 1.0), 3, "median", time_axis=0)
    assert np.allclose(got, want, atol=1e-5)
    reversed_order = spatial_smooth(temporal_smooth(a, 3, "median", 0), 1.0)
    assert not np.allclose(got, reversed_order, atol=1e-3)


def test_smoothing_a_5d_TCZYX_volume_keeps_z_intact():
    a = np.zeros((5, 2, 4, 12, 12), np.float32)
    a[:, :, 2] = 50.0                                   # one bright z-plane
    out = smooth_channels(a, sigma=1.0, frames=3, channel_axis=1, time_axis=0)
    assert out.shape == a.shape
    assert out[0, 0, 1].max() == 0.0, "leaked across Z"
    assert out[0, 0, 2].max() > 0.0


# ── restorers ─────────────────────────────────────────────────────────────────────────────────

def test_restorers_agree_with_the_primitives_they_wrap():
    proj = _movie()[:, 0]                               # (T,Y,X)
    assert np.allclose(gaussian_restorer(1.0)(proj), spatial_smooth(proj, 1.0), atol=1e-6)
    assert np.allclose(temporal_mean_restorer(3)(proj),
                       temporal_smooth(proj, 3, "mean", 0), atol=1e-6)
    assert np.allclose(temporal_median_restorer(3)(proj),
                       temporal_smooth(proj, 3, "median", 0), atol=1e-6)


def test_temporal_restorers_refuse_a_single_plane():
    plane = np.zeros((8, 8), np.float32)
    for r in (temporal_mean_restorer(3), temporal_median_restorer(3)):
        with pytest.raises(ValueError, match="time axis"):
            r(plane)


def test_denoise_still_re_exports_the_restorers():
    """Moving them must not break `from coastal.denoise import gaussian_restorer`."""
    from coastal import denoise
    assert denoise.gaussian_restorer is gaussian_restorer
    assert denoise.temporal_mean_restorer is temporal_mean_restorer
    assert denoise.temporal_median_restorer is temporal_median_restorer


# ── the property that motivates the module ────────────────────────────────────────────────────

def test_smoothing_makes_a_background_findable_on_photon_limited_input():
    """The measured reason this exists: on sparse counts a triangle threshold lands inside the
    signal. Smoothing gives the histogram a background population to find."""
    from cecelia.utils import intensity_utils as iu
    rng = np.random.default_rng(3)
    # Photon counts TIMES A DIGITISER GAIN — the gain matters: without it the smoothed values sit
    # below 1 and an integer-binned histogram collapses them all into bin 0, which is an artefact of
    # the fixture rather than of the data. Real `fXgbTl`: 86-95% zeros per channel, max 522, and the
    # cell interior is itself sparse (p99 = 35), which is why raw fails here too.
    gain = 18
    a = (rng.poisson(0.09, size=(9, 1, 6, 64, 64)) * gain).astype(np.float32)
    a[:, 0, :, 20:34, 20:34] += rng.poisson(2.0, size=(9, 6, 14, 14)) * gain
    assert (a == 0).mean() > 0.8, "fixture must be photon-limited to be a fair test"
    sm = smooth_channels(a, sigma=1.0, frames=3, channel_axis=1, time_axis=0)

    def kept(v):
        h = np.bincount(np.clip(np.rint(v.ravel()), 0, 65535).astype(np.int64), minlength=65536)
        bg = float(iu.background_threshold(h.astype(float), "triangle"))
        return 100.0 * (v[:, 0, :, 22:32, 22:32] > bg).mean()

    raw_kept, sm_kept = kept(a), kept(sm)
    # measured on this fixture: 63.0% -> 100.0%, background threshold 19 -> 6
    assert sm_kept > raw_kept + 20.0, f"smoothing must recover signal: {raw_kept:.1f}% -> {sm_kept:.1f}%"
