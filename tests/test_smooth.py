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
    spatial_smooth, temporal_smooth, smooth_channels, temporal_gated, gated_frame,
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
    signal, so background subtraction discards most of the cell. Smoothing gives the histogram a
    background population to find.

    Uses skimage's `threshold_triangle` rather than cecelia's `intensity_utils.background_threshold`:
    same algorithm (Zack 1977), but scikit-image is a coastal RUNTIME dep while cecelia is not —
    `coastal/` stays array-only, and cecelia is an optional extra for the notebook/napari glue only.
    Importing it here passed locally (editable install) and failed CI, which is the point of the
    boundary.

    Thresholding the NONZERO voxels is what makes this reproduce cecelia's behaviour, and it is the
    mechanism rather than a detail: `background_threshold(..., ignore_zero=True)` drops the zero bin,
    because otherwise the histogram's peak IS the zero spike and triangle returns a threshold near 0
    (measured on this fixture: 88% of cell signal "kept", i.e. the test passes vacuously). With zeros
    excluded the peak sits in the real background population, the threshold rises into the signal,
    and the failure this module exists to fix appears.
    """
    from skimage.filters import threshold_triangle
    rng = np.random.default_rng(3)
    # Photon counts TIMES A DIGITISER GAIN — the gain matters: without it the smoothed values sit
    # below 1 and any integer-binned histogram collapses them into one bin, which is an artefact of
    # the fixture rather than of the data. Real `fXgbTl`: 86-95% zeros per channel, max 522, and the
    # cell interior is itself sparse (p99 = 35), which is why raw fails here too.
    gain = 18
    a = (rng.poisson(0.09, size=(9, 1, 6, 64, 64)) * gain).astype(np.float32)
    a[:, 0, :, 20:34, 20:34] += rng.poisson(2.0, size=(9, 6, 14, 14)) * gain
    assert (a == 0).mean() > 0.8, "fixture must be photon-limited to be a fair test"
    sm = smooth_channels(a, sigma=1.0, frames=3, channel_axis=1, time_axis=0)

    def kept(v):
        bg = float(threshold_triangle(v[v > 0]))       # == ignore_zero=True
        return 100.0 * (v[:, 0, :, 22:32, 22:32] > bg).mean(), bg

    raw_kept, raw_bg = kept(a)
    sm_kept, sm_bg = kept(sm)
    # measured on this fixture: 63.0% (bg 18.8) -> 100.0% (bg 5.4)
    assert sm_kept > raw_kept + 20.0, (
        f"smoothing must recover signal: {raw_kept:.1f}% (bg {raw_bg:.1f}) -> "
        f"{sm_kept:.1f}% (bg {sm_bg:.1f})")


# ── gated temporal averaging ──────────────────────────────────────────────────────────────────
#
# The property that distinguishes it from a median is a GUARANTEE, not a tuning: where nothing
# matches, the output is the input. Measured on a 30 s intravital movie, `median(9)` keeps 69% of
# punctum amplitude and 77% of moving-cell sharpness while `gated(9)` keeps 100% and 100% at the same
# noise reduction — but the tests below pin the mechanism, because the measurement cannot run in CI.

def _moving_dot(T=7, Y=24, X=24, step=1, amp=60.0, noise=2.0, seed=1):
    """A bright dot translating one px per frame over a noisy background."""
    rng = np.random.default_rng(seed)
    a = rng.normal(10, noise, size=(T, Y, X)).astype(np.float32)
    for t in range(T):
        a[t, 8 + t * step, 8 + t * step] += amp
    return a


def test_gated_preserves_a_moving_feature_that_a_median_destroys():
    a = _moving_dot()
    t = len(a) // 2
    y, x = 8 + t, 8 + t
    def contrast(stack):
        return float(stack[t, y, x] - np.median(stack[t]))
    raw, med, gat = contrast(a), contrast(temporal_smooth(a, 5, 'median')), contrast(temporal_gated(a, 5))
    assert med < 0.6 * raw, f'median unexpectedly preserved the moving dot ({med:.1f} vs {raw:.1f})'
    assert gat > 0.9 * raw, f'gated lost the moving dot ({gat:.1f} vs {raw:.1f})'


def test_gated_still_removes_noise_where_nothing_moves():
    rng = np.random.default_rng(2)
    static = np.zeros((9, 20, 20), np.float32) + 30.0
    static[:, 6:14, 6:14] = 80.0
    noisy = static + rng.normal(0, 4.0, static.shape).astype(np.float32)
    out = temporal_gated(noisy, 9)
    # it must actually average — a filter that only ever returned its input would pass the test above
    assert np.std(out - static) < 0.75 * np.std(noisy - static)


def test_gated_never_blurs_more_than_doing_nothing():
    """The guarantee: with nothing matchable anywhere, the output is the input, not a smear."""
    rng = np.random.default_rng(3)
    a = rng.normal(0, 50, size=(5, 16, 16)).astype(np.float32)   # independent frames, no structure
    out = temporal_gated(a, 5, sigma=0.05)                        # a gate this tight can match nothing
    np.testing.assert_allclose(out, a, rtol=1e-4, atol=1e-3)


def test_gated_noops_like_the_other_stats():
    a = _moving_dot()
    np.testing.assert_array_equal(temporal_gated(a, 1), a)
    np.testing.assert_array_equal(temporal_gated(a[:1], 5), a[:1])


def test_gated_is_reachable_through_both_entry_points():
    a = _moving_dot()
    np.testing.assert_allclose(temporal_smooth(a, 5, 'gated'), temporal_gated(a, 5), rtol=1e-5)


def test_gated_does_not_average_across_z():
    """A (T,Z,Y,X) stack is filtered plane by plane — the same opt-in rule the other stats follow."""
    rng = np.random.default_rng(4)
    a = rng.normal(10, 1.0, size=(5, 3, 16, 16)).astype(np.float32)
    a[:, 1] += 100.0                                   # one z-plane far brighter than its neighbours
    out = temporal_gated(a, 5)
    assert out[:, 1].mean() > out[:, 0].mean() + 90    # the offset survives: no bleed across z


def test_every_channel_gets_the_SAME_gate():
    """The AF ratio invariant, for the adaptive kernel.

    Gating each channel on its own content would decide differently at one voxel. Feeding channels
    that are exact scalar multiples of each other makes that detectable: identical weights keep the
    ratio EXACTLY, any per-channel decision does not.
    """
    rng = np.random.default_rng(5)
    base = rng.normal(20, 3, size=(7, 18, 18)).astype(np.float32)
    base[:, 5:12, 5:12] += 50
    a = np.stack([base, base * 3.0, base * 0.25], axis=1)          # (T,C,Y,X), exact multiples
    out = smooth_channels(a, sigma=0.0, frames=5, stat='gated', channel_axis=1, time_axis=0)
    np.testing.assert_allclose(out[:, 1], out[:, 0] * 3.0, rtol=1e-4)
    np.testing.assert_allclose(out[:, 2], out[:, 0] * 0.25, rtol=1e-4)


def test_gated_channels_share_one_guide_so_a_dim_channel_follows_the_bright_one():
    """A dim channel must inherit the match found in the total signal, not gate on its own noise."""
    rng = np.random.default_rng(6)
    bright = rng.normal(10, 1, size=(7, 20, 20)).astype(np.float32)
    for t in range(7):
        bright[t, 6 + t, 6 + t] += 80                              # a clear moving feature
    dim = rng.normal(2, 1, size=(7, 20, 20)).astype(np.float32)     # almost pure noise
    a = np.stack([bright, dim], axis=1)
    shared = smooth_channels(a, sigma=0.0, frames=5, stat='gated', channel_axis=1, time_axis=0)
    alone = temporal_gated(dim, 5)
    # gated on the total signal, the dim channel is averaged more than it would gate for itself
    assert np.std(shared[:, 1]) < np.std(alone)


def test_smooth_channels_rejects_an_unknown_stat_including_near_misses():
    a = _movie()
    with pytest.raises(ValueError):
        temporal_smooth(a, 3, 'gate')


def test_streaming_and_series_forms_agree_given_one_sigma():
    """`gated_frame` exists so a streaming caller does not compute W outputs to keep one. It must
    produce exactly what the series form produces for that frame, or the task and the library
    silently diverge."""
    rng = np.random.default_rng(7)
    a = rng.normal(20, 3, size=(7, 20, 20)).astype(np.float32)
    a[:, 8, 8] += 50
    series = temporal_gated(a, 5, sigma=2.0)
    np.testing.assert_allclose(series[3], gated_frame(a[1:6], sigma=2.0), rtol=1e-4, atol=1e-3)


def test_an_unshared_sigma_is_what_makes_the_two_forms_differ():
    """Pins the REASON, so nobody 'fixes' the agreement test by loosening its tolerance: estimated
    per-window, sigma is a small sample and the gate strictness drifts across a movie."""
    rng = np.random.default_rng(7)
    a = rng.normal(20, 3, size=(7, 20, 20)).astype(np.float32)
    a[:, 8, 8] += 50
    from coastal.smooth import noise_sigma
    assert noise_sigma(a) != noise_sigma(a[1:6])          # different samples, different estimate
    assert not np.allclose(temporal_gated(a, 5)[3], gated_frame(a[1:6]), rtol=1e-6, atol=1e-6)
