"""`flow_metrics_for_frame` must be an OPTIMISATION, not a second definition of the feature set.

The metric keys and their sorted order are a silent train/inference coupling — `predict_frame`
stacks in `sorted(key)` order and zero-fills the remainder, so a missing plane shifts every later
channel with no error (see `test_flow_metric_count.py`). A cheap per-frame path that computed
*almost* the same planes would therefore be worse than no optimisation at all.

So the load-bearing test here is equivalence: for every frame of a window, the cheap path must equal
`prepare_data_for_unet(window, ...)` at that frame, plane for plane.
"""
import numpy as np
import pytest

from coastal.flow import flow_metrics_for_frame, prepare_data_for_unet


def _window(t=17, n=48, seed=0):
    """A moving blob — flow needs actual motion, and pure noise gives degenerate fields."""
    rng = np.random.default_rng(seed)
    frames = np.zeros((t, n, n), np.float32)
    yy, xx = np.ogrid[:n, :n]
    for i in range(t):
        cy, cx = 14 + 0.9 * i, 14 + 0.6 * i
        frames[i] = 200 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 5.0 ** 2)))
    return frames + rng.normal(0, 3.0, frames.shape).astype(np.float32)


SCALES, CUMWIN = [1, 2, 4, 8], 5


@pytest.fixture(scope="module")
def reference():
    """The full pipeline, computed once — it is the expensive thing this helper exists to avoid."""
    w = _window()
    frames_norm, _, _, metrics = prepare_data_for_unet(
        w, temporal_scales=SCALES, cumulative_window=CUMWIN, verbose=False)
    return w, frames_norm, metrics


@pytest.mark.parametrize("center", [0, 1, 4, 8, 12, 16])
def test_equals_the_full_pipeline_at_every_frame(reference, center):
    """Including both truncated ends, where the flow indices clamp rather than mirror."""
    w, frames_norm, metrics = reference
    frame, got = flow_metrics_for_frame(w, center, SCALES, CUMWIN)
    want = metrics[center]

    assert sorted(got.keys()) == sorted(want.keys()), "metric SET differs — channels would shift"
    np.testing.assert_array_equal(frame, frames_norm[center])
    for k in want:
        np.testing.assert_allclose(got[k], want[k], rtol=0, atol=0,
                                   err_msg=f"plane '{k}' differs at frame {center}")


def test_yields_the_15_metric_contract(reference):
    """Same count the training defaults produce (test_flow_metric_count.py)."""
    w, _, _ = reference
    _, m = flow_metrics_for_frame(w, 8, SCALES, CUMWIN)
    assert len(m) == 15, sorted(m.keys())
    assert "mag_8" in m


def test_computes_only_the_flows_the_frame_reads(monkeypatch):
    """The point of the helper: 7 Farneback calls where the full pipeline needs 53.

    Counted by patching the module-level flow function, which only works because this path is
    SERIAL — `prepare_data_for_unet` fans out over joblib, so its own calls cannot be counted the
    same way. Its 53 is arithmetic instead: sum(N - s for s in scales) + N cumulative centres, each
    summing up to cumulative_window-1 pairs.
    """
    import coastal.flow as flow_mod

    calls = []
    call_pairs = []
    real = flow_mod.calc_flow_farneback_between_frames
    window = _window()

    def counting(a, b):
        calls.append(1)
        # identify the pair by which window rows these arrays are
        idx = [next(i for i in range(len(window)) if np.array_equal(window[i], f))
               for f in (a, b)]
        call_pairs.append(tuple(idx))
        return real(a, b)

    monkeypatch.setattr(flow_mod, "calc_flow_farneback_between_frames", counting)
    flow_metrics_for_frame(window, 8, SCALES, CUMWIN)

    # 4 scales x 1 flow each + (cumulative_window - 1) consecutive pairs, MINUS the one pair the
    # two sets share: scale 1 reads (center-1, center), which the cumulative window also covers.
    # That overlap is not incidental to this config — it holds whenever 1 is among the scales and
    # the cumulative window reaches the centre, i.e. every stock setup — and it used to be flowed
    # twice per plane.
    shared = 1 if (1 in SCALES and CUMWIN > 1) else 0
    assert len(calls) == len(SCALES) + (CUMWIN - 1) - shared

    pairs = {tuple(sorted(c)) for c in call_pairs}
    assert len(pairs) == len(calls), "a frame pair was flowed more than once"

    full_pipeline = sum(17 - s for s in SCALES) + 17 * (CUMWIN - 1)
    assert len(calls) < full_pipeline / 5, (
        f"{len(calls)} vs {full_pipeline} — the saving is the reason this exists")


def test_short_window_drops_a_scale_rather_than_raising():
    """Documented failure mode, asserted so nobody 'fixes' it into a silent pad.

    A window shorter than the largest scale yields no flow for it, and the plane vanishes — which
    shifts every later channel. The caller must supply a long enough window; `CoastalUtils` derives
    its temporal radius from the model's scales for exactly this reason.
    """
    _, m = flow_metrics_for_frame(_window(t=5), 2, SCALES, CUMWIN)
    assert "mag_8" not in m, "scale 8 needs 9 frames; 5 cannot produce it"
    assert "mag_4" in m, "scale 4 still fits in 5 frames — the cutoff is per scale, not per window"
    assert len(m) == 14


def test_value_range_makes_two_windows_of_one_movie_photometrically_consistent():
    """The reason tiled inference needs the override.

    Two windows of the same movie have different min/max, so with per-window scaling the SAME frame
    comes out with different contrast depending on which window it was read in — and the structure
    tensor reads the scaled frame directly. Pinning the range to the movie's makes the frame
    identical, which is what training saw.
    """
    movie = _window(t=25)
    lo, hi = float(movie.min()), float(movie.max())

    early, _ = flow_metrics_for_frame(movie[4:21], 8, SCALES, CUMWIN, value_range=(lo, hi))
    late, _ = flow_metrics_for_frame(movie[8:25], 4, SCALES, CUMWIN, value_range=(lo, hi))
    np.testing.assert_allclose(early, late, rtol=0, atol=0)   # both are movie frame 12

    drift_a, _ = flow_metrics_for_frame(movie[4:21], 8, SCALES, CUMWIN)
    drift_b, _ = flow_metrics_for_frame(movie[8:25], 4, SCALES, CUMWIN)
    assert not np.allclose(drift_a, drift_b), \
        "sanity: without the override the same frame must scale differently per window"


def test_center_outside_the_window_is_an_error():
    with pytest.raises(IndexError):
        flow_metrics_for_frame(_window(t=6), 6, SCALES, CUMWIN)


def test_flow_cache_is_reused_across_overlapping_windows(monkeypatch):
    """The cross-call saving: stepping t re-flows only the pairs it has not seen.

    Tiled inference centres a window on each timepoint, so consecutive windows overlap by all but
    one frame and the cumulative sum re-reads the same consecutive pairs. With a shared cache the
    second timepoint pays for the scale flows (whose pairs move with t) and the ONE new consecutive
    pair, not for all seven.
    """
    import coastal.flow as flow_mod

    movie = _window(t=20)
    calls = []
    real = flow_mod.calc_flow_farneback_between_frames
    monkeypatch.setattr(flow_mod, "calc_flow_farneback_between_frames",
                        lambda a, b: (calls.append(1), real(a, b))[1])

    cache = {}
    # window centred on t=9 then on t=10, offsets chosen so the absolute keys line up
    flow_metrics_for_frame(movie[1:18], 8, SCALES, CUMWIN, flow_cache=cache, window_offset=1)
    first = len(calls)
    flow_metrics_for_frame(movie[2:19], 8, SCALES, CUMWIN, flow_cache=cache, window_offset=2)
    second = len(calls) - first

    assert first == 7
    assert second == len(SCALES), (
        f"{second} flows on the second window; only the {len(SCALES)} scale pairs move with t")


def test_flow_cache_does_not_change_the_metrics():
    """A cache is a speed change or it is a bug. Same planes, cached or not."""
    movie = _window(t=20)
    plain = flow_metrics_for_frame(movie[1:18], 8, SCALES, CUMWIN)
    cache = {}
    flow_metrics_for_frame(movie[0:17], 8, SCALES, CUMWIN, flow_cache=cache, window_offset=0)
    cached = flow_metrics_for_frame(movie[1:18], 8, SCALES, CUMWIN,
                                    flow_cache=cache, window_offset=1)

    assert np.array_equal(plain[0], cached[0])
    assert sorted(plain[1]) == sorted(cached[1])
    for k in plain[1]:
        np.testing.assert_array_equal(plain[1][k], cached[1][k], err_msg=f"metric {k} differs")
