"""`TemporalMetrics` must be a bit-identical, lazy stand-in for the eager metric list.

The 4D segmentation path holds 14 float32 metric planes per frame (3.1 GB at T=180,
531×586) purely because `prepare_data_for_unet` used to compute all T up front, while
`predict_sequence` reads one frame at a time. These tests pin the two properties that
make streaming them safe:

  * **same values** — the per-frame normalisation reproduces the old whole-stack float32
    copy bit-for-bit, and the cached `frame_range` (global by design) does not change it.
  * **actually lazy** — no metric plane is computed until indexed, and nothing is cached
    (caching all T would reinstate the memory it exists to avoid).
"""

import numpy as np
import pytest

from coastal.flow import (TemporalMetrics,
                          compute_multi_scale_optical_flow, compute_cumulative_displacement,
                          extract_temporal_metrics, prepare_data_for_unet,
                          normalize_and_project)

T, H, W = 7, 48, 56


@pytest.fixture(scope="module")
def flow_inputs():
    """A moving blob, so the flow fields are non-degenerate."""
    rng = np.random.default_rng(7)
    frames = (rng.random((T, H, W)) * 40).astype(np.uint8)
    for t in range(T):
        frames[t, 10 + t:20 + t, 12 + 2 * t:24 + 2 * t] = 230
    ms = compute_multi_scale_optical_flow(frames, scales=[1, 2, 4], verbose=False)
    cum = compute_cumulative_displacement(frames, window_size=2, verbose=False)
    # prepare_data_for_unet normalises before computing metrics; mirror that here.
    f = frames.astype(np.float32)
    f = (f - f.min()) / (f.max() - f.min() + 1e-5)
    return f, ms, cum


def test_frame_normalisation_matches_the_old_whole_stack_copy(flow_inputs):
    """The numerical heart of the change.

    `extract_temporal_metrics` used to build `np.array(frames, dtype=np.float32)` and
    normalise the *entire* stack on every call, just to read one frame out of it — O(T)
    full-stack copies per z-slice. It now scales the single frame it needs. Same scaling
    (global min/max), so the result must be bit-identical.
    """
    frames, _, _ = flow_inputs

    legacy = np.array(frames, dtype=np.float32)
    legacy = (legacy - legacy.min()) / (legacy.max() - legacy.min() + 1e-5)

    fa = np.asarray(frames)
    lo, hi = fa.min(), fa.max()
    for t in range(T):
        current = (fa[t].astype(np.float32) - lo) / (hi - lo + 1e-5)
        np.testing.assert_array_equal(legacy[t], current, err_msg=f'frame {t}')


def test_lazy_matches_a_materialised_list_exactly(flow_inputs):
    frames, ms, cum = flow_inputs
    lazy = TemporalMetrics(frames, ms, cum)
    eager = list(lazy)          # the one way to materialise (what training does)

    assert len(lazy) == len(eager) == T
    for t in range(T):
        assert lazy[t].keys() == eager[t].keys()
        assert len(eager[t]) == 14
        for key in eager[t]:
            np.testing.assert_array_equal(lazy[t][key], eager[t][key], err_msg=f't={t} {key}')
            assert lazy[t][key].dtype == np.float32


def test_cached_frame_range_matches_a_full_stack_scan(flow_inputs):
    """The cached (min, max) must reproduce the un-cached global normalisation."""
    frames, ms, cum = flow_inputs
    for t in (0, T // 2, T - 1):
        without = extract_temporal_metrics(frames, ms, cum, t)
        with_range = extract_temporal_metrics(frames, ms, cum, t,
                                              frame_range=(frames.min(), frames.max()))
        for key in without:
            np.testing.assert_array_equal(without[key], with_range[key], err_msg=f't={t} {key}')


def test_is_lazy_and_uncached(flow_inputs):
    """Indexing computes on demand; nothing accumulates across accesses."""
    frames, ms, cum = flow_inputs
    lazy = TemporalMetrics(frames, ms, cum)

    calls = []
    import coastal.flow as flow_mod
    real = flow_mod.extract_temporal_metrics

    def counting(*a, **kw):
        calls.append(a[3])
        return real(*a, **kw)

    flow_mod.extract_temporal_metrics = counting
    try:
        assert calls == []            # construction computes nothing
        lazy[2]
        assert calls == [2]           # only the frame asked for
        lazy[2]
        assert calls == [2, 2]        # recomputed, not cached — that is the point
        list(lazy)
        assert sorted(calls[2:]) == list(range(T))
    finally:
        flow_mod.extract_temporal_metrics = real


def test_sequence_protocol(flow_inputs):
    frames, ms, cum = flow_inputs
    lazy = TemporalMetrics(frames, ms, cum)

    assert len(list(lazy)) == T                      # iteration
    assert len(lazy[1:4]) == 3                       # slicing -> list of dicts
    assert lazy[-1].keys() == lazy[T - 1].keys()     # negative indexing
    np.testing.assert_array_equal(lazy[-1]['mag_1'], lazy[T - 1]['mag_1'])
    with pytest.raises(IndexError):
        lazy[T]
    with pytest.raises(IndexError):
        lazy[-T - 1]


def test_prepare_data_for_unet_returns_lazy_metrics(flow_inputs):
    """The pipeline entry point hands back the lazy sequence, still 14 metrics/frame."""
    frames, _, _ = flow_inputs
    frames_u8 = (frames * 255).astype(np.uint8)
    _, _, _, metrics = prepare_data_for_unet(
        frames_u8, temporal_scales=[1, 2, 4], cumulative_window=2, verbose=False)

    assert isinstance(metrics, TemporalMetrics)
    assert len(metrics) == T
    assert len(metrics[0]) == 14
    assert all(v.shape == (H, W) for v in metrics[0].values())


def test_normalize_and_project_channel_selection_order():
    """Selecting channels before the float32 cast must not change the output."""
    rng = np.random.default_rng(11)
    seq = (rng.random((5, 4, 24, 26)) * 1000).astype(np.uint16)
    ch = [1, 2, 3]

    multi, proj = normalize_and_project(seq, ch)
    # Reference: pre-select the channels ourselves, so ordering cannot matter.
    multi_ref, proj_ref = normalize_and_project(seq[:, ch], None)

    np.testing.assert_array_equal(multi, multi_ref)
    np.testing.assert_array_equal(proj, proj_ref)
    assert multi.shape == (5, 3, 24, 26) and proj.shape == (5, 24, 26)
