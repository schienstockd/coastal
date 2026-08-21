"""The cumulative displacement must not re-flow what the multi-scale pass already flowed.

`compute_cumulative_displacement` sums CONSECUTIVE-frame flows, and `scale=1` of
`compute_multi_scale_optical_flow` is exactly that set over the whole movie. It computed them again
from scratch inside every centre's window, so each pair was paid for once per centre that reached it
— ~8T Farneback calls on the stock [1,2,4,8]/5 config where 4T do. These pin that the sharing
happens and that it is only a speed change.
"""

import numpy as np
import pytest

from coastal.flow import (compute_cumulative_displacement, compute_multi_scale_optical_flow,
                          consecutive_pair_flows)

SCALES, CUMWIN = [1, 2, 4, 8], 5


def _movie(t=14, n=48, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.ogrid[:n, :n]
    frames = np.stack([200 * np.exp(-(((yy - (12 + 0.9 * i)) ** 2 + (xx - (12 + 0.6 * i)) ** 2)
                                      / (2 * 5.0 ** 2))) for i in range(t)])
    return (frames + rng.normal(0, 3.0, frames.shape)).astype(np.float32)


def _counted(monkeypatch):
    import coastal.flow as flow_mod
    calls = []
    real = flow_mod.calc_flow_farneback_between_frames
    monkeypatch.setattr(flow_mod, 'calc_flow_farneback_between_frames',
                        lambda a, b: (calls.append(1), real(a, b))[1])
    return calls


def test_sharing_the_scale_1_flows_changes_nothing(monkeypatch):
    frames = _movie()
    multi = compute_multi_scale_optical_flow(frames, scales=SCALES, n_jobs=1, verbose=False)

    alone = compute_cumulative_displacement(frames, window_size=CUMWIN, n_jobs=1, verbose=False)
    shared = compute_cumulative_displacement(frames, window_size=CUMWIN, n_jobs=1, verbose=False,
                                             multi_scale_flows=multi)

    assert len(alone) == len(shared)
    for a, b in zip(alone, shared):
        assert a['center_frame'] == b['center_frame']
        np.testing.assert_array_equal(a['u'], b['u'])
        np.testing.assert_array_equal(a['v'], b['v'])


def test_sharing_removes_the_flow_calls_entirely(monkeypatch):
    frames = _movie()
    multi = compute_multi_scale_optical_flow(frames, scales=SCALES, n_jobs=1, verbose=False)

    calls = _counted(monkeypatch)
    compute_cumulative_displacement(frames, window_size=CUMWIN, n_jobs=1, verbose=False,
                                    multi_scale_flows=multi)
    assert len(calls) == 0, 'every consecutive pair was already in the multi-scale flows'

    calls.clear()
    compute_cumulative_displacement(frames, window_size=CUMWIN, n_jobs=1, verbose=False)
    assert len(calls) > 0, 'without the multi-scale flows it still has to compute them'


def test_without_scale_1_each_pair_is_still_computed_once(monkeypatch):
    """A caller whose scales exclude 1 has no consecutive flows to borrow — but the pairs are still
    shared BETWEEN centres rather than recomputed per centre."""
    frames = _movie()
    multi = compute_multi_scale_optical_flow(frames, scales=[2, 4], n_jobs=1, verbose=False)

    calls = _counted(monkeypatch)
    compute_cumulative_displacement(frames, window_size=CUMWIN, n_jobs=1, verbose=False,
                                    multi_scale_flows=multi)
    assert len(calls) == len(frames) - 1, 'one Farneback per consecutive pair, not per centre'


def test_pair_flows_are_indexed_by_their_left_frame():
    frames = _movie(t=6)
    multi = compute_multi_scale_optical_flow(frames, scales=[1], n_jobs=1, verbose=False)
    pairs = consecutive_pair_flows(frames, multi, n_jobs=1)

    assert len(pairs) == len(frames) - 1
    for i, flow in enumerate(multi[1]):
        assert flow['frame_pair'] == (i, i + 1)
        np.testing.assert_array_equal(pairs[i][0], flow['u'])
