"""The flow-metric count is a silent train/inference contract.

`predict_frame` stacks metrics in `sorted(key)` order and zero-fills whatever the model expects
beyond what it was handed. So a metric that is present at training and absent at inference does NOT
leave a hole at its own channel — every metric sorting after it shifts down one slot, and the
zero-fill lands at the END. Shapes still match, nothing raises, and the model is fed misaligned
inputs.

Two ways to trip it, both real:
  * `Inference3D.predict_temporal_volume` defaults to `temporal_scales=[1,2,4], cumulative_window=2`
    while `prepare_data_for_unet` (training) defaults to `[1,2,4,8], cumulative_window=5`;
  * fewer than 9 frames drops `mag_8` regardless of the scales requested.

See docs/SEGMENTATION.md -> *Memory: the 4D path*.
"""
import numpy as np
import pytest

from coastal.flow import prepare_data_for_unet


def _frames(t, n=64, seed=0):
    return (np.random.default_rng(seed).random((t, n, n)) * 255).astype(np.uint8)


def _keys(t, scales, cw):
    _, _, _, tm = prepare_data_for_unet(_frames(t), temporal_scales=scales,
                                        cumulative_window=cw, verbose=False)
    return sorted(tm[0].keys())


def test_training_defaults_yield_15_metrics():
    keys = _keys(31, [1, 2, 4, 8], 5)
    assert len(keys) == 15, keys
    assert "mag_8" in keys


def test_the_4d_entry_point_defaults_yield_one_fewer():
    """Not a hypothetical — these are `predict_temporal_volume`'s actual default arguments."""
    keys = _keys(31, [1, 2, 4], 2)
    assert len(keys) == 14, keys
    assert "mag_8" not in keys


def test_the_missing_metric_shifts_later_channels_rather_than_leaving_a_hole():
    """THE failure mode. Because the stack is sorted, dropping mag_8 moves everything after it."""
    full = _keys(31, [1, 2, 4, 8], 5)
    short = _keys(31, [1, 2, 4], 2)
    i = full.index("mag_8")
    # everything before mag_8 keeps its channel index
    assert full[:i] == short[:i]
    # everything after it is shifted down by one — so those channels now mean something else
    assert short[i:] == full[i + 1:]
    assert short[i] != full[i], (
        f"channel {i} means '{full[i]}' at training but '{short[i]}' at inference")


def test_a_short_sequence_silently_drops_mag_8_even_when_requested():
    """A quick test run with few frames does not produce the channels training used."""
    keys = _keys(8, [1, 2, 4, 8], 5)
    assert "mag_8" not in keys, "expected mag_8 to be unavailable below 9 frames"
    assert len(keys) == 14


def test_asking_for_the_training_scales_at_full_length_is_reproducible():
    """The fix is to pass them explicitly; check that doing so is stable."""
    assert _keys(31, [1, 2, 4, 8], 5) == _keys(31, [1, 2, 4, 8], 5)
    assert len(_keys(20, [1, 2, 4, 8], 5)) == 15
