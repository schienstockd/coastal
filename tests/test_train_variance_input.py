"""`variance_as_input` — confetti as supervision without confetti as a dead input channel.

The two were coupled: passing `variance_metrics_norm` to `train_with_metrics` both fed the confetti
losses AND concatenated `softmax_ch_*` onto the model input. Those input channels are zeros at
inference — `TwoPassSegmentationInference.predict_frame` takes no `variance_metrics` argument at all,
and `LearnedAffinityInference` zero-fills any channel the model expects beyond what it was passed —
while training sees them present 50% of the time per channel. So the coupling forced a
train/inference mismatch on anyone who wanted the supervision.

See docs/SEGMENTATION.md -> *What confetti actually contributes*.
"""
import numpy as np
import torch

from coastal.train import TemporalDatasetWithAugmentation


def _data(T=4, H=16, W=16, n_temporal=15, n_variance=3):
    rng = np.random.default_rng(0)
    frames = rng.random((T, H, W)).astype(np.float32)
    temporal = [{f"m_{i}": rng.random((H, W)).astype(np.float32) for i in range(n_temporal)}
                for _ in range(T)]
    variance = [{f"softmax_ch_{i}": rng.random((H, W)).astype(np.float32)
                 for i in range(n_variance)} for _ in range(T)]
    return frames, temporal, variance


def test_variance_as_input_false_drops_the_channels_from_the_model_input():
    frames, temporal, variance = _data()
    on = TemporalDatasetWithAugmentation(frames, temporal, variance, variance_as_input=True)
    off = TemporalDatasetWithAugmentation(frames, temporal, variance, variance_as_input=False)
    assert on[0]["frame_and_metrics"].shape[0] == 1 + 15 + 3
    assert off[0]["frame_and_metrics"].shape[0] == 1 + 15, \
        "variance channels must not reach the input when variance_as_input=False"


def test_the_losses_still_receive_the_variance_metrics():
    """The point of the flag: supervision keeps working, only the input shrinks."""
    frames, temporal, variance = _data()
    off = TemporalDatasetWithAugmentation(frames, temporal, variance, variance_as_input=False)
    item = off[0]
    assert sorted(item["variance_metrics"].keys()) == ["softmax_ch_0", "softmax_ch_1",
                                                       "softmax_ch_2"]
    for k, v in item["variance_metrics"].items():
        assert np.array_equal(v, variance[0][k]), f"{k} was altered"


def test_the_frame_and_temporal_block_is_untouched_by_the_flag():
    """Only the trailing variance block may differ, so an existing model's first 16 channels keep
    exactly the same meaning."""
    frames, temporal, variance = _data()
    on = TemporalDatasetWithAugmentation(frames, temporal, variance, variance_as_input=True)
    off = TemporalDatasetWithAugmentation(frames, temporal, variance, variance_as_input=False)
    assert torch.equal(on[0]["frame_and_metrics"][:16], off[0]["frame_and_metrics"][:16])
    assert torch.equal(on[0]["channels"], off[0]["channels"])


def test_default_is_the_historical_behaviour():
    """Not a behaviour change for anyone who does not opt in."""
    frames, temporal, variance = _data()
    d = TemporalDatasetWithAugmentation(frames, temporal, variance)
    assert d.variance_as_input is True
    assert d[0]["frame_and_metrics"].shape[0] == 1 + 15 + 3


def test_the_warp_path_honours_the_flag_too():
    """`frame_and_metrics_next` is fed to the same model, so it must have the same width — a
    mismatch here would only surface as a shape error once warp_weight was turned on."""
    frames, temporal, variance = _data()
    flow = [np.zeros((16, 16, 2), np.float32) for _ in range(len(frames))]
    off = TemporalDatasetWithAugmentation(frames, temporal, variance, flow_pairs=flow,
                                          variance_as_input=False)
    item = off[0]
    assert item["frame_and_metrics_next"] is not None
    assert item["frame_and_metrics_next"].shape == item["frame_and_metrics"].shape


def test_no_variance_metrics_at_all_is_unaffected():
    frames, temporal, _ = _data()
    for flag in (True, False):
        d = TemporalDatasetWithAugmentation(frames, temporal, None, variance_as_input=flag)
        assert d[0]["frame_and_metrics"].shape[0] == 1 + 15
        assert d[0]["variance_metrics"] == {}
