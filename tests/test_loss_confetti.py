"""`ConfettiForegroundLoss` must build a cell-shaped foreground target from colour.

The premise being encoded: a cell is a contiguous region of ONE confetti colour, so
"one colour dominates here, brightly" is a foreground signal. Unlike `IntensityLoss` —
whose target is half a per-pixel intensity threshold and half two edge detectors — it must
be smooth across a cell interior rather than peaking on boundaries and noise, because a
speckled target is what produced 2535 prob-map components per frame (median 3 px).
"""

import numpy as np
import torch

from coastal.loss import ConfettiForegroundLoss, IntensityLoss

H = W = 48


def _variance_metrics(n_ch=3, blob=(12, 12, 20, 20)):
    """softmax_ch_* as compute_variance_metrics emits them: one colour inside the blob."""
    y, x, h, w = blob
    m = {}
    for c in range(n_ch):
        a = np.zeros((H, W), dtype=np.float32)
        if c == 0:
            a[y:y + h, x:x + w] = 1.0        # channel 0 dominates the cell
        m[f'softmax_ch_{c}'] = a
    return m


def test_target_is_blob_shaped_not_speckled():
    """The loss is minimised by a solid blob, not by a noisy map of the same mean."""
    loss = ConfettiForegroundLoss(blur_sigma=2.0)
    vm = [_variance_metrics()]

    solid = np.zeros((H, W), dtype=np.float32)
    solid[12:32, 12:32] = 1.0
    logits_blob = torch.from_numpy(solid * 8 - 4).view(1, 1, H, W)

    rng = np.random.default_rng(0)
    speckle = (rng.random((H, W)) < solid.mean()).astype(np.float32)
    logits_speckle = torch.from_numpy(speckle * 8 - 4).view(1, 1, H, W)

    assert loss(logits_blob, vm).item() < loss(logits_speckle, vm).item()


def test_prefers_predicting_the_cell_over_predicting_background():
    loss = ConfettiForegroundLoss(blur_sigma=2.0)
    vm = [_variance_metrics()]
    all_bg = torch.full((1, 1, H, W), -4.0)
    solid = np.zeros((H, W), dtype=np.float32)
    solid[12:32, 12:32] = 1.0
    on_cell = torch.from_numpy(solid * 8 - 4).view(1, 1, H, W)
    assert loss(on_cell, vm).item() < loss(all_bg, vm).item()


def test_colour_ambiguous_regions_are_not_foreground():
    """Where two colours are equally present the target must be lower than a pure cell."""
    loss = ConfettiForegroundLoss(blur_sigma=0.0)   # no blur: inspect the raw target

    pure = [_variance_metrics()]
    mixed = [{k: (v.copy() * 0.5 if k in ('softmax_ch_0', 'softmax_ch_1') else v)
              for k, v in _variance_metrics().items()}]
    mixed[0]['softmax_ch_1'] = mixed[0]['softmax_ch_0'].copy()   # two colours tie

    solid = np.zeros((H, W), dtype=np.float32)
    solid[12:32, 12:32] = 1.0
    logits = torch.from_numpy(solid * 8 - 4).view(1, 1, H, W)

    # Claiming the region is foreground is penalised more when the colour is ambiguous.
    assert loss(logits, mixed).item() > loss(logits, pure).item()


def test_no_variance_metrics_is_a_no_op():
    """Training without variance metrics must not crash or inject a spurious gradient."""
    loss = ConfettiForegroundLoss()
    out = loss(torch.zeros(1, 1, H, W, requires_grad=True), [{}])
    assert out.item() == 0.0


def test_gradients_flow():
    loss = ConfettiForegroundLoss(blur_sigma=1.0)
    logits = torch.zeros(1, 1, H, W, requires_grad=True)
    loss(logits, [_variance_metrics()]).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_intensity_loss_prefers_speckle_where_confetti_loss_does_not():
    """Pins the contrast being fixed: the old target rewards a noisy prob map.

    IntensityLoss builds its target from a per-pixel threshold plus local contrast and
    edges, so on a noisy frame a speckled prediction scores at least as well as a solid
    blob. That is the origin of the fragmented prob map.
    """
    rng = np.random.default_rng(1)
    frame = rng.random((H, W)).astype(np.float32) * 0.4
    frame[12:32, 12:32] += 0.6                       # a cell, on a noisy background
    frame_t = torch.from_numpy(frame).view(1, 1, H, W)

    solid = np.zeros((H, W), dtype=np.float32)
    solid[12:32, 12:32] = 1.0
    blob = torch.from_numpy(solid * 8 - 4).view(1, 1, H, W)

    intensity = IntensityLoss()
    speckle = (rng.random((H, W)) < 0.25).astype(np.float32)
    speckled = torch.from_numpy(speckle * 8 - 4).view(1, 1, H, W)

    # The confetti target unambiguously prefers the blob...
    conf = ConfettiForegroundLoss(blur_sigma=2.0)
    vm = [_variance_metrics()]
    assert conf(blob, vm).item() < conf(speckled, vm).item()

    # ...while the intensity target's preference is far weaker (it scores texture, not
    # objects). Recorded as an inequality on the *margin*, not a golden value.
    margin_conf = conf(speckled, vm).item() - conf(blob, vm).item()
    margin_int = intensity(speckled, frame_t).item() - intensity(blob, frame_t).item()
    assert margin_conf > margin_int
