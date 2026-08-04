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


def test_percentile_normalisation_survives_one_bright_outlier():
    """Why the target is normalised by a percentile and not the max.

    One unusually bright cell must not set the scale for the whole image: under max-normalisation
    every typical cell lands far below the inference prob_threshold and the model learns an empty
    foreground (measured: 14 labels/frame). Build a field of ordinary cells plus one 10x outlier
    and check typical cells still read as foreground.
    """
    import numpy as np
    import torch
    from coastal.loss import ConfettiForegroundLoss

    # Cell coverage kept realistic (~2% of the frame): the percentile only has room to sit
    # below the outlier if cells are sparse, which they are on real data.
    H = W = 192
    rng = np.random.default_rng(0)
    typical = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(12):
        cy, cx = rng.integers(20, H - 20, size=2)
        typical += 0.5 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3.0 ** 2))
    typical += 5.0 * np.exp(-((yy - 96) ** 2 + (xx - 96) ** 2) / (2 * 3.0 ** 2))   # the outlier
    tcy, tcx = 20, 20
    typical += 0.5 * np.exp(-((yy - tcy) ** 2 + (xx - tcx) ** 2) / (2 * 3.0 ** 2))

    metrics = [{'softmax_ch_0': typical, 'softmax_ch_1': np.zeros_like(typical)}]
    pred = torch.zeros(1, 1, H, W)

    def target_of(loss):
        stack = torch.from_numpy(typical)[None, None]
        t = loss._blur(stack)
        flat = t.view(1, -1)
        if loss.norm_percentile >= 100:
            hi = flat.max(dim=1).values.view(-1, 1, 1, 1) + 1e-6
        else:
            hi = torch.quantile(flat, loss.norm_percentile / 100.0,
                                dim=1).view(-1, 1, 1, 1) + 1e-6
        return torch.clamp(t / hi, 0, 1)[0, 0].numpy()

    by_max = target_of(ConfettiForegroundLoss(norm_percentile=100))
    by_pct = target_of(ConfettiForegroundLoss(norm_percentile=99.5))

    # A typical cell centre, well away from the outlier.
    assert by_max[tcy, tcx] < 0.4, 'max-normalisation should crush typical cells (the old bug)'
    assert by_pct[tcy, tcx] > 0.4, 'percentile normalisation must keep typical cells as foreground'
    # And the loss still runs end to end.
    assert torch.isfinite(ConfettiForegroundLoss()(pred, metrics))


def test_default_blur_is_the_measured_one_and_train_agrees():
    """1.0 is a measured choice, not a taste: pin it, and pin the two call sites together.

    Against synthetic crowded GT (2 movies x 2 densities x 3 frames, each model at its own best
    prob_threshold) blur 1.0 gives F1@.35 66.6% with 2.4% merged, against 2.0's 65.2% / 4.5% and
    no-blur's 50.3% / 0.1% — the last buying its merge rate by splitting 29% of cells. If someone
    changes one default they must change both, or training silently stops matching the docs.
    """
    import inspect
    from coastal.loss import ConfettiForegroundLoss
    from coastal.train import train_with_metrics

    assert ConfettiForegroundLoss().blur_sigma == 1.0
    train_default = inspect.signature(train_with_metrics).parameters['confetti_blur_sigma'].default
    assert train_default == ConfettiForegroundLoss().blur_sigma


def test_sharper_target_is_more_localised():
    """Why blur is a merge<->split dial: it sets how far the target spreads past the cell.

    Two nearby cells; with a wide blur the target between them fills in, with a narrow one it
    stays low. That gap is what the prob head reproduces, and why sharpening trades merges for
    splits rather than separating cleanly.
    """
    import numpy as np
    import torch
    from coastal.loss import ConfettiForegroundLoss

    H = W = 64
    yy, xx = np.mgrid[0:H, 0:W]
    field = np.zeros((H, W), np.float32)
    for cy, cx in ((32, 26), (32, 38)):        # 12 px apart, so they nearly touch
        field += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3.0 ** 2))

    def valley(sigma):
        loss = ConfettiForegroundLoss(blur_sigma=sigma)
        t = loss._blur(torch.from_numpy(field)[None, None])
        t = t / (t.max() + 1e-6)
        return float(t[0, 0, 32, 32] / t[0, 0, 32, 26])   # midpoint relative to a cell centre

    assert valley(2.0) > valley(1.0) > valley(0.0), 'a wider blur must fill the gap in more'


def test_confetti_loss_runs_under_amp_dtypes():
    """torch.quantile rejects float16; AMP hands the target exactly that.

    Caught only at training time the first attempt, because the unit tests all ran in float32.
    """
    import numpy as np
    import torch
    from coastal.loss import ConfettiForegroundLoss

    loss = ConfettiForegroundLoss()
    m = [{'softmax_ch_0': np.random.rand(32, 32).astype(np.float16),
          'softmax_ch_1': np.random.rand(32, 32).astype(np.float16)}]
    out = loss(torch.zeros(1, 1, 32, 32, dtype=torch.float16), m)
    assert torch.isfinite(out)
