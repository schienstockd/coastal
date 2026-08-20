"""The irreducible floor of a soft-target BCE term — `loss.bce_floor` and the `with_floor` pairs.

WHY this exists at all. Every prob-head term here is `binary_cross_entropy_with_logits` against a
target that is a deterministic function of the input, and BCE against a SOFT target cannot reach 0:
its minimum is the target's own binary entropy. That minimum belongs to the DATA, not the model, so a
raw loss curve is unreadable — measured on flow.cyto (6 images, 2880 frames, 100 epochs) `foreground`
settles at 0.26508 against a floor of 0.26499. The model's entire remaining error is 0.00009, and
"the loss plateaus after five epochs and nothing is learned" turns out to be a description of
convergence. 85% of the plotted TOTAL was a constant.

Two things have to hold for the reported number to be worth anything:

  1. it must actually BE the minimum — no achievable loss below it, and exactly reached by a perfect
     prediction. That is the definition, and it is cheap to check directly rather than trust.
  2. it must come from the SAME target the loss used. A floor built from a separately constructed
     target is the classic way for a curve and the constant subtracted from it to drift apart while
     both still look plausible — which is why `with_floor` exists and `forward` delegates to it.

Real torch, tiny CPU tensors: this is arithmetic, not segmentation.
"""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from coastal.loss import (bce_floor, ConfettiForegroundLoss, ForegroundLoss, IntensityLoss)


def _frame(B=2, C=1, H=16, W=16, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.random((B, C, H, W)).astype(np.float32))


def _variance(B=2, H=16, W=16, n=2, seed=3):
    rng = np.random.default_rng(seed)
    return [{f"softmax_ch_{i}": rng.random((H, W)).astype(np.float32) for i in range(n)}
            for _ in range(B)]


# ── bce_floor itself ────────────────────────────────────────────────────────────────────────────
def test_the_floor_is_what_a_perfect_prediction_scores():
    """The definition: predict the target exactly and the loss IS the floor."""
    target = torch.rand(2, 1, 8, 8).clamp(1e-3, 1 - 1e-3)
    perfect = torch.logit(target)                    # the logits whose sigmoid is the target
    assert F.binary_cross_entropy_with_logits(perfect, target).item() == \
        pytest.approx(bce_floor(target).item(), abs=1e-6)


@pytest.mark.parametrize('seed', range(8))
def test_no_prediction_beats_the_floor(seed):
    """The claim that makes it a floor. Any logits at all, never below."""
    g = torch.Generator().manual_seed(seed)
    target = torch.rand(2, 1, 8, 8, generator=g)
    logits = torch.randn(2, 1, 8, 8, generator=g) * 4      # wide, so some are wildly wrong
    assert F.binary_cross_entropy_with_logits(logits, target).item() >= \
        bce_floor(target).item() - 1e-6


def test_a_hard_target_has_a_floor_of_zero():
    """0/1 targets are separable, so a perfect model reaches 0 and there is nothing to subtract.

    Not exactly 0: the clamp at `1e-7` puts `H` at ~1.7e-6 there, four orders below anything a curve
    is read to. Asserted at that tolerance rather than hidden, because the alternative — an unclamped
    `log(0)` — is a `nan` that silently takes out the whole epoch mean.
    """
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    assert bce_floor(target).item() == pytest.approx(0.0, abs=1e-5)


def test_a_maximally_uncertain_target_floors_at_ln_2():
    """The other end: H(0.5) = ln 2 = 0.693, i.e. the whole loss is the constant."""
    assert bce_floor(torch.full((1, 1, 4, 4), 0.5)).item() == pytest.approx(math.log(2), abs=1e-6)


def test_it_is_finite_at_exactly_zero_and_one():
    """`log(0)` would be `-inf` and one saturated pixel would poison the epoch's mean."""
    assert torch.isfinite(bce_floor(torch.tensor([[[[0.0, 1.0]]]])))


def test_it_carries_no_gradient():
    """A reported constant, never an objective — a floor in the graph would be optimised against."""
    target = torch.rand(1, 1, 4, 4, requires_grad=True)
    assert not bce_floor(target).requires_grad


# ── the loss classes ───────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('name', ['foreground', 'intensity', 'confetti'])
def test_with_floor_returns_the_same_loss_forward_does(name):
    """`forward` delegates to `with_floor`, so these cannot disagree — pinned so they stay that way.

    If they ever diverge, the plot subtracts a floor from a curve measured on a different target.
    """
    logits = torch.randn(2, 1, 16, 16)
    if name == 'confetti':
        loss, arg = ConfettiForegroundLoss(), _variance()
    elif name == 'intensity':
        loss, arg = IntensityLoss(), _frame()
    else:
        loss, arg = ForegroundLoss(), _frame()
    assert loss.with_floor(logits, arg)[0].item() == pytest.approx(loss(logits, arg).item())


@pytest.mark.parametrize('name', ['foreground', 'intensity', 'confetti'])
def test_the_reported_floor_is_the_floor_of_the_losss_own_target(name):
    """Same target, two ways: `target()` -> `bce_floor` must equal what `with_floor` reports."""
    logits = torch.randn(2, 1, 16, 16)
    if name == 'confetti':
        loss, arg = ConfettiForegroundLoss(), _variance()
        target = loss.target(logits, arg)
    elif name == 'intensity':
        loss, arg = IntensityLoss(), _frame()
        target = loss.target(arg)
    else:
        loss, arg = ForegroundLoss(), _frame()
        target = loss.target(arg)
    assert loss.with_floor(logits, arg)[1].item() == pytest.approx(bce_floor(target).item())


@pytest.mark.parametrize('name', ['foreground', 'intensity', 'confetti'])
def test_the_loss_is_never_below_its_own_reported_floor(name):
    """The end-to-end version of the property, through the real target constructions."""
    logits = torch.randn(2, 1, 16, 16) * 3
    if name == 'confetti':
        loss, arg = ConfettiForegroundLoss(), _variance()
    elif name == 'intensity':
        loss, arg = IntensityLoss(), _frame()
    else:
        loss, arg = ForegroundLoss(), _frame()
    value, floor = loss.with_floor(logits, arg)
    assert value.item() >= floor.item() - 1e-5


def test_confetti_with_no_metrics_reports_zero_for_both_not_nan():
    """No target means no objective. A `nan` floor would poison the whole epoch mean."""
    loss = ConfettiForegroundLoss()
    value, floor = loss.with_floor(torch.randn(1, 1, 8, 8), [{}])
    assert value.item() == 0.0 and floor.item() == 0.0


def _blobs(B=2, H=128, W=128, n=25, sigma_px=3.0, seed=5):
    """Sparse bright blobs on a ZERO background — a frame shaped like the real data.

    `_frame`'s uniform noise is the wrong fixture for anything about the blur, and inverts the result:
    on i.i.d. noise there is no structure to spread, so blurring pulls every pixel toward the mean and
    the floor FALLS (0.548 -> 0.390 over sigma 1..6). Real intravital frames are a few percent bright
    cells on a background that is exactly 0 (photon-limited and clipped at import — see
    `_blob_target`), and there blurring spreads sparse peaks into mid-range values over a wider area,
    which is what raises the entropy.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    out = np.zeros((B, 1, H, W), dtype=np.float32)
    for b in range(B):
        for _ in range(n):
            cy, cx = rng.integers(8, H - 8), rng.integers(8, W - 8)
            out[b, 0] += 200 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma_px ** 2))
    return torch.from_numpy(np.clip(out, 0, 255))


def test_the_foreground_floor_rises_with_the_blur():
    """Why the loss cannot referee `foreground_blur_sigma`.

    A wider blur makes the target SOFTER, so its entropy — and therefore the best achievable loss —
    goes UP. Measured on real crops of zolIMa/VJy1Nx (0.331 um/px): floor 0.262 at sigma=1 and 0.334
    at sigma=6, while the target goes from 70 specks/frame (median 0.9 um^2) to 6 cell-sized blobs
    (median 46 um^2). So the better-shaped target scores WORSE, and ranking two blurs by their loss
    curves ranks them backwards — the choice needs a downstream fragment count instead.

    Pinned because it is the trap, not an incidental property. Reproduced here on `_blobs`, whose
    docstring records what happens on the wrong fixture.
    """
    frame = _blobs()
    logits = torch.zeros(frame.shape[0], 1, *frame.shape[2:])
    floors = [ForegroundLoss(blur_sigma=s).with_floor(logits, frame)[1].item()
              for s in (1.0, 2.0, 3.0, 6.0, 9.0)]
    assert floors == sorted(floors), floors
    assert floors[-1] > 2 * floors[0], floors


# ── what the training loop records ─────────────────────────────────────────────────────────────
def _tiny(T=4, H=16, W=16, seed=0):
    rng = np.random.default_rng(seed)
    frames = rng.random((T, H, W)).astype(np.float32)
    temporal = [{f"m_{i}": rng.random((H, W)).astype(np.float32) for i in range(3)}
                for _ in range(T)]
    return frames, temporal


RUN = dict(num_epochs=2, batch_size=2, num_workers=0, use_amp=False, device='cpu', embedding_dim=4,
           foreground_weight=1.0, intensity_weight=1.0, temporal_weight=2.0)


def test_the_history_carries_a_floor_per_bce_term_and_none_for_the_contrastive_ones():
    """The manifest's `lossFloors` is this, split out by `opticalFlow.train`.

    A floor for the contrastive terms would have to be a fabricated 0, indistinguishable from a
    measured one — they are hinges whose minimum genuinely is 0, so the honest report is no key.
    """
    from coastal.train import train_with_metrics
    frames, temporal = _tiny()
    _, history = train_with_metrics(frames, temporal, **RUN)
    assert sorted(k for k in history if k.startswith('floor_')) == \
        ['floor_confetti', 'floor_foreground', 'floor_intensity']
    assert not any(f'floor_{t}' in history for t in ('temporal', 'variance', 'warp', 'boundary'))
    assert len(history['floor_foreground']) == len(history['foreground']) == 2


def test_the_recorded_term_never_dips_below_its_recorded_floor():
    """The property the plot depends on: subtracting the floor cannot produce a negative curve."""
    from coastal.train import train_with_metrics
    frames, temporal = _tiny(seed=2)
    _, history = train_with_metrics(frames, temporal, **RUN)
    for term in ('foreground', 'intensity'):
        for value, floor in zip(history[term], history[f'floor_{term}']):
            assert value >= floor - 1e-5, (term, value, floor)


def test_the_held_out_pass_records_its_own_floors():
    """Its own, not the training ones — different frames, different target entropy.

    Measured on flow.cyto the two differ by 0.0005 (0.26499 train, 0.26449 val), which is 5x the
    train excess: subtracting the wrong one would put the entire generalisation gap in the noise.
    """
    from coastal.train import train_with_metrics
    frames, temporal = _tiny()
    v_frames, v_temporal = _tiny(T=2, seed=7)
    _, history = train_with_metrics(frames, temporal, val_frames=v_frames,
                                    val_temporal_metrics_norm=v_temporal, **RUN)
    assert len(history['val_floor_foreground']) == 2
    assert history['val_floor_foreground'][0] != history['floor_foreground'][0]
