"""`loss.ForegroundLoss` — the no-confetti prob-head supervisor.

The claim this file defends is the measured one: **the colour term in `ConfettiForegroundLoss` is
not what makes it work.** Its target is `max_c softmax_ch_c`, and `compute_variance_metrics` builds
those channels so that the dominance factor is pinned near its `1/C` floor (measured in the
foreground on real confetti, C=3: p5 0.356, median 0.397, p95 0.528, and 0% of foreground pixels
above 0.8). What carries the signal is the cell-scale blur and the p99 rescale — both now in the
shared `_blob_target`, so the two losses cannot drift apart.

The consequence, which is the point of the class: on data that is not confetti — `zolIMa/fXgbTl`,
where nuc-GFP and mem-Tom label the nucleus and membrane of the SAME cell — the confetti premise is
false, and this loss is the one to use. Record: docs/SEGMENTATION.md -> *What confetti actually
contributes*.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from coastal.flow import VarianceMetricsConfig, compute_variance_metrics
from coastal.loss import (ConfettiForegroundLoss, ForegroundLoss, IntensityLoss,
                          _blob_target, _blur_at_cell_scale)


def _cells(seed=0, n=96, nc=3, r=7):
    """(C,H,W) uint8-ish: round bright cells, each dominant in one channel, on a noise floor."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 6, size=(nc, n, n)).astype(np.float32)
    yy, xx = np.mgrid[0:n, 0:n]
    for i, (cy, cx) in enumerate([(24, 24), (24, 68), (68, 24), (68, 68), (46, 46)]):
        m = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
        a[i % nc][m] += 180.0
    return a


def _capture_target(loss_mod, *args):
    """The target a loss builds, taken from the BCE call it makes — no reimplementation."""
    grabbed, real = {}, F.binary_cross_entropy_with_logits

    def spy(pred, target, *a, **kw):
        grabbed["t"] = target.detach().clone()
        return real(pred, target, *a, **kw)

    F.binary_cross_entropy_with_logits = spy
    try:
        loss_mod(*args)
    finally:
        F.binary_cross_entropy_with_logits = real
    return grabbed["t"]


# ── the target is blob-shaped, which is the whole point vs IntensityLoss ───────────────────────

def _photon_limited(seed=0, n=96, r=7):
    """A sparse, noisy single-plane frame — the regime the speckle measurements were made in.

    Real resonance-scanner data is 86-95% zeros with a maximum in the hundreds, and the cell
    interior is itself sparse. A clean fixture of solid discs on a flat floor does NOT reproduce
    IntensityLoss's documented failure: its `frame > mean+std` term finds those discs perfectly.
    The failure needs the sparse counts.
    """
    rng = np.random.default_rng(seed)
    a = rng.poisson(0.08, size=(n, n)).astype(np.float32)
    yy, xx = np.mgrid[0:n, 0:n]
    for cy, cx in [(24, 24), (24, 68), (68, 24), (68, 68), (46, 46)]:
        m = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
        a[m] += rng.poisson(2.2, size=m.sum())          # sparse interior, not a solid disc
    return a


def test_target_is_cell_shaped_not_speckle():
    """IntensityLoss is half a per-pixel threshold and half two edge detectors, so on sparse counts
    its target shatters; this one is a blob because brightness is blurred at cell scale first."""
    frame = torch.from_numpy(_photon_limited())[None, None]
    assert (frame == 0).float().mean() > 0.6, "fixture must be photon-limited to be a fair test"
    logits = torch.zeros_like(frame)

    fg = _capture_target(ForegroundLoss(blur_sigma=1.0), logits, frame)[0, 0].numpy()
    it = _capture_target(IntensityLoss(), logits, frame)[0, 0].numpy()

    from scipy import ndimage

    def blobs(t):
        lab, n = ndimage.label(t > 0.4)
        sizes = np.bincount(lab.ravel())[1:]
        return n, (float(np.median(sizes)) if n else 0.0)

    n_fg, med_fg = blobs(fg)
    n_it, med_it = blobs(it)
    assert med_fg > med_it, f"foreground target should be blobbier: {med_fg} vs {med_it}"
    assert n_fg <= n_it, f"and should not fragment more: {n_fg} vs {n_it}"


def test_every_cell_clears_the_inference_threshold():
    """The p99 rescale exists so typical cells clear prob_threshold — normalising by the max is
    what collapsed the first confetti retrain to 14 labels/frame."""
    frame = torch.from_numpy(_cells().max(axis=0))[None, None] / 255.0
    target = _capture_target(ForegroundLoss(), torch.zeros_like(frame), frame)[0, 0].numpy()
    from scipy import ndimage
    lab, n = ndimage.label(frame[0, 0].numpy() > 0.5)      # the 5 planted cells
    assert n >= 4
    cleared = [target[lab == i].max() > 0.4 for i in range(1, n + 1)]
    assert all(cleared), f"only {sum(cleared)}/{n} cells clear the threshold"


def test_normalising_by_max_would_lose_cells_but_p99_does_not():
    """Pin the diagnosed failure: one very bright cell must not set the scale for the rest."""
    a = _cells()
    a[0, 10:16, 10:16] += 4000.0                            # one saturated outlier
    frame = torch.from_numpy(a.max(axis=0))[None, None] / 255.0
    p99 = _capture_target(ForegroundLoss(norm_percentile=99.0),
                          torch.zeros_like(frame), frame)[0, 0]
    as_max = _blob_target(frame, 1.0, 100.0)[0, 0]          # percentile 100 == the max
    assert (p99 > 0.4).sum() > 3 * (as_max > 0.4).sum(), \
        "p99 must keep far more foreground than max-normalisation"


# ── the equivalence finding, which is why the shared helper exists ─────────────────────────────

def test_the_colour_term_barely_changes_the_target():
    """THE measured claim. On confetti-like input the confetti target and the colour-blind one
    agree closely, because `max_c softmax_ch_c` is dominance x brightness and dominance is pinned
    near 1/C. Measured on real data: r=0.9993 (confetti), 0.9986 (co-expressed markers).
    """
    a = _cells(seed=3, nc=3)
    multi = np.clip(a, 0, 255).astype(np.uint8)[None]        # [T=1, C, H, W]
    vm = compute_variance_metrics(multi, VarianceMetricsConfig())
    logits = torch.zeros(1, 1, *a.shape[1:])

    conf = _capture_target(ConfettiForegroundLoss(blur_sigma=1.0), logits, vm)[0, 0].numpy()
    frame = torch.from_numpy(multi[0].max(axis=0).astype(np.float32))[None, None] / 255.0
    agn = _capture_target(ForegroundLoss(blur_sigma=1.0), logits, frame)[0, 0].numpy()

    r = np.corrcoef(conf.ravel(), agn.ravel())[0, 1]
    iou = ((conf > 0.4) & (agn > 0.4)).sum() / max(((conf > 0.4) | (agn > 0.4)).sum(), 1)
    assert r > 0.9, f"the two targets should track each other closely, got r={r:.4f}"
    assert iou > 0.7, f"and pick out the same foreground, got IoU={iou:.3f}"


def test_dominance_is_pinned_near_its_floor():
    """The mechanism behind the equivalence: `softmax_ch_*` cannot express a confident colour.
    `softmax_temp=0.3` over Gaussian-pooled (`pool_radius=5`) intensities, then a PER-CHANNEL
    `normalize_metric`, leaves the dominant share near 1/C even for cells that are unambiguously
    one colour in the raw data."""
    a = _cells(seed=4, nc=3)
    multi = np.clip(a, 0, 255).astype(np.uint8)[None]
    vm = compute_variance_metrics(multi, VarianceMetricsConfig())
    keys = sorted(k for k in vm[0] if k.startswith("softmax_ch_"))
    sm = np.stack([vm[0][k] for k in keys])
    tot = sm.sum(axis=0)
    real = tot > 1e-4                                        # exclude the 0/1e-6 background
    dom = sm.max(axis=0)[real] / tot[real]
    assert dom.max() < 0.8, (
        f"a planted single-colour cell should read as confidently coloured but the max dominance "
        f"is {dom.max():.3f} — this is why the colour term does no work")


def test_both_losses_share_one_target_builder():
    """Not tidiness — if these diverge, the equivalence measured above silently stops holding.

    Checked by feeding both losses inputs chosen so their scalar maps are IDENTICAL: a single
    variance channel makes `max_c softmax_ch_c` equal to that channel, so the two must agree
    exactly, which they can only do if one implementation builds both targets.
    """
    x = torch.rand(2, 1, 32, 32)
    assert torch.allclose(ConfettiForegroundLoss(blur_sigma=1.5)._blur(x),
                          _blur_at_cell_scale(x, 1.5))

    m = np.abs(np.random.default_rng(2).normal(0.3, 0.2, (40, 40))).astype(np.float32)
    logits = torch.zeros(1, 1, 40, 40)
    conf = _capture_target(ConfettiForegroundLoss(blur_sigma=1.0), logits,
                           [{"softmax_ch_0": m}])
    agn = _capture_target(ForegroundLoss(blur_sigma=1.0), logits,
                          torch.from_numpy(m)[None, None])
    assert torch.equal(conf, agn), "the two losses no longer share _blob_target"


# ── shape / plumbing contracts ────────────────────────────────────────────────────────────────

def test_multi_channel_input_is_reduced_by_max_not_mean():
    """A cell bright in ONE channel must read as full foreground; a mean would dilute it by C."""
    a = np.zeros((3, 64, 64), np.float32)
    a[1, 28:36, 28:36] = 1.0                                 # bright in exactly one channel
    frame = torch.from_numpy(a)[None]                        # [1, 3, H, W]
    t = _capture_target(ForegroundLoss(), torch.zeros(1, 1, 64, 64), frame)[0, 0]
    assert t[28:36, 28:36].max() > 0.9, f"single-channel cell diluted to {t.max():.3f}"


def test_zero_blur_is_a_passthrough_and_needs_no_variance_metrics():
    frame = torch.rand(2, 1, 24, 24)
    ForegroundLoss(blur_sigma=0.0)(torch.zeros(2, 1, 24, 24), frame)   # must not raise
    assert torch.allclose(_blur_at_cell_scale(frame, 0.0), frame)


def test_loss_is_finite_and_differentiable():
    frame = torch.from_numpy(_cells().max(axis=0))[None, None] / 255.0
    logits = torch.zeros_like(frame, requires_grad=True)
    loss = ForegroundLoss()(logits, frame)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_an_empty_plane_with_zero_background_yields_an_empty_target():
    """The real case, and the reason the pedestal limitation below is latent rather than live.

    Photon-limited data is clipped at import, so an out-of-tissue z-plane is exactly zero — the p99
    is 0, and 0/1e-6 is 0. Measured: kSUFux/r0hufV z=8..13 are all like this and produce 0.00%
    foreground.
    """
    frame = torch.zeros(1, 1, 64, 64)
    t = _capture_target(ForegroundLoss(), torch.zeros(1, 1, 64, 64), frame)[0, 0]
    assert t.max() == 0.0, f"an empty plane claimed foreground up to {t.max():.3f}"


def test_a_frame_of_pure_noise_on_a_pedestal_is_claimed_entirely():
    """A KNOWN LIMITATION, pinned so it is a documented property rather than a surprise.

    `_blob_target` normalises by a percentile of the frame itself, so it cannot express "there is
    nothing here". Noise sitting on a nonzero pedestal gets stretched to fill [0,1] and the whole
    frame reads as foreground. This does NOT occur on the real data measured (background is exactly
    0 there — see the test above), and it is left unguarded on purpose: a relative contrast test
    cannot separate noise spread from cells. See `_blob_target`'s docstring.

    If this test ever starts failing, someone has added an absolute signal estimate — that is an
    improvement; update the docstring and delete this test.
    """
    rng = np.random.default_rng(11)
    frame = torch.from_numpy(rng.normal(0.1, 0.01, (1, 1, 64, 64)).astype(np.float32))
    t = _capture_target(ForegroundLoss(), torch.zeros(1, 1, 64, 64), frame)[0, 0].numpy()
    assert (t > 0.4).mean() > 0.9, (
        "the documented limitation no longer reproduces — see this test's docstring")


# ── the empty-metrics contract ─────────────────────────────────────────────────────────────────

def test_contrastive_losses_return_a_tensor_when_there_are_no_metrics():
    """Regression: `_contrastive_metric_loss` initialised `total_loss = 0.0`, so with an empty
    metrics dict every batch item hit its `continue` and `total_loss / B` returned a plain FLOAT.
    The training loop then did `l_temporal.item()` -> AttributeError.

    Only reachable when there are no metrics at all, which is exactly a flow ablation (train with
    zero flow metrics to ask what they contribute), so it sat unnoticed.
    """
    from coastal.loss import TemporalMetricsLoss, VarianceMetricsLoss
    emb = torch.rand(2, 16, 32, 32, requires_grad=True)
    for loss in (TemporalMetricsLoss(), VarianceMetricsLoss()):
        out = loss(emb, [{}, {}])
        assert isinstance(out, torch.Tensor), f"{type(loss).__name__} returned {type(out)}"
        assert out.item() == 0.0
