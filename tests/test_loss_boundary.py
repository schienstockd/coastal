"""`ConfettiBoundaryLoss` must make a colour boundary visible in the embeddings.

The gap it exists to close, measured on real data: segmentation merges 86.7% of genuinely
touching different-colour cell pairs, because embedding cosine across such a contact (0.920)
is barely distinguishable from cosine inside one cell (0.945). No inference-side rule can
recover a step that is not there — see docs/SEGMENTATION.md.

Confetti builds the *pairing* only. At inference the variance channels are zero-filled, so the
loss must be a strict no-op without them.
"""

import numpy as np
import pytest
import torch

from coastal.loss import ConfettiBoundaryLoss

H = W = 32


def _two_colours(split=W // 2, conf=1.0):
    """softmax_ch_* for two bright, confidently-coloured cells meeting at a vertical line."""
    a = np.zeros((H, W), np.float32)
    b = np.zeros((H, W), np.float32)
    a[:, :split] = conf
    b[:, split:] = conf
    return {'softmax_ch_0': a, 'softmax_ch_1': b, 'softmax_ch_2': np.zeros((H, W), np.float32)}


def _embedding(left, right, split=W // 2, dim=8):
    """[1, D, H, W] with one vector left of the split and another right of it."""
    e = torch.zeros(1, dim, H, W)
    e[0, :, :, :split] = left.detach().clone().float().view(-1, 1, 1)
    e[0, :, :, split:] = right.detach().clone().float().view(-1, 1, 1)
    return e


def _vec(i, dim=8):
    v = torch.zeros(dim)
    v[i] = 1.0
    return v


def test_separated_embeddings_beat_uniform_ones_across_a_colour_boundary():
    """The whole point: identical embeddings either side of a contact must cost more."""
    loss = ConfettiBoundaryLoss()
    vm = [_two_colours()]

    uniform = _embedding(_vec(0), _vec(0))          # no boundary in the embedding
    separated = _embedding(_vec(0), _vec(1))        # orthogonal across the boundary

    assert loss(separated, vm).item() < loss(uniform, vm).item()


def test_no_confetti_metrics_is_a_no_op():
    """At inference the variance channels are zero-filled — this must contribute nothing."""
    loss = ConfettiBoundaryLoss()
    emb = torch.randn(1, 8, H, W)
    assert loss(emb, []).item() == 0.0
    assert loss(emb, [{}]).item() == 0.0
    # A single channel cannot define a "different colour" either.
    assert loss(emb, [{'softmax_ch_0': np.ones((H, W), np.float32)}]).item() == 0.0


def test_dim_background_is_never_a_boundary():
    """One channel "winning" on noise in dark background must not create negatives.

    Without the brightness gate every background pixel pair would be mined as a contact and
    the term would be dominated by noise rather than by cells.
    """
    loss = ConfettiBoundaryLoss(min_confidence=0.5)
    rng = np.random.default_rng(0)
    faint = {f'softmax_ch_{c}': (rng.random((H, W)) * 1e-3).astype(np.float32) for c in range(3)}
    emb = _embedding(_vec(0), _vec(0))
    assert loss(emb, [faint]).item() == 0.0


def test_positive_term_prevents_scattering_everything():
    """Negatives alone are minimised by making every embedding orthogonal — the pull stops that."""
    vm = [_two_colours()]
    dim = 8
    coherent = _embedding(_vec(0), _vec(1))
    # Same colour blocks, but noisy inside each one: correct across the boundary, wrong within.
    scattered = torch.randn(1, dim, H, W)

    with_pos = ConfettiBoundaryLoss(pos_weight=0.3)
    assert with_pos(coherent, vm).item() < with_pos(scattered, vm).item()

    # With the pull switched off, scattering is no longer penalised the same way: the point is
    # that pos_weight is what makes within-colour coherence part of the objective.
    no_pos = ConfettiBoundaryLoss(pos_weight=0.0)
    gap_with = with_pos(scattered, vm).item() - with_pos(coherent, vm).item()
    gap_without = no_pos(scattered, vm).item() - no_pos(coherent, vm).item()
    assert gap_with > gap_without


def test_gradients_flow_to_the_embeddings():
    """Perturbed, not exactly uniform: with identical unit vectors everywhere the cosine
    gradient is purely radial and normalisation legitimately zeroes it, which says nothing
    about whether the term trains."""
    loss = ConfettiBoundaryLoss()
    emb = (_embedding(_vec(0), _vec(0)) + 0.05 * torch.randn(1, 8, H, W)).requires_grad_(True)
    out = loss(emb, [_two_colours()])
    out.backward()
    assert emb.grad is not None and torch.isfinite(emb.grad).all()
    assert emb.grad.abs().sum() > 0


def test_runs_under_amp_dtypes():
    """torch.quantile rejects float16, which AMP hands us — the bug that hit ConfettiForegroundLoss."""
    loss = ConfettiBoundaryLoss()
    m = [{f'softmax_ch_{c}': np.random.rand(H, W).astype(np.float16) for c in range(3)}]
    out = loss(torch.randn(1, 8, H, W, dtype=torch.float16), m)
    assert torch.isfinite(out)


@pytest.mark.parametrize('offsets', [(1,), (2, 4), (3,)])
def test_offsets_all_detect_the_same_boundary(offsets):
    """Whatever the sampling distance, a real boundary must still be preferred."""
    loss = ConfettiBoundaryLoss(offsets=offsets)
    vm = [_two_colours()]
    assert loss(_embedding(_vec(0), _vec(1)), vm).item() < \
           loss(_embedding(_vec(0), _vec(0)), vm).item()


def test_batch_is_averaged_not_summed():
    """Two identical images must cost the same as one, or the term scales with batch size."""
    loss = ConfettiBoundaryLoss()
    emb1 = _embedding(_vec(0), _vec(0))
    vm1 = [_two_colours()]
    one = loss(emb1, vm1).item()
    two = loss(torch.cat([emb1, emb1], 0), vm1 * 2).item()
    assert one == pytest.approx(two, rel=1e-5)
