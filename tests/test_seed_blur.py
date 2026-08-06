"""`seed_blur_sigma` splits seeding from the outline, which used to share `prob_blur_sigma`.

Why the parameter exists: seeds are local maxima of the prob map and every connected component of
the thresholded map is guaranteed one, so the instance count has a hard floor at the component
count. Blurring the prob map is the only way under that floor — and it simultaneously rounds the
boundary off, which on a cytoplasmic reporter destroys the actual readout. The two jobs want
opposite amounts of blur.

The load-bearing test is the first one: default 0.0 must reproduce the previous behaviour exactly,
because every existing tuned parameter set was found against it.
"""
import numpy as np
import pytest
import torch

from coastal.segment import LearnedAffinityInference


class _TwoBlobModel(torch.nn.Module):
    """Returns a fixed prob map + embedding, so the test exercises inference, not training.

    The prob map is two cells that each break into 3 bright specks — the fragmentation pattern
    `seed_blur_sigma` exists to survive.
    """

    def __init__(self, H=96, W=96, D=8):
        super().__init__()
        self.H, self.W, self.D = H, W, D
        self.num_metrics = 0          # inference zero-fills up to this; we pass no metrics
        prob = np.zeros((H, W), np.float32)
        for cy, cx in ((28, 28), (28, 66), (66, 28), (66, 66)):
            yy, xx = np.ogrid[:H, :W]
            prob += 0.9 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3.0 ** 2)))
        self._prob = np.clip(prob, 0, 1)
        emb = np.zeros((D, H, W), np.float32)
        emb[0] = 1.0                       # one uniform embedding -> growing is unconstrained
        self._emb = emb
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b = x.shape[0]
        p = torch.from_numpy(self._prob)[None, None].repeat(b, 1, 1, 1)
        e = torch.from_numpy(self._emb)[None].repeat(b, 1, 1, 1)
        return p, e


@pytest.fixture
def model():
    return _TwoBlobModel().eval()


def _frame(H=96, W=96):
    return np.zeros((H, W), np.float32)


def _run(model, **kw):
    base = dict(device="cpu", prob_threshold=0.3, seed_size=9, affinity_threshold=0.3,
                merge_max_distance=0.0, min_component_size=5, max_iter=50)
    base.update(kw)
    inf = LearnedAffinityInference(model=model, **base)
    _, instances, _ = inf.predict_frame(_frame(), {})
    return np.asarray(instances)


def test_default_is_bit_identical_to_previous_behaviour(model):
    """seed_blur_sigma=0.0 must change nothing — every tuned param set assumes the old path."""
    a = _run(model, prob_blur_sigma=0.0)
    b = _run(model, prob_blur_sigma=0.0, seed_blur_sigma=0.0)
    assert np.array_equal(a, b)
    c = _run(model, prob_blur_sigma=2.0)
    d = _run(model, prob_blur_sigma=2.0, seed_blur_sigma=0.0)
    assert np.array_equal(c, d), "with prob_blur set, seed_map must still BE prob_map"


def test_seed_blur_reduces_the_instance_floor(model):
    """Blurring only the seed map must merge the fragments into fewer objects."""
    n_sharp = len(np.unique(_run(model, prob_blur_sigma=0.0))) - 1
    n_blur = len(np.unique(_run(model, prob_blur_sigma=0.0, seed_blur_sigma=8.0))) - 1
    assert n_blur < n_sharp, f"seed blur did not merge anything ({n_sharp} -> {n_blur})"


def test_outline_stays_sharp_when_only_seeds_are_blurred(model):
    """The whole point: fewer objects WITHOUT moving the boundary.

    `prob_blur_sigma` necessarily changes the footprint, because it is the map the mask is
    thresholded from — on real data it inflates it (area 25% vs a 14% baseline); on compact
    synthetic blobs it shrinks it slightly instead, since blurring lowers the peak. Either way it
    MOVES. `seed_blur_sigma` must leave it alone: that is the whole distinction.
    """
    area_sharp = (_run(model, prob_blur_sigma=0.0) > 0).sum()
    area_seedblur = (_run(model, prob_blur_sigma=0.0, seed_blur_sigma=8.0) > 0).sum()
    area_probblur = (_run(model, prob_blur_sigma=8.0) > 0).sum()
    assert area_seedblur == pytest.approx(area_sharp, rel=0.05), \
        "seed blur changed the footprint — it must only affect seeding"
    assert area_probblur != area_sharp, \
        "sanity: prob blur is expected to move the boundary; if not, the contrast is untested"


def test_seeds_outside_the_growth_mask_are_not_stranded(model):
    """A seed found on the blurred map can land where the sharp mask is empty. It must not create
    a label there, and must not crash."""
    inst = _run(model, prob_blur_sigma=0.0, seed_blur_sigma=12.0, prob_threshold=0.6)
    assert inst.min() >= 0
    labels = set(np.unique(inst)) - {0}
    for lab in labels:
        assert (inst == lab).sum() > 0
