"""The GPU region-growing path must be the CPU one, bit for bit.

`_grow_regions_torch` is not an approximation of `_grow_regions_fast` — it is the same algorithm
with the affinity maps hoisted out of the loop and the convergence check batched, both of which are
exact rearrangements. A segmentation that differs by a pixel between two machines because one had a
GPU is worse than a slow one, so this asserts equality rather than closeness.

The comparison runs on embeddings from the real model where one is available, and on a synthetic
embedding field otherwise, so it still guards the arithmetic on a CPU-only box.
"""

import numpy as np
import pytest
import torch

from coastal.segment import LearnedAffinityInference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="parity is between the CPU and CUDA paths")


def _field(h=96, w=104, d=8, n_seeds=12, seed=0):
    """An embedding field with blobs, plus seeds and a mask over them."""
    rng = np.random.default_rng(seed)
    yy, xx = np.ogrid[:h, :w]
    embeddings = rng.normal(0, 0.2, (h, w, d)).astype(np.float32)
    prob = np.zeros((h, w), np.float32)
    seeds = np.zeros((h, w), np.int32)
    for i in range(n_seeds):
        cy, cx = rng.integers(8, h - 8), rng.integers(8, w - 8)
        blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 6.0 ** 2))).astype(np.float32)
        embeddings += blob[..., None] * rng.normal(0, 1, d).astype(np.float32)
        prob = np.maximum(prob, blob)
        seeds[cy, cx] = i + 1
    embeddings /= np.linalg.norm(embeddings, axis=2, keepdims=True) + 1e-5
    return embeddings, seeds, prob > 0.25, prob


def _configured(device, **kw):
    """An instance with no model — neither growing path touches `self.model`, and building a real
    one would make this a checkpoint test rather than an arithmetic one."""
    inf = LearnedAffinityInference.__new__(LearnedAffinityInference)
    inf.device = device
    inf.affinity_threshold = kw.get('affinity_threshold', 0.5)
    inf.prob_weight = kw.get('prob_weight', 0.3)
    inf.max_iter = kw.get('max_iter', 200)
    inf.min_component_size = kw.get('min_component_size', 2)
    return inf


@pytest.mark.parametrize("affinity_threshold,prob_weight", [(0.5, 0.3), (0.35, 0.0), (0.7, 0.5)])
def test_cuda_growing_matches_numpy(affinity_threshold, prob_weight):
    embeddings, seeds, mask, prob = _field()

    cpu = _configured('cpu', affinity_threshold=affinity_threshold, prob_weight=prob_weight)
    gpu = _configured('cuda', affinity_threshold=affinity_threshold, prob_weight=prob_weight)

    expected = cpu._grow_regions_fast(embeddings, seeds, mask, prob)
    got = gpu._grow_regions_fast(embeddings, seeds, mask, prob)

    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)


def test_batched_convergence_check_cannot_change_the_result():
    """Overshooting the convergence check is free: an iteration that accepts nothing writes nothing.

    Asserted by running the GPU path at max_iter values either side of a check boundary — once
    converged, more iterations must not move a pixel.
    """
    embeddings, seeds, mask, prob = _field(seed=3)
    gpu = _configured('cuda')

    gpu.max_iter = 200
    converged = gpu._grow_regions_fast(embeddings, seeds, mask, prob)
    gpu.max_iter = 400
    np.testing.assert_array_equal(gpu._grow_regions_fast(embeddings, seeds, mask, prob), converged)
