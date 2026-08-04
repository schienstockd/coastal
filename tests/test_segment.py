"""Construction smoke test for the two-pass segmentation inference, plus the 4D
`predict_temporal_volume` contract (label shape/dtype, Z-consistency, and the
streaming/memory guarantee that the input volume is never materialised whole).

Pins the fix for the constructor crash where `TwoPassSegmentationInference` forwarded a
`prob_merge_weight=` kwarg that `LearnedAffinityInference` does not accept (it is `prob_weight`,
the region-growing relaxation). Before the fix, *constructing* the recommended inference path
raised TypeError; this test would have caught it. No model forward pass is exercised (model=None).
"""

import numpy as np
import pytest
import torch

from coastal.segment import TwoPassSegmentationInference, LearnedAffinityInference, Inference3D


def test_two_pass_inference_constructs():
    # A stand-in module: construction only stores model.to(device).eval(); no forward pass here.
    model = torch.nn.Identity()
    seg = TwoPassSegmentationInference(
        model=model,
        seed_size_large=32, affinity_threshold_large=0.2, embedding_blur_sigma_large=1.5,
        seed_size_small=8, affinity_threshold_small=0.8, embedding_blur_sigma_small=1.5,
        merge_affinity_threshold_large=0.90, merge_affinity_threshold_small=0.90,
        prob_weight_large=0.3, prob_weight_small=0.3,
        prob_threshold=0.3, min_component_size=10, device="cpu",
    )
    # Both passes are real LearnedAffinityInference instances with the relaxation wired through.
    assert isinstance(seg.pass1, LearnedAffinityInference)
    assert isinstance(seg.pass2, LearnedAffinityInference)
    assert seg.pass1.prob_weight == 0.3
    assert seg.pass2.prob_weight == 0.3


# --------------------------------------------------------------------------- #
# predict_temporal_volume: 4D contract + streaming guarantee                   #
# --------------------------------------------------------------------------- #

class _StubUNet(torch.nn.Module):
    """Two fixed, well-separated blobs per frame; constant (unit) embeddings.

    Deterministic stand-in for the trained UNet so the 4D bookkeeping — per-slice
    label writes, Z-stitching, dtype — is testable without a real forward pass.
    """
    num_metrics = 14

    def forward(self, x):
        n, _, h, w = x.shape
        prob = torch.full((n, 1, h, w), -6.0)   # sigmoid ≈ 0 (background)
        prob[:, :, 3:7, 3:7] = 6.0              # blob A: sigmoid ≈ 1
        prob[:, :, 3:7, 11:15] = 6.0            # blob B
        emb = torch.ones(n, 4, h, w)            # affinity 1.0 everywhere → seeds grow
        return prob, emb


class _LazyVolume:
    """Minimal stand-in for a cecelia dask pyramid level.

    Supports only `.shape` and the per-z-slice indexing `predict_temporal_volume`
    is documented to use. `__array__` raises, so any attempt to materialise the
    whole movie (`np.asarray(volume)`) fails the test instead of silently costing
    T×C×Z×H×W bytes.
    """

    def __init__(self, arr):
        self._arr = arr
        self.shape = arr.shape
        self.slices_read = []

    def __array__(self, dtype=None):
        raise AssertionError(
            "whole volume materialised — predict_temporal_volume must stream per z-slice"
        )

    def __getitem__(self, key):
        assert isinstance(key, tuple) and len(key) == 5, f"unexpected indexing: {key!r}"
        assert isinstance(key[2], (int, np.integer)), f"expected a single z index, got {key!r}"
        self.slices_read.append(int(key[2]))
        return self._arr[key]


def _stub_inferencer():
    return Inference3D(
        model=_StubUNet(), device="cpu",
        seed_size=4, embedding_blur_sigma=0.5, prob_threshold=0.4,
        min_component_size=2, min_boundary_pixels=1,
        affinity_threshold=0.5, merge_affinity_threshold=0.9,
        merge_max_distance=0.0,   # merging off: keep the two blobs distinct
    )


@pytest.fixture(scope="module")
def temporal_volume():
    """[T, C, Z, Y, X] with the two blobs bright in both channels."""
    rng = np.random.default_rng(0)
    vol = rng.integers(0, 20, size=(4, 2, 3, 18, 18)).astype(np.uint16)
    vol[:, :, :, 3:7, 3:7] = 900
    vol[:, :, :, 3:7, 11:15] = 900
    return vol


def test_predict_temporal_volume_streams_and_stitches(temporal_volume):
    T, C, Z, H, W = temporal_volume.shape
    lazy = _LazyVolume(temporal_volume)

    instances, results = _stub_inferencer().predict_temporal_volume(
        lazy, ch_indices=[0, 1], stitch_threshold=0.1, n_workers=2,
    )

    # Streaming contract: each z-slice read exactly once, never the whole volume.
    assert sorted(lazy.slices_read) == list(range(Z))
    # Labels keep the documented shape/dtype.
    assert instances.shape == (T, Z, H, W)
    assert instances.dtype == np.int32
    # Default drops the per-frame prob maps / regionprops (the 4D memory blow-up).
    assert results is None

    # The stub yields two separated blobs, stitched to the same ids down Z.
    for t in range(T):
        labels_per_z = [set(np.unique(instances[t, z])) - {0} for z in range(Z)]
        assert all(len(s) == 2 for s in labels_per_z), labels_per_z
        assert all(s == labels_per_z[0] for s in labels_per_z), labels_per_z


def test_predict_temporal_volume_keep_results_matches_labels(temporal_volume):
    """keep_results is inspection-only: it must not change the labels."""
    lean, none_results = _stub_inferencer().predict_temporal_volume(
        _LazyVolume(temporal_volume), ch_indices=[0, 1], stitch_threshold=0.1, n_workers=2,
    )
    kept, results = _stub_inferencer().predict_temporal_volume(
        _LazyVolume(temporal_volume), ch_indices=[0, 1], stitch_threshold=0.1, n_workers=2,
        keep_results=True,
    )

    assert none_results is None
    np.testing.assert_array_equal(lean, kept)

    T, Z = temporal_volume.shape[0], temporal_volume.shape[2]
    assert len(results) == Z and all(len(r) == T for r in results)
    assert 'prob_map' in results[0][0] and 'props' in results[0][0]
