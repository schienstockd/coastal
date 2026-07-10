"""Tests for coastal.utils — 3D label matching, IOU, small-cell filtering.

These exercise the segmentation post-processing invariants (see docs/SEGMENTATION.md §6).
`utils.py` has no torch/cv2 dependency, so this is the cheapest real test in the suite.
Importing `coastal.utils` still triggers the package __init__ (which imports torch etc.), so
the package must be installed (`pip install -e .`) to run these.
"""

import numpy as np

from coastal.utils import (
    filter_small_cells,
    intersection_over_union,
    match_masks_3d,
)


def test_filter_small_cells_drops_only_small_labels():
    # One timepoint, one Z-slice, 4x4. Label 1 = 2 voxels, label 2 = 8 voxels.
    frame = np.zeros((4, 4), dtype=np.int32)
    frame[0, :2] = 1              # 2 voxels
    frame[2:, :] = 2             # 8 voxels
    instances_4d = frame[None, None]  # [T=1, Z=1, H, W]

    out = filter_small_cells(instances_4d, min_voxels=5)

    assert 1 not in np.unique(out)          # small label removed
    assert 2 in np.unique(out)              # large label kept
    assert (out == 2).sum() == 8            # large label untouched
    assert instances_4d[0, 0, 0, 0] == 1    # input not mutated (copy semantics)


def test_filter_small_cells_keeps_all_when_threshold_low():
    frame = np.zeros((4, 4), dtype=np.int32)
    frame[0, :2] = 1
    frame[2:, :] = 2
    out = filter_small_cells(frame[None, None], min_voxels=1)
    assert set(np.unique(out)) == {0, 1, 2}


def test_intersection_over_union_perfect_overlap():
    # Identical single-label maps → IOU of label 1 vs label 1 is 1.0.
    x = np.zeros((4, 4), dtype=np.int32)
    x[1:3, 1:3] = 1
    iou = intersection_over_union(x, x).toarray()
    assert np.isclose(iou[1, 1], 1.0)


def test_match_masks_3d_unifies_same_object_across_slices():
    # Same 2x2 object in two Z-slices under different label ids (5 and 9).
    # After matching, both slices must carry one identical nonzero label.
    z0 = np.zeros((4, 4), dtype=np.int32)
    z0[1:3, 1:3] = 5
    z1 = np.zeros((4, 4), dtype=np.int32)
    z1[1:3, 1:3] = 9
    masks_3d = np.stack([z0, z1], axis=0)

    matched = match_masks_3d(masks_3d, stitch_threshold=0.0, gap_tolerance=0)

    lbls0 = set(np.unique(matched[0])) - {0}
    lbls1 = set(np.unique(matched[1])) - {0}
    assert len(lbls0) == 1 and len(lbls1) == 1
    assert lbls0 == lbls1                    # the object keeps one label through Z

# NOTE: a "non-overlapping objects get distinct labels" test was intentionally NOT added:
# at stitch_threshold=0.0, match_masks_3d keeps two zero-overlap objects that share an input
# label as the SAME label rather than relabeling them apart. That relabeling semantic is a
# quirk worth pinning down with a dedicated characterization test — see docs/TODO.md.
