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


def test_intersection_over_union_is_jaccard_not_containment():
    # x label 1 = 10 px, y label 1 = 20 px, overlap = 5 px.
    # True IOU (Jaccard) = 5 / (10 + 20 - 5) = 0.2.
    # The old (buggy) row-L1 containment would give 5/10 = 0.5 — this test pins the fix.
    x = np.zeros(30, dtype=np.int32); x[:10] = 1
    y = np.zeros(30, dtype=np.int32); y[5:25] = 1
    iou = intersection_over_union(x.reshape(6, 5), y.reshape(6, 5)).toarray()
    assert np.isclose(iou[1, 1], 0.2)
    assert not np.isclose(iou[1, 1], 0.5)


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

def _plane_ids_collide():
    """Two unrelated cells on planes 0 and 2, an empty plane between, both with per-plane id 1."""
    m = np.zeros((3, 10, 10), dtype=np.int32)
    m[0, 1:3, 1:3] = 1
    m[2, 7:9, 7:9] = 1
    return m


def test_match_masks_3d_empty_plane_does_not_reuse_ids():
    # Plane 1 is empty, so plane 2 has nothing to match against. Its raw per-plane id 1 used to
    # survive and collide with plane 0's cell 1: two cells, one label, centroid in the gap.
    matched = match_masks_3d(_plane_ids_collide(), stitch_threshold=0.0)
    assert matched[0, 1, 1] != matched[2, 7, 7]
    assert (matched > 0).sum() == (_plane_ids_collide() > 0).sum()


def test_match_masks_3d_zero_overlap_objects_get_distinct_labels():
    # Same input id on ADJACENT planes but no overlap: not a match, so two labels.
    m = np.zeros((2, 10, 10), dtype=np.int32)
    m[0, 1:3, 1:3] = 1
    m[1, 7:9, 7:9] = 1
    matched = match_masks_3d(m, stitch_threshold=0.0, gap_tolerance=0)
    assert matched[0, 1, 1] != matched[1, 7, 7]


def test_match_masks_3d_bridges_the_same_cell_across_an_empty_plane():
    # The fresh id given after an empty plane must not stop the gap bridge reconnecting a real cell.
    m = np.zeros((3, 10, 10), dtype=np.int32)
    m[0, 2:6, 2:6] = 1
    m[2, 2:6, 2:6] = 1
    matched = match_masks_3d(m, stitch_threshold=0.0, gap_tolerance=1)
    assert matched[0, 3, 3] == matched[2, 3, 3] != 0


def test_match_masks_3d_bridge_leaves_a_continuing_chain_whole():
    # A ends at plane 1. B runs planes 1-4 and widens over A's footprint from plane 3. The bridge
    # from A (plane 1) to plane 3 must not take B's tail: B never broke, so it stays one label.
    m = np.zeros((5, 12, 12), dtype=np.int32)
    m[0:2, 1:5, 1:5] = 1
    m[1:3, 1:5, 6:10] = 2
    m[3:5, 1:5, 1:10] = 2
    matched = match_masks_3d(m, stitch_threshold=0.0, gap_tolerance=1)
    b = {int(matched[z, 2, 7]) for z in range(1, 5)}
    assert len(b) == 1
    assert matched[0, 2, 2] not in b
