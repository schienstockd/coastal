"""The bounding-box/LUT rewrites of the per-label loops must be output-identical.

`_fill_holes`, `_merge_split_instances` and `_remove_small_components` each used to loop
over ~600 labels per frame doing whole-frame work (`binary_fill_holes`,
`distance_transform_edt`, `instances == label`), which profiling put at 45% / 30% / 9% of
segmentation time. They now work inside per-label bounding boxes (or a single lookup-table
gather).

These are pure speed rewrites, so the oracle here is the **previous implementation, kept
verbatim below**, run against the new one over randomised label maps. Anything that shifts a
label boundary shows up as a mismatch.
"""

import numpy as np
import pytest
import torch
from scipy.ndimage import distance_transform_edt, binary_fill_holes

from coastal.segment import LearnedAffinityInference


# --------------------------------------------------------------------------- #
# Reference implementations (verbatim pre-optimisation code)                    #
# --------------------------------------------------------------------------- #

def ref_fill_holes(instances):
    instances_filled = instances.copy()
    for inst_id in np.unique(instances):
        if inst_id == 0:
            continue
        mask = (instances == inst_id).astype(np.uint8)
        mask_filled = binary_fill_holes(mask).astype(np.uint8)
        instances_filled[mask_filled == 1] = inst_id
    return instances_filled


def ref_remove_small_components(seg, instances, min_component_size):
    instances = instances.copy()          # the old code mutated its argument
    unique_labels = np.unique(instances)
    for label_id in unique_labels:
        if label_id == 0:
            continue
        if (instances == label_id).sum() < min_component_size:
            instances[instances == label_id] = 0

    unique_labels = np.unique(instances)
    instances_reindexed = np.zeros_like(instances)
    for new_id, old_id in enumerate(unique_labels[1:], 1):
        instances_reindexed[instances == old_id] = new_id
    return instances_reindexed


def ref_merge_split_instances(seg, instances, embeddings, prob_map):
    unique_ids = np.unique(instances)
    unique_ids = unique_ids[unique_ids > 0]
    merges = {}

    for inst_id in unique_ids:
        mask = instances == inst_id
        dist1 = distance_transform_edt(~mask)
        candidate_pixels = (dist1 <= seg.merge_max_distance) & ~mask & (instances > 0)
        neighbors = np.unique(instances[candidate_pixels])

        for neighbor_id in neighbors:
            if neighbor_id in merges:
                continue
            contact_prob, _, _ = seg._compute_boundary_intensity(
                instances, prob_map, inst_id, neighbor_id, dist1)
            n_contact = seg._count_contact_pixels(instances, inst_id, neighbor_id, dist1)
            if n_contact < seg.min_boundary_pixels:
                continue
            if contact_prob < seg.merge_contact_brightness_threshold:
                continue
            affinity = seg._compute_fragment_affinity(instances, embeddings, inst_id, neighbor_id)
            if affinity < seg.merge_affinity_threshold:
                continue
            id_keep, id_remove = min(inst_id, neighbor_id), max(inst_id, neighbor_id)
            merges[id_remove] = id_keep

    def find_root(x):
        while x in merges:
            x = merges[x]
        return x

    instances_merged = instances.copy()
    for id_remove in merges:
        instances_merged[instances_merged == id_remove] = find_root(id_remove)
    return ref_remove_small_components(seg, instances_merged, seg.min_component_size)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

def _seg(**kw):
    params = dict(
        seed_size=4, embedding_blur_sigma=0.5, prob_threshold=0.4, min_component_size=4,
        min_boundary_pixels=1, affinity_threshold=0.5, merge_affinity_threshold=0.2,
        merge_max_distance=2.0, merge_contact_brightness_threshold=0.3, prob_weight=0.3,
    )
    params.update(kw)
    # No forward pass here — only the post-processing helpers are exercised.
    return LearnedAffinityInference(model=torch.nn.Identity(), device='cpu', **params)


def _label_map(seed, H=64, W=70, n=14):
    """Blobs with holes, touching pairs, and labels on the frame edge."""
    rng = np.random.default_rng(seed)
    inst = np.zeros((H, W), dtype=np.int32)
    for lab in range(1, n + 1):
        h = int(rng.integers(4, 11))
        w = int(rng.integers(4, 11))
        y = int(rng.integers(0, H - h))
        x = int(rng.integers(0, W - w))
        inst[y:y + h, x:x + w] = lab
        if h >= 6 and w >= 6 and rng.random() < 0.6:        # punch a hole
            inst[y + 2:y + h - 2, x + 2:x + w - 2] = 0
    # guarantee frame-edge cases (holes/boxes clipped by the border)
    inst[0:5, 0:5] = n + 1
    inst[2:4, 2:4] = 0
    inst[H - 6:H, W - 6:W] = n + 2
    inst[H - 4:H - 2, W - 4:W - 2] = 0
    return inst


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('seed', range(6))
def test_fill_holes_matches_whole_frame(seed):
    inst = _label_map(seed)
    np.testing.assert_array_equal(_seg()._fill_holes(inst), ref_fill_holes(inst))


def test_fill_holes_handles_empty_and_gappy_labels():
    seg = _seg()
    empty = np.zeros((12, 12), dtype=np.int32)
    np.testing.assert_array_equal(seg._fill_holes(empty), ref_fill_holes(empty))

    # non-contiguous label ids (find_objects yields None for the missing ones)
    gappy = np.zeros((16, 16), dtype=np.int32)
    gappy[2:8, 2:8] = 3
    gappy[3:6, 3:6] = 0
    gappy[10:14, 10:14] = 9
    np.testing.assert_array_equal(seg._fill_holes(gappy), ref_fill_holes(gappy))


@pytest.mark.parametrize('seed', range(6))
@pytest.mark.parametrize('min_size', [1, 4, 25])
def test_remove_small_components_matches_loop(seed, min_size):
    inst = _label_map(seed)
    seg = _seg(min_component_size=min_size)
    np.testing.assert_array_equal(
        seg._remove_small_components(inst.copy()),
        ref_remove_small_components(seg, inst, min_size),
    )


def test_remove_small_components_does_not_mutate_its_argument():
    """The old loop zeroed labels in place; callers only use the return value."""
    inst = _label_map(0)
    before = inst.copy()
    _seg(min_component_size=25)._remove_small_components(inst)
    np.testing.assert_array_equal(inst, before)


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('merge_max_distance', [0.62, 1.0, 2.0, 3.0])
def test_merge_split_instances_matches_whole_frame(seed, merge_max_distance):
    """Spans merge_max_distance < 1 (tuned default, merges impossible) through 3.0."""
    rng = np.random.default_rng(100 + seed)
    inst = _label_map(seed)
    H, W = inst.shape
    emb = rng.random((H, W, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=2, keepdims=True) + 1e-5
    prob = rng.random((H, W)).astype(np.float32)

    seg = _seg(merge_max_distance=merge_max_distance)
    got = seg._merge_split_instances(inst.copy(), emb, prob)
    want = ref_merge_split_instances(seg, inst.copy(), emb, prob)
    np.testing.assert_array_equal(got, want)


def test_seed_guarantee_picks_the_same_peak_as_a_whole_frame_argmax():
    """Bounding-box seed selection must reproduce the whole-frame argmax, ties included."""
    from scipy import ndimage

    rng = np.random.default_rng(3)
    for trial in range(8):
        prob = rng.random((40, 44)).astype(np.float32)
        if trial % 2:                                   # force exact ties within components
            prob = np.round(prob, 2)
        binary = prob > 0.4
        components, n_components = ndimage.label(binary)
        seeds_binary = np.zeros_like(binary)            # nothing seeded -> every comp needs one

        want = seeds_binary.copy()
        for comp_id in range(1, n_components + 1):
            comp_mask = components == comp_id
            if not want[comp_mask].any():
                peak = np.unravel_index(np.where(comp_mask, prob, -1).argmax(), prob.shape)
                want[peak] = True

        from scipy.ndimage import find_objects
        got = seeds_binary.copy()
        for comp_id, sl in enumerate(find_objects(components), start=1):
            comp_mask = components[sl] == comp_id
            if not got[sl][comp_mask].any():
                scored = np.where(comp_mask, prob[sl], -1.0)
                local = np.unravel_index(scored.argmax(), scored.shape)
                got[local[0] + sl[0].start, local[1] + sl[1].start] = True

        np.testing.assert_array_equal(got, want, err_msg=f'trial {trial}')


def test_embedding_blur_single_call_matches_per_channel_loop():
    """gaussian_filter(sigma=(s, s, 0)) must equal filtering each channel separately."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(5)
    emb = rng.random((32, 36, 12)).astype(np.float32)
    for sigma in (0.5, 1.0, 1.5, 3.0):
        want = np.zeros_like(emb)
        for d in range(emb.shape[2]):
            want[:, :, d] = gaussian_filter(emb[:, :, d], sigma=sigma)
        got = gaussian_filter(emb, sigma=(sigma, sigma, 0))
        np.testing.assert_array_equal(got, want, err_msg=f'sigma={sigma}')


def test_merge_below_one_pixel_cannot_merge():
    """Documents why the tuned merge_max_distance=0.6198 never merges.

    distance_transform_edt(~mask) is 0 on the fragment and >=1 off it, so a threshold
    below 1 selects only the fragment's own pixels — which `& ~mask` then removes. The
    candidate set is always empty, so the step only ever drops small components.
    """
    inst = np.zeros((20, 20), dtype=np.int32)
    inst[4:10, 4:10] = 1
    inst[4:10, 10:16] = 2                      # directly touching fragment
    emb = np.ones((20, 20, 4), dtype=np.float32)
    prob = np.ones((20, 20), dtype=np.float32)

    seg = _seg(merge_max_distance=0.6198, min_component_size=1)
    out = seg._merge_split_instances(inst.copy(), emb, prob)
    assert set(np.unique(out)) == {0, 1, 2}, 'nothing should merge below 1 px'

    # ...whereas at 1.0 the same touching pair does merge (identical embeddings, bright).
    merged = _seg(merge_max_distance=1.0, min_component_size=1)._merge_split_instances(
        inst.copy(), emb, prob)
    assert set(np.unique(merged)) == {0, 1}
