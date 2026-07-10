"""Utilities for 3D mask matching and IOU computation."""

import numpy as np
from scipy.sparse import coo_array
from sklearn.preprocessing import normalize


def filter_small_cells(instances_4d: np.ndarray, min_voxels: int = 200) -> np.ndarray:
    """Remove cells smaller than min_voxels from each timepoint.

    Args:
        instances_4d: [T, Z, H, W] integer label array
        min_voxels:   minimum 3D voxel count to keep a cell

    Returns:
        filtered copy of instances_4d with small labels set to 0
    """
    out = instances_4d.copy()
    for t in range(out.shape[0]):
        flat = out[t].ravel()
        if flat.max() == 0:
            continue
        counts = np.bincount(flat)                                    # index = label
        small  = np.where((counts < min_voxels) & (np.arange(len(counts)) > 0))[0]
        if small.size:
            out[t][np.isin(out[t], small)] = 0
    return out


def label_overlap(x, y, dtype=np.uint16):
    """Compute overlap matrix between two label maps (sparse)."""
    x = x.ravel()
    y = y.ravel()
    z = [1] * len(x)
    return coo_array((z, (x, y)), shape=(int(x.max()) + 1, int(y.max()) + 1), dtype=dtype)


def intersection_over_union(x, y, dtype=np.uint16):
    """Compute IOU between label maps x and y (sparse matrix)."""
    overlap = label_overlap(x, y, dtype=dtype)
    return normalize(overlap, norm='l1', axis=1).astype(np.float32)


def _bridge_label_gaps(masks_list, gap_tolerance, iou_threshold):
    """Reconnect label chains broken by at most gap_tolerance bad slices.

    Leaves gap slices untouched — only renames the label that resumes after
    the gap so it matches the label that ended before it.

    Vectorized: builds a pixel overlap matrix once per (z_last, z_target) pair
    instead of looping over every (label, candidate) combination with boolean masks.
    """
    from scipy.sparse import coo_matrix as sp_coo

    Z = len(masks_list)

    # Find last Z-slice for each label
    label_last_z = {}
    for z, mask in enumerate(masks_list):
        for lbl in np.unique(mask):
            if lbl == 0:
                continue
            label_last_z[int(lbl)] = z

    # Group labels by their last z (only those ending before the final slice)
    ending_at = {}
    for lbl, z in label_last_z.items():
        if z < Z - 1:
            ending_at.setdefault(z, []).append(lbl)

    for z_last in sorted(ending_at.keys()):
        remaining = set(ending_at[z_last])

        for skip in range(1, gap_tolerance + 1):
            if not remaining:
                break
            z_target = z_last + skip + 1
            if z_target >= Z:
                break

            ref = masks_list[z_last]
            tgt = masks_list[z_target]
            if ref.max() == 0 or tgt.max() == 0:
                continue

            # Build pixel overlap matrix in one pass: overlap[i,j] = pixels where ref==i, tgt==j
            rx = ref.ravel().astype(np.int64)
            tx = tgt.ravel().astype(np.int64)
            n_ref = int(rx.max()) + 1
            n_tgt = int(tx.max()) + 1

            overlap = sp_coo(
                (np.ones(len(rx), dtype=np.int32), (rx, tx)),
                shape=(n_ref, n_tgt),
            ).toarray()  # [n_ref, n_tgt]

            size_ref = overlap.sum(axis=1)  # [n_ref]
            size_tgt = overlap.sum(axis=0)  # [n_tgt]

            # True IOU: overlap / (size_ref + size_tgt - overlap)
            union = size_ref[:, None] + size_tgt[None, :] - overlap
            with np.errstate(divide='ignore', invalid='ignore'):
                iou = np.where(union > 0, overlap / union, 0.0)
            iou[0, :] = 0.0  # ignore background row/col
            iou[:, 0] = 0.0

            claimed = set()
            bridged = set()
            for lbl in remaining:
                if lbl >= n_ref:
                    continue
                best_col = int(iou[lbl].argmax())
                if best_col == 0 or iou[lbl, best_col] <= iou_threshold or best_col in claimed:
                    continue
                claimed.add(best_col)
                bridged.add(lbl)
                for z in range(z_target, Z):
                    masks_list[z][masks_list[z] == best_col] = lbl

            remaining -= bridged

    return masks_list


def match_masks_3d(masks_3d, stitch_threshold=0.0, gap_tolerance=1, gap_iou_threshold=0.3, dtype=None):
    """
    Match labels across Z dimension using IOU overlap.
    Connects instances across consecutive slices via optimal matching.

    Args:
        masks_3d:          [Z, H, W] array or list of [H, W] arrays
        stitch_threshold:  min IOU to match consecutive slices (default 0.0)
        gap_tolerance:     bridge chains broken by up to this many bad slices (default 1, 0 to disable)
        gap_iou_threshold: min IOU to accept a gap bridge (default 0.3)
        dtype:             output dtype

    Returns:
        masks_3d_matched: [Z, H, W] with consistent labels across Z
    """
    if not isinstance(masks_3d, np.ndarray):
        masks_3d = np.stack(masks_3d, axis=0)
    Z = masks_3d.shape[0]
    masks_list = [masks_3d[z].copy() for z in range(Z)]

    if dtype is None:
        dtype = masks_3d.dtype

    # Normalize labels to be contiguous from 1
    mmin = min([x[x > 0].min() - 1 if np.any(x) else 0 for x in masks_list])
    for i in range(len(masks_list)):
        masks_list[i][masks_list[i] > 0] = masks_list[i][masks_list[i] > 0] - mmin

    mmax = max([x.max() for x in masks_list if x.size > 0])

    # Pass 1: match consecutive slices
    for i in range(len(masks_list) - 1):
        iou = intersection_over_union(masks_list[i + 1], masks_list[i])[1:, 1:]

        if not iou.size:
            continue

        n_next = iou.shape[0]  # cells in slice i+1
        iou = iou > stitch_threshold
        x = np.array(iou.argmax(axis=0))
        if len(x.shape) > 1:
            x = x[0, :]

        y = np.arange(0, x.size, 1, dtype=dtype)
        z = iou.max(axis=0).toarray()
        if len(z.shape) > 1:
            z = z[0, :]

        iou = coo_array((z, (x, y)), shape=(n_next, len(y)))

        istitch = iou.argmax(axis=1)
        if hasattr(istitch, 'A'):
            istitch = istitch.A.ravel() + 1
        else:
            istitch = istitch.ravel() + 1

        ino = np.nonzero(iou.max(axis=1).toarray() == 0.0)[0]
        istitch[ino] = np.arange(mmax + 1, mmax + len(ino) + 1, 1, dtype=dtype)
        mmax += len(ino)
        istitch = np.append(np.array(0), istitch)

        masks_list[i + 1] = istitch[masks_list[i + 1]]

    # Pass 2: bridge single-slice gaps
    if gap_tolerance > 0:
        masks_list = _bridge_label_gaps(masks_list, gap_tolerance, gap_iou_threshold)

    # Restore original label range
    for i in range(len(masks_list)):
        masks_list[i][masks_list[i] > 0] = masks_list[i][masks_list[i] > 0] + mmin

    masks_3d_matched = np.stack(masks_list, axis=0)
    return masks_3d_matched.astype(dtype)
