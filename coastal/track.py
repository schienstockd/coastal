"""Cell tracking utilities: data structures and feature extraction."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from skimage.measure import regionprops_table


# --------------------------------------------------------------------------- #
# Data structures                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class Track:
    """Single cell trajectory in pixel space."""
    track_id: int
    color_id: int                        # dominant confetti channel (0–n_ch-1), -1 = dim
    centroids_px: Dict[int, np.ndarray]  # t -> [z, y, x] in pixels
    cell_ids:     Dict[int, int]         # t -> cell_id in instances_4d

    def timepoints(self) -> List[int]:
        return sorted(self.centroids_px.keys())

    def centroids_um(self, pix_res: dict) -> Dict[int, np.ndarray]:
        scale = np.array([pix_res['z'], pix_res['y'], pix_res['x']], dtype=np.float32)
        return {t: c * scale for t, c in self.centroids_px.items()}


# --------------------------------------------------------------------------- #
# 3D centroid computation                                                      #
# --------------------------------------------------------------------------- #

def compute_3d_centroids(instances_4d: np.ndarray) -> Dict[int, Dict[int, np.ndarray]]:
    """3D centroids for all cells at each timepoint.

    Returns:
        {t: {cell_id: np.ndarray([z, y, x])}}
    """
    T = instances_4d.shape[0]
    result = {}
    for t in range(T):
        vol = instances_4d[t]
        if vol.max() == 0:
            result[t] = {}
            continue
        props = regionprops_table(vol, properties=['label', 'centroid'])
        result[t] = {
            int(lab): np.array([cz, cy, cx], dtype=np.float32)
            for lab, cz, cy, cx in zip(
                props['label'],
                props['centroid-0'],
                props['centroid-1'],
                props['centroid-2'],
            )
        }
    return result


# --------------------------------------------------------------------------- #
# Color / intensity extraction                                                 #
# --------------------------------------------------------------------------- #

def extract_cell_colors(
    instances_4d: np.ndarray,
    volumes: np.ndarray,
    ch_indices: List[int],
    dim_quantile: float = 0.1,
) -> Dict[int, Dict[int, int]]:
    """Assign confetti color ID to each cell at each timepoint.

    Cells below the dim_quantile brightness threshold (per movie) are assigned -1.

    Args:
        instances_4d: [T, Z, H, W] label array
        volumes:      [T, C, Z, Y, X] raw image data
        ch_indices:   channel indices corresponding to confetti colors
        dim_quantile: cells below this brightness quantile are excluded (per movie)

    Returns:
        {t: {cell_id: int}}  — int is 0..n_ch-1 or -1
    """
    T = instances_4d.shape[0]
    n_ch = len(ch_indices)

    all_max_intens: List[float] = []
    cell_intens: Dict = {}  # (t, cell_id) -> np.ndarray [n_ch]

    for t in range(T):
        labels_flat = instances_4d[t].ravel().astype(np.int32)
        n_labels = int(labels_flat.max()) + 1 if labels_flat.max() > 0 else 1
        counts = np.bincount(labels_flat, minlength=n_labels).astype(np.float32)

        vol_t = np.asarray(volumes[t][ch_indices], dtype=np.float32)  # [n_ch, Z, H, W]

        mean_intens = np.zeros((n_labels, n_ch), dtype=np.float32)
        for ci in range(n_ch):
            mean_intens[:, ci] = np.bincount(
                labels_flat, weights=vol_t[ci].ravel(), minlength=n_labels
            )

        valid = counts > 0
        mean_intens[valid] /= counts[valid, None]

        for lab in np.unique(labels_flat):
            if lab == 0:
                continue
            mi = mean_intens[lab]
            cell_intens[(t, int(lab))] = mi
            all_max_intens.append(float(mi.max()))

    threshold = float(np.quantile(all_max_intens, dim_quantile)) if all_max_intens else 0.0

    color_ids: Dict[int, Dict[int, int]] = {}
    n_excluded = 0

    for t in range(T):
        color_ids[t] = {}
        for lab in np.unique(instances_4d[t]):
            if lab == 0:
                continue
            mi = cell_intens.get((t, int(lab)))
            if mi is None or float(mi.max()) <= threshold:
                color_ids[t][int(lab)] = -1
                n_excluded += 1
            else:
                color_ids[t][int(lab)] = int(mi.argmax())

    print(f"Color assignment: {n_excluded} dim cells excluded "
          f"(dim_quantile={dim_quantile}, threshold={threshold:.4f})")
    return color_ids


def extract_cell_intensities(
    instances_4d: np.ndarray,
    volumes: np.ndarray,
    ch_indices: List[int],
) -> Dict[int, Dict[int, np.ndarray]]:
    """Mean confetti channel intensity per cell per timepoint.

    Args:
        instances_4d: [T, Z, H, W] label array
        volumes:      [T, C, Z, Y, X] raw image data
        ch_indices:   confetti channel indices

    Returns:
        {t: {cell_id: np.ndarray [n_ch]}}
    """
    T   = instances_4d.shape[0]
    n_ch = len(ch_indices)
    result: Dict[int, Dict[int, np.ndarray]] = {}

    for t in range(T):
        flat    = instances_4d[t].ravel().astype(np.int32)
        n_lab   = int(flat.max()) + 1 if flat.max() > 0 else 1
        counts  = np.bincount(flat, minlength=n_lab).astype(np.float32)
        vol_t   = np.asarray(volumes[t][ch_indices], dtype=np.float32)

        mi = np.zeros((n_lab, n_ch), np.float32)
        for ci in range(n_ch):
            mi[:, ci] = np.bincount(flat, weights=vol_t[ci].ravel(), minlength=n_lab)

        valid = counts > 0
        mi[valid] /= counts[valid, None]

        result[t] = {int(lab): mi[lab] for lab in np.unique(flat) if lab > 0}

    return result
