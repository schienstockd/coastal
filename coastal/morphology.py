"""Cell morphology: 2D polygon extraction and shape descriptors (Z-max projection).

NOTE: The polygon + shape-feature helpers in this module are RETAINED as a
standalone morphology / QC readout utility. They are NOT currently wired into
the segmentation or tracking pipelines, and are kept for potential future use.
The former HMM boundary-state code path was a tried-and-failed tracking
direction and has been removed.
"""

import numpy as np
from multiprocessing import Pool
from skimage import measure


# --------------------------------------------------------------------------- #
# Polygon extraction (Z-max projection)                                       #
# --------------------------------------------------------------------------- #

def _worker_polygon(args):
    lab, ys, xs, Y, X, min_area, min_coords = args
    import numpy as np
    from skimage import measure
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union

    if len(ys) < 3:
        return None
    mask = np.zeros((Y, X), dtype=np.uint8)
    mask[ys, xs] = 1
    contours = measure.find_contours(mask, level=0.5)
    polys = []
    for c in contours:
        if len(c) < 3:
            continue
        coords = [(float(x), float(y)) for y, x in c]
        p = Polygon(coords)
        if not p.is_valid:
            p = p.buffer(0)
        if isinstance(p, MultiPolygon):
            p = max(p.geoms, key=lambda g: g.area)
        if p.is_valid and p.area >= min_area and len(p.exterior.coords) > min_coords:
            polys.append(p)
    if not polys:
        return None
    poly = unary_union(polys) if len(polys) > 1 else polys[0]
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if not isinstance(poly, Polygon) or poly.area < min_area:
        return None
    return poly


def labels_to_polygons(instances_4d, n_workers=8, min_area=10, min_coords=8):
    """Extract 2D polygons from Z-max projection of each labeled cell.

    Args:
        instances_4d: [T, Z, H, W] integer label array
        n_workers:    parallel workers for contour extraction
        min_area:     minimum polygon area in pixels
        min_coords:   minimum number of polygon vertices

    Returns:
        dict {t: {cell_id: shapely.Polygon}}
    """
    T, Z, H, W = instances_4d.shape
    result = {}

    for t in range(T):
        proj = instances_4d[t].max(axis=0)  # [H, W]
        labels = np.unique(proj)
        labels = labels[labels > 0]

        if len(labels) == 0:
            result[t] = {}
            continue

        ys_all, xs_all = np.nonzero(proj)
        labs_all = proj[ys_all, xs_all]
        order = np.argsort(labs_all)
        labs_s = labs_all[order]
        ys_s = ys_all[order]
        xs_s = xs_all[order]

        unique_labs, starts = np.unique(labs_s, return_index=True)
        ends = np.empty_like(starts)
        ends[:-1] = starts[1:]
        ends[-1] = len(labs_s)

        args_list = [
            (int(unique_labs[i]),
             ys_s[starts[i]:ends[i]],
             xs_s[starts[i]:ends[i]],
             H, W, min_area, min_coords)
            for i in range(len(unique_labs))
        ]

        if n_workers > 1:
            with Pool(n_workers) as pool:
                polys = pool.map(_worker_polygon, args_list)
        else:
            polys = [_worker_polygon(a) for a in args_list]

        result[t] = {
            int(unique_labs[i]): polys[i]
            for i in range(len(unique_labs))
            if polys[i] is not None
        }

    return result


# --------------------------------------------------------------------------- #
# Shape features (2D regionprops + 5 derived)                                #
# --------------------------------------------------------------------------- #

_STANDARD_PROPS = [
    'area', 'area_convex', 'area_filled',
    'axis_major_length', 'axis_minor_length',
    'eccentricity', 'equivalent_diameter_area',
    'euler_number', 'extent',
    'orientation', 'perimeter', 'perimeter_crofton',
    'solidity',
]  # 13 standard scalar properties

DERIVED_SHAPE_FEATURES = ['oblate', 'prolate', 'aspect_ratio', 'perimeter_to_area', 'fill']

SHAPE_FEATURE_NAMES = _STANDARD_PROPS + DERIVED_SHAPE_FEATURES  # 18 total


def _shape_features_from_proj(proj_2d):
    """Compute shape features for all labeled cells in a 2D label map.

    Returns dict {cell_id: np.ndarray [18]}
    """
    if proj_2d.max() == 0:
        return {}

    props = measure.regionprops_table(proj_2d, properties=['label'] + _STANDARD_PROPS)

    labels = props['label']
    eps = 1e-8

    major = props['axis_major_length']
    minor = props['axis_minor_length']
    equiv = props['equivalent_diameter_area']
    perim = props['perimeter']
    area = props['area']
    convex_area = props['area_convex']

    oblate            = minor / (major + eps)
    prolate           = major / (minor + eps)
    aspect_ratio      = major / (equiv + eps)
    perimeter_to_area = (perim ** 2) / (area + eps)
    fill              = (convex_area - area) / (convex_area + eps)

    std_feats = np.column_stack([props[k] for k in _STANDARD_PROPS]).astype(np.float32)
    derived   = np.column_stack([oblate, prolate, aspect_ratio,
                                  perimeter_to_area, fill]).astype(np.float32)
    all_feats = np.concatenate([std_feats, derived], axis=1)  # [N, 18]

    return {int(lab): all_feats[i] for i, lab in enumerate(labels)}


def extract_shape_features(instances_4d):
    """Compute shape features for all cells at each timepoint (Z-max projection).

    Returns:
        shape_feats: dict {t: {cell_id: np.ndarray [18]}}
    """
    T = instances_4d.shape[0]
    result = {}
    for t in range(T):
        proj = instances_4d[t].max(axis=0)
        result[t] = _shape_features_from_proj(proj)
    return result


# --------------------------------------------------------------------------- #
# Combined morphology extraction                                              #
# --------------------------------------------------------------------------- #

def extract_cell_morphology(instances_4d, polygons=None):
    """Per-cell 2D shape-feature readout per timepoint (Z-max projection).

    Standalone morphology / QC utility — NOT wired into the segmentation or
    tracking pipelines (see the module note). The former HMM boundary-state
    features have been removed; this now returns shape features plus the
    cell's polygon when one is supplied.

    Args:
        instances_4d: [T, Z, H, W] label array
        polygons:     optional output of labels_to_polygons(); when provided the
                      cell's polygon is attached alongside its shape features.

    Returns:
        morphology: dict {t: {cell_id: {
            'shape_feats': np.ndarray [18],
            'polygon':     shapely.Polygon or None,
        }}}
    """
    T = instances_4d.shape[0]
    shape_feats_all = extract_shape_features(instances_4d)
    n_shape = len(SHAPE_FEATURE_NAMES)

    morphology = {}
    for t in range(T):
        morphology[t] = {}
        polys_t  = polygons.get(t, {}) if polygons is not None else {}
        shape_t  = shape_feats_all.get(t, {})
        cell_ids = set(polys_t.keys()) | set(shape_t.keys())

        for cell_id in cell_ids:
            shape = shape_t.get(cell_id)
            if shape is None:
                shape = np.zeros(n_shape, dtype=np.float32)
            morphology[t][cell_id] = {
                'shape_feats': shape.astype(np.float32),
                'polygon':     polys_t.get(cell_id),
            }

    return morphology
