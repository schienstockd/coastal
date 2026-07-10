"""Cell morphology: boundary HMM states and 2D shape descriptors."""

import warnings
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter1d
from skimage import measure
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm as hmmlearn_hmm
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


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
# Boundary feature extraction                                                 #
# --------------------------------------------------------------------------- #

def _signed_curvature(X, Y):
    dx = np.diff(X)
    dy = np.diff(Y)
    ddx = np.diff(dx)
    ddy = np.diff(dy)
    curvature = np.zeros(len(dx), dtype=np.float32)
    denom = (dx[:-1] ** 2 + dy[:-1] ** 2) ** 1.5
    valid = denom > 0
    curvature[1:][valid] = (ddx * dy[:-1] - dx[:-1] * ddy)[valid] / denom[valid]
    return curvature


def extract_boundary_features(polygon, sigma=2.0):
    """Per-segment boundary features: distance, angle_change, fold_score.

    Returns:
        np.ndarray [N_segments, 3] or None if polygon is too small
    """
    xs, ys = polygon.exterior.xy
    X = gaussian_filter1d(np.array(xs, dtype=np.float32), sigma)
    Y = gaussian_filter1d(np.array(ys, dtype=np.float32), sigma)

    if len(X) < 4:
        return None

    dx = np.diff(X)
    dy = np.diff(Y)
    L = len(dx)

    distances = np.sqrt(dx ** 2 + dy ** 2)

    # Absolute angle change: how sharply the boundary turns, regardless of direction.
    # 0 padded at start (first segment has no predecessor).
    angles = np.degrees(np.arctan2(dy, dx))
    wrapped = ((np.diff(angles) + 180.0) % 360.0) - 180.0
    angle_change = np.empty(L, dtype=np.float32)
    angle_change[0] = 0.0
    angle_change[1:] = np.abs(wrapped)

    # Signed fold score: positive = folding inward, negative = folding outward.
    # Corrected for polygon winding direction, then tanh-normalized per polygon
    # using the 90th-percentile as scale so values are bounded in (-1, 1) and
    # comparable across cells of different sizes.
    curvature = _signed_curvature(X, Y)[:L]

    # Determine winding (positive signed area = CCW in image coords)
    Xa = X[:-1] if (np.isclose(X[0], X[-1]) and np.isclose(Y[0], Y[-1])) else X
    Ya = Y[:-1] if (np.isclose(X[0], X[-1]) and np.isclose(Y[0], Y[-1])) else Y
    signed_area = 0.5 * float(np.sum(Xa * np.roll(Ya, -1) - np.roll(Xa, -1) * Ya))
    sign_to_interior = 1.0 if signed_area > 0 else -1.0

    raw_fold = sign_to_interior * curvature
    abs_fold  = np.abs(raw_fold)
    scale = float(np.percentile(abs_fold, 90)) if abs_fold.size else 0.0
    if scale <= 0:
        scale = float(abs_fold.max()) if abs_fold.max() > 0 else 1.0
    fold_score = np.tanh(raw_fold / scale).astype(np.float32)
    fold_score[np.abs(fold_score) < 1e-3] = 0.0

    return np.stack([distances, angle_change, fold_score], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# HMM fitting                                                                 #
# --------------------------------------------------------------------------- #

def fit_boundary_hmm(polygons, n_states=4, train_fraction=0.8, max_polygons=3000,
                     sigma=2.0, random_state=42):
    """Fit Gaussian HMM on boundary segment features across all polygons.

    Args:
        polygons:       dict {t: {cell_id: Polygon}} or flat list of Polygons
        n_states:       number of HMM hidden states (tunable)
        train_fraction: fraction of polygons used for fitting
        max_polygons:   hard cap on training polygons (HMM saturates quickly;
                        3000 is ample for 4 states)
        sigma:          boundary smoothing sigma
        random_state:   RNG seed

    Returns:
        (hmm_model, scaler)
    """
    if isinstance(polygons, dict):
        poly_list = [p for t_dict in polygons.values() for p in t_dict.values()]
    else:
        poly_list = list(polygons)

    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(poly_list))
    n_train = max(1, min(int(len(poly_list) * train_fraction), max_polygons))
    train_polys = [poly_list[int(i)] for i in idx[:n_train]]

    all_feats, lengths = [], []
    for poly in train_polys:
        feat = extract_boundary_features(poly, sigma=sigma)
        if feat is None or len(feat) < 2:
            continue
        all_feats.append(feat)
        lengths.append(len(feat))

    if not all_feats:
        raise ValueError("No valid polygons for HMM training")

    X_raw = np.vstack(all_feats)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    model = hmmlearn_hmm.GaussianHMM(
        n_components=n_states,
        covariance_type='diag',
        n_iter=100,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_scaled, lengths=lengths)

    print(f"HMM: {n_states} states fitted on {len(all_feats)} polygons "
          f"({len(X_scaled)} boundary segments)")
    return model, scaler


def assign_boundary_states(
    polygon,
    hmm_model,
    scaler,
    sigma:                float = 2.0,
    apply_state_smoothing: bool = True,
    median_w:              int  = 3,
    min_run:               int  = 3,
) -> np.ndarray:
    """Return per-state proportions for a polygon as a float array [n_states] summing to 1.

    Returns zeros if the polygon is too small or invalid.
    """
    n = hmm_model.n_components
    feat = extract_boundary_features(polygon, sigma=sigma)
    if feat is None or len(feat) < 2:
        return np.zeros(n, dtype=np.float32)
    X = scaler.transform(feat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        states = hmm_model.predict(X)
    if apply_state_smoothing:
        states = median_filter_states(states, w=median_w)
        states = enforce_min_run_length(states, min_len=min_run)
    counts = np.bincount(states, minlength=n).astype(np.float32)
    return counts / counts.sum()


def hmm_feature_dim(n_states: int) -> int:
    """Total HMM feature vector length for a given n_states.

    Components: proportions [n] + self_transitions [n] + run_mean [n]
    """
    return n_states * 3


def _hmm_sequence_features(states: np.ndarray, n_states: int) -> np.ndarray:
    """Compact HMM features from the ordered boundary state sequence.

    All components are rotation-invariant (closed-loop):
    - Proportions      [n_states] — fraction of boundary in each state
    - Self-transitions [n_states] — P(stay | current state), diagonal of transition matrix
    - Run-length mean  [n_states] — mean contiguous run per state / seq_len
    """
    n = len(states)
    counts = np.bincount(states, minlength=n_states).astype(np.float32)
    proportions = counts / max(float(counts.sum()), 1.0)

    if n < 2:
        return np.concatenate([
            proportions,
            np.zeros(n_states, dtype=np.float32),  # self-transitions
            np.zeros(n_states, dtype=np.float32),  # run_mean
        ])

    # Transition matrix — closed loop (last → first); only diagonal is kept
    trans = np.zeros((n_states, n_states), dtype=np.float32)
    for i in range(n - 1):
        trans[int(states[i]), int(states[i + 1])] += 1.0
    trans[int(states[-1]), int(states[0])] += 1.0
    row_sums = trans.sum(axis=1, keepdims=True)
    trans /= np.where(row_sums > 0, row_sums, 1.0)
    self_transitions = np.diag(trans)  # [n_states] — stickiness per state

    # Mean run length, normalised by sequence length
    rl_mean = np.zeros(n_states, dtype=np.float32)
    runs: list = [[] for _ in range(n_states)]
    run_s, run_l = int(states[0]), 1
    for s in states[1:]:
        s = int(s)
        if s == run_s:
            run_l += 1
        else:
            runs[run_s].append(run_l)
            run_s, run_l = s, 1
    runs[run_s].append(run_l)
    for s in range(n_states):
        if runs[s]:
            rl_mean[s] = float(np.mean(runs[s])) / n

    return np.concatenate([proportions, self_transitions, rl_mean])


def median_filter_states(states: np.ndarray, w: int = 3) -> np.ndarray:
    """Sliding-window mode filter: replace each state with the most common in a window of width w."""
    half = w // 2
    n = len(states)
    out = states.copy()
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        vals, counts = np.unique(states[lo:hi], return_counts=True)
        out[i] = vals[counts.argmax()]
    return out


def enforce_min_run_length(states: np.ndarray, min_len: int = 3) -> np.ndarray:
    """Absorb runs shorter than min_len into the longer of their two neighboring runs.

    Repeats until all runs are >= min_len (or there is only one run left).
    """
    states = states.copy()
    changed = True
    while changed:
        changed = False
        # Build run list: (start, end, state)
        runs = []
        i = 0
        while i < len(states):
            s = int(states[i])
            j = i
            while j < len(states) and int(states[j]) == s:
                j += 1
            runs.append((i, j, s))
            i = j
        for ri, (start, end, s) in enumerate(runs):
            rlen = end - start
            if rlen >= min_len:
                continue
            prev_len = runs[ri - 1][1] - runs[ri - 1][0] if ri > 0 else 0
            next_len = runs[ri + 1][1] - runs[ri + 1][0] if ri < len(runs) - 1 else 0
            if prev_len == 0 and next_len == 0:
                break
            if next_len >= prev_len:
                merge_s = runs[ri + 1][2]
            else:
                merge_s = runs[ri - 1][2]
            states[start:end] = merge_s
            changed = True
            break   # restart run detection after each merge
    return states


def assign_boundary_hmm_features(
    polygon,
    hmm_model,
    scaler,
    sigma:                float = 2.0,
    apply_state_smoothing: bool = True,
    median_w:              int  = 3,
    min_run:               int  = 3,
) -> np.ndarray:
    """Full HMM feature vector for a polygon.

    Returns hmm_feature_dim(n_states) values:
    proportions + row-normalised transition matrix + run-length mean + run-length max.
    Returns zeros if the polygon is too small or invalid.

    apply_state_smoothing: apply median_filter_states then enforce_min_run_length before
        computing the feature vector (cleans single-segment state noise).
    """
    n = hmm_model.n_components
    feat = extract_boundary_features(polygon, sigma=sigma)
    if feat is None or len(feat) < 2:
        return np.zeros(hmm_feature_dim(n), dtype=np.float32)
    X = scaler.transform(feat)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        states = hmm_model.predict(X)
    if apply_state_smoothing:
        states = median_filter_states(states, w=median_w)
        states = enforce_min_run_length(states, min_len=min_run)
    return _hmm_sequence_features(states, n)


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

def extract_cell_morphology(
    instances_4d,
    polygons,
    hmm_model,
    scaler,
    n_states:              int   = 4,
    sigma:                 float = 2.0,
    apply_state_smoothing: bool  = True,
    median_w:              int   = 3,
    min_run:               int   = 3,
):
    """Combine HMM features and shape features per cell per timepoint.

    Args:
        instances_4d:          [T, Z, H, W] label array
        polygons:              output of labels_to_polygons()
        hmm_model:             fitted HMM from fit_boundary_hmm()
        scaler:                StandardScaler from fit_boundary_hmm()
        n_states:              number of HMM states (must match hmm_model)
        sigma:                 boundary smoothing sigma
        apply_state_smoothing: smooth raw HMM state sequence before feature extraction
        median_w:              mode-filter window width
        min_run:               minimum run length to keep (shorter runs are merged)

    Returns:
        morphology: dict {t: {cell_id: {
            'hmm_feats':   np.ndarray [hmm_feature_dim(n_states)]
                           — proportions + transition matrix + run-length stats,
            'shape_feats': np.ndarray [18]
        }}}
    """
    T = instances_4d.shape[0]
    shape_feats_all = extract_shape_features(instances_4d)
    n_shape   = len(SHAPE_FEATURE_NAMES)
    n_hmm     = hmm_feature_dim(n_states)

    morphology = {}
    for t in range(T):
        morphology[t] = {}
        polys_t  = polygons.get(t, {})
        shape_t  = shape_feats_all.get(t, {})
        cell_ids = set(polys_t.keys()) | set(shape_t.keys())

        for cell_id in cell_ids:
            poly = polys_t.get(cell_id)
            hmm_feats = (
                assign_boundary_hmm_features(
                    poly, hmm_model, scaler, sigma,
                    apply_state_smoothing=apply_state_smoothing,
                    median_w=median_w, min_run=min_run,
                )
                if poly is not None
                else np.zeros(n_hmm, dtype=np.float32)
            )

            shape = shape_t.get(cell_id)
            if shape is None:
                shape = np.zeros(n_shape, dtype=np.float32)

            morphology[t][cell_id] = {
                'hmm_feats':   hmm_feats,
                'shape_feats': shape.astype(np.float32),
            }

    return morphology
