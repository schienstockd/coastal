"""CMA-ES parameter optimization for segmentation using channel purity as objective."""

import numpy as np
import cma

from coastal.segment import LearnedAffinityInference


# Parameters included in the search and their valid ranges.
PARAM_NAMES = [
    # 'prob_threshold',
    'affinity_threshold',
    'merge_affinity_threshold',
    'merge_max_distance',
    'merge_contact_brightness_threshold',
    'prob_weight',
]

PARAM_BOUNDS = {
    # 'prob_threshold':                      (0.2, 0.5),
    'affinity_threshold':                  (0.2, 0.6),
    'merge_affinity_threshold':            (0.2, 0.6),
    'merge_max_distance':                  (0.5, 3.0),
    'merge_contact_brightness_threshold':  (0.2, 0.6),
    'prob_weight':                         (0.0, 0.6),
}

_LO = np.array([PARAM_BOUNDS[n][0] for n in PARAM_NAMES])
_HI = np.array([PARAM_BOUNDS[n][1] for n in PARAM_NAMES])


def _vec_to_params(x):
    return {name: float(np.clip(x[i], _LO[i], _HI[i]))
            for i, name in enumerate(PARAM_NAMES)}


def score_segmentation(results, frames_multi, min_cell_size=100,
                       purity_threshold=0.7, count_penalty_weight=0.0,
                       verbose=False):
    """
    Score segmentation quality using channel purity and a total cell count penalty.

    Labels are classified as:
      - good:       large (>= min_cell_size) AND dominant channel >= purity_threshold
      - merged:     large but impure — multiple channel types blended
      - fragmented: too small (< min_cell_size)

    Purity is computed on large cells only. The count penalty applies to ALL cells
    (large + fragments) so the optimizer discourages both oversegmentation into large
    pieces and into small fragments equally.

    Frame score = purity - count_penalty_weight * (n_large + n_fragmented)

    Args:
        results:              list of result dicts from predict_sequence / predict_frame
        frames_multi:         [T, C, H, W] raw multi-channel frames (uint8 or float)
        min_cell_size:        pixel threshold below which a label is "fragmented" (default 100)
        purity_threshold:     dominant-channel fraction above which a cell is "good" (default 0.7)
        count_penalty_weight: penalty per cell found, large or fragment (default 0.0 = off)
        verbose:              print per-frame counts (default False)

    Returns:
        scalar score, higher is better
    """
    frames_arr = np.asarray(frames_multi, dtype=np.float32)
    if frames_arr.ndim == 3:
        raise ValueError("frames_multi must be [T, C, H, W], not [T, H, W]")
    T, C, H, W = frames_arr.shape

    frame_scores = []

    for t, result in enumerate(results):
        if t >= T:
            continue

        instances = result['instances']
        frame = frames_arr[t]  # [C, H, W]

        ch_mean = frame.mean(axis=(1, 2))  # [C]
        frame_norm = frame / (ch_mean[:, None, None] + 1e-6)

        labels = np.unique(instances)
        labels = labels[labels > 0]
        if len(labels) == 0:
            continue

        n_good = 0
        n_merged = 0
        n_fragmented = 0

        for label in labels:
            mask = instances == label
            size = int(mask.sum())

            if size < min_cell_size:
                n_fragmented += 1
                continue

            mean_ch = frame_norm[:, mask].mean(axis=1)  # [C]
            total_intensity = mean_ch.sum()

            if total_intensity < 1e-6:
                n_fragmented += 1
                continue

            purity = float((mean_ch / total_intensity).max())

            if purity >= purity_threshold:
                n_good += 1
            else:
                n_merged += 1

        n_large = n_good + n_merged
        n_total = n_large + n_fragmented
        purity_score = n_good / n_large if n_large > 0 else 0.0
        count_penalty = count_penalty_weight * n_total

        score = purity_score - count_penalty
        frame_scores.append(score)

        if verbose:
            print(f"  Frame {t}: {n_total} total | "
                  f"{n_good} good | {n_merged} merged | {n_fragmented} fragmented | "
                  f"purity={purity_score:.3f} | count_penalty={count_penalty:.3f} | "
                  f"score={score:.3f}")

    return float(np.mean(frame_scores)) if frame_scores else 0.0


def optimize_segmentation_cma(
    model,
    frames,
    frames_multi,
    temporal_metrics,
    x0=None,
    sigma0=0.15,
    max_evals=150,
    n_frames=5,
    min_cell_size=100,
    purity_threshold=0.7,
    count_penalty_weight=0.0,
    fixed_params=None,
    device=None,
):
    """
    CMA-ES optimization of LearnedAffinityInference parameters.

    Optimizes 5 core parameters to maximise the fraction of cells classified as
    "good" (large and single-channel-dominant).  Each cell is labelled:
      - good:       >= min_cell_size px AND dominant channel >= purity_threshold
      - merged:     large but impure (multiple channel types blended)
      - fragmented: too small to be a real cell

    Score = n_good / (n_good + n_merged + n_fragmented) averaged over n_frames.

    Args:
        model:            trained UNet
        frames:           [T, H, W] max-projected frames used for segmentation
        frames_multi:     [T, C, H, W] raw multi-channel frames used for scoring
        temporal_metrics: list of T metric dicts
        x0:               initial parameter vector (None → mid-range)
        sigma0:           initial CMA-ES step size (default 0.15)
        max_evals:        evaluation budget (default 150)
        n_frames:         frames evaluated per candidate (evenly spaced, default 5)
        min_cell_size:    pixel threshold for "fragmented" label (default 100)
        purity_threshold: dominant-channel fraction to call a large cell "good" (default 0.7)
        fixed_params:     additional LearnedAffinityInference kwargs held fixed
        device:           torch device; None/'auto' → cuda→mps→cpu (resolved by the inferencer)

    Returns:
        best_params:  dict of best parameters found (pass directly to LearnedAffinityInference)
        history:      list of (params_dict, score) tuples, one per evaluation
    """
    if x0 is None:
        x0 = list(0.5 * (_LO + _HI))

    fixed = fixed_params or {}

    T = len(frames)
    eval_idx = np.linspace(0, T - 1, min(n_frames, T), dtype=int)
    eval_frames = frames[eval_idx]
    eval_frames_multi = np.asarray(frames_multi)[eval_idx]
    eval_temporal = [temporal_metrics[i] for i in eval_idx]

    history = []
    eval_count = [0]

    def objective(x):
        params = _vec_to_params(x)
        params.update(fixed)

        segmentor = LearnedAffinityInference(model=model, device=device, **params)
        results = segmentor.predict_sequence(eval_frames, eval_temporal)
        score = score_segmentation(
            results, eval_frames_multi,
            min_cell_size=min_cell_size,
            purity_threshold=purity_threshold,
            count_penalty_weight=count_penalty_weight,
        )
        history.append((_vec_to_params(x), score))
        eval_count[0] += 1
        return -score  # CMA-ES minimizes

    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            'maxfevals': max_evals,
            'bounds': [_LO.tolist(), _HI.tolist()],
            'BoundaryHandler': cma.BoundPenalty,
            'verbose': -9,
        },
    )

    iteration = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)

        iteration += 1
        best_f = min(fitnesses)
        best_x = solutions[np.argmin(fitnesses)]
        best_p = _vec_to_params(best_x)
        print(
            f"Iter {iteration:3d} ({eval_count[0]:4d} evals) | "
            f"score={-best_f:.4f} | "
            + " | ".join(f"{n}={best_p[n]:.3f}" for n in PARAM_NAMES)
        )

    best_entry = max(history, key=lambda h: h[1])
    best_params = best_entry[0]
    best_params.update(fixed)

    print(f"\nOptimization complete ({eval_count[0]} evaluations)")
    print(f"Best score: {best_entry[1]:.4f}")
    print("Best parameters:")
    for name in PARAM_NAMES:
        print(f"  {name}: {best_params[name]:.4f}")

    return best_params, history


# --------------------------------------------------------------------------- #
# Tracking parameter optimisation                                              #
# --------------------------------------------------------------------------- #

# Only the parameters the settled track_sequence actually accepts (see DEAD_ENDS.md — the
# w_collective/w_persistence/w_exclusion/cost_appear/cost_disappear terms were removed).
TRACKING_PARAM_NAMES = [
    'chi2_gate', 'process_noise', 'obs_noise',
    'w_flow', 'w_color', 'max_cost', 'momentum_decay',
]

TRACKING_PARAM_BOUNDS = {
    'chi2_gate':       (2.0,  20.0),
    'process_noise':   (0.1,  10.0),
    'obs_noise':       (1.0,  50.0),
    'w_flow':          (0.0,   1.0),
    'w_color':         (0.0,   2.0),
    'max_cost':        (0.5,   2.0),
    'momentum_decay':  (0.5,  0.99),
}

_TLO = np.array([TRACKING_PARAM_BOUNDS[n][0] for n in TRACKING_PARAM_NAMES])
_THI = np.array([TRACKING_PARAM_BOUNDS[n][1] for n in TRACKING_PARAM_NAMES])


def _vec_to_tracking_params(x):
    return {name: float(np.clip(x[i], _TLO[i], _THI[i]))
            for i, name in enumerate(TRACKING_PARAM_NAMES)}


def score_tracking_scalar(metrics, w_switch=0.5, w_continuity=0.5) -> float:
    """Collapse score_tracking dict to a single scalar for CMA-ES optimisation.

    Returns a value where higher = better (so CMA-ES minimises the negative).

    Args:
        metrics:      dict returned by score_tracking()
        w_switch:     weight for switch_rate minimisation (default 0.5)
        w_continuity: weight for continuity maximisation  (default 0.5)

    Returns:
        scalar score (higher is better)
    """
    switch_rate = metrics.get('color_switch_rate', 1.0)
    continuity  = metrics.get('continuity', {}).get('mean', 0.0)
    return -(w_switch * switch_rate + w_continuity * (1.0 - continuity))


def optimize_tracking_cma(
    instances_4d,
    volumes,
    ch_indices,
    pix_res,
    cell_flows=None,
    dense_flow_fields=None,
    x0=None,
    sigma0=0.15,
    max_evals=100,
    fixed_params=None,
    w_switch=0.5,
    w_continuity=0.5,
):
    """CMA-ES optimisation of track_sequence parameters.

    Uses score_tracking (switch_rate + continuity) as the closed-loop objective.

    Args:
        instances_4d:      [T, Z, H, W] segmentation output
        volumes:           [T, C, Z, H, W] raw multi-channel volumes for colour scoring
        ch_indices:        list of channel indices to use for confetti colour assignment
        pix_res:           {'z', 'y', 'x'} µm/pixel
        cell_flows:        {t: {cid: [u, v]}} from compute_cell_flows()
        dense_flow_fields: {t: [H, W, 2]} from compute_cell_flow_features()
        x0:                initial parameter vector (None → mid-range)
        sigma0:            initial CMA-ES step size (default 0.15)
        max_evals:         evaluation budget (default 100)
        fixed_params:      dict of additional track_sequence kwargs held fixed
                           (e.g. {'max_gap': 2, 'min_cell_size_px': 200})
        w_switch:          weight for switch_rate in scalar objective (default 0.5)
        w_continuity:      weight for (1 - continuity) in scalar objective (default 0.5)

    Returns:
        best_params:  dict of best parameters found (pass to track_sequence)
        history:      list of (params_dict, score) tuples, one per evaluation
    """
    from coastal.abm import track_sequence, score_tracking
    from coastal.track import extract_cell_colors, compute_3d_centroids

    fixed = fixed_params or {}

    # Build search space excluding fixed params — don't waste dimensions on locked values
    search_names = [n for n in TRACKING_PARAM_NAMES if n not in fixed]
    lo = np.array([TRACKING_PARAM_BOUNDS[n][0] for n in search_names])
    hi = np.array([TRACKING_PARAM_BOUNDS[n][1] for n in search_names])

    def _vec_to_search_params(x):
        return {name: float(np.clip(x[i], lo[i], hi[i]))
                for i, name in enumerate(search_names)}

    if x0 is None:
        x0 = list(0.5 * (lo + hi))

    history = []
    eval_count = [0]

    # Precompute once — these don't change between evaluations
    print("Precomputing cell colours and centroids...")
    _color_ids = extract_cell_colors(instances_4d, volumes, ch_indices)
    _centroids = compute_3d_centroids(instances_4d)
    print("Done.")

    def objective(x):
        params = _vec_to_search_params(x)
        params.update(fixed)
        tracks = track_sequence(
            instances_4d=instances_4d,
            pix_res=pix_res,
            cell_flows=cell_flows,
            dense_flow_fields=dense_flow_fields,
            **params,
        )
        metrics = score_tracking(
            tracks=tracks,
            instances_4d=instances_4d,
            volumes=volumes,
            ch_indices=ch_indices,
            pix_res=pix_res,
            verbose=False,
            _color_ids=_color_ids,
            _centroids=_centroids,
        )
        score = score_tracking_scalar(metrics, w_switch=w_switch, w_continuity=w_continuity)
        history.append(({**_vec_to_search_params(x), **fixed}, score))
        eval_count[0] += 1
        return -score  # CMA-ES minimises

    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            'maxfevals':       max_evals,
            'bounds':          [lo.tolist(), hi.tolist()],
            'BoundaryHandler': cma.BoundPenalty,
            'verbose':         -9,
        },
    )

    iteration = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)

        iteration += 1
        best_f = min(fitnesses)
        best_x = solutions[np.argmin(fitnesses)]
        best_p = {**_vec_to_search_params(best_x), **fixed}
        print(
            f"Iter {iteration:3d} ({eval_count[0]:4d} evals) | "
            f"score={-best_f:.4f} | "
            + " | ".join(f"{n}={best_p[n]:.3f}" for n in TRACKING_PARAM_NAMES if n in best_p)
        )

    best_entry = max(history, key=lambda h: h[1])
    best_params = best_entry[0]
    best_params.update(fixed)

    print(f"\nOptimization complete ({eval_count[0]} evaluations)")
    print(f"Best score: {best_entry[1]:.4f}")
    print("Best parameters:")
    for name in TRACKING_PARAM_NAMES:
        if name in best_params:
            print(f"  {name}: {best_params[name]:.4f}")

    return best_params, history
