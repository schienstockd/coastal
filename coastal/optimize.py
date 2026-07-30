"""CMA-ES parameter optimization for segmentation using channel purity as objective."""

import numpy as np
import cma
from scipy import ndimage

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
    # Left at 0.6 deliberately. A tune with a working (non-flat) objective pins this at
    # whatever the upper bound is — 0.6, then 0.8 when widened — because the objective
    # rewards under-segmentation, not because higher is better. Widen it only after
    # score_segmentation stops being gameable; see docs/OPTIMIZATION.md.
    'affinity_threshold':                  (0.2, 0.6),
    'merge_affinity_threshold':            (0.2, 0.6),
    # Floored at 1.0: distance_transform_edt(~mask) is 0 on a fragment and >=1 off it, so any
    # value below 1 selects only the fragment's own pixels — which are then excluded — and
    # merging is silently disabled. The old (0.5, ...) bound let the search sit in that dead
    # zone, where the score is flat because the parameter has no effect. See
    # docs/OPTIMIZATION.md.
    'merge_max_distance':                  (1.0, 3.0),
    'merge_contact_brightness_threshold':  (0.2, 0.6),
    'prob_weight':                         (0.0, 0.6),
}

_LO = np.array([PARAM_BOUNDS[n][0] for n in PARAM_NAMES])
_HI = np.array([PARAM_BOUNDS[n][1] for n in PARAM_NAMES])


def _vec_to_params(x):
    return {name: float(np.clip(x[i], _LO[i], _HI[i]))
            for i, name in enumerate(PARAM_NAMES)}


def score_label_size_confetti(results, frames_multi, max_cell_size=300,
                              purity_threshold=0.8, background_percentile=25,
                              min_coloured_pixels=5, flows=None,
                              coherence_threshold=0.5, verbose=False):
    """
    "The largest reasonable label size while preserving confetti" (Dominik's formulation).

    Uses both signals, in their respective roles — **confetti is identity, flow is
    separation**:

      * confetti (identity)  — a label holding two colours holds two cells. Catches merges
        *across* colours, and nothing else: with ~270 cells per channel two merged cells
        share a colour about a third of the time, and then identity cannot see it.
      * flow (separation)    — a label spanning a motion discontinuity spans a boundary,
        whatever the colours are. This is what catches the same-colour merges identity is
        blind to, and it is the original premise of the package: cells are separated by
        their differential motion, not by their appearance.

    Pass `flows` to enable the separation constraint. A label then counts only if it is
    both colour-pure *and* motion-coherent.

    Confetti constrains segmentation from one side only: a label spanning two colours is a
    merge error, but a fragment of a single colour looks perfectly pure. So purity alone
    cannot see over-segmentation — which is ~86% of the errors. The fix is to make label
    *size* the thing being maximised and colour purity the *constraint*:

        score = sum over colour-pure labels of min(size, max_cell_size)**2
                ----------------------------------------------------------
                    max_cell_size * (coloured pixels in the IMAGE)

    Every failure mode moves it the right way, which is what `n_good`-counting could not do:

      * split a pure label in half   -> 2*(s/2)^2 < s^2                       score falls
      * merge two same-colour pieces
        of one cell                  -> (a+b)^2 > a^2 + b^2                   score rises
      * merge two different-colour
        cells                        -> label impure, contributes 0           score falls hard
      * drop a label                 -> numerator loses its term              score falls
      * grow one label over a whole
        region instead of tiling it  -> numerator capped at one cap^2         score falls

    The denominator counts colour-carrying pixels of the **image**, not of the labelling.
    That is deliberate and was a bug on the first attempt: normalising by labelled area
    makes the score a quality ratio, so segmenting a single perfect cell and ignoring the
    rest of the frame scores 1.0 — the same "ratio over a subset" trap that made the old
    `n_good / n_large` gameable. Against a fixed, image-derived denominator, missing a cell
    always costs.

    The cap is what makes it "reasonable". Without it the maximum is one enormous label, and
    with only 3 confetti channels shared by ~270 cells each (see README) a huge label can be
    colour-pure by accident, so an uncapped size reward would be gamed.

    It is normalised to roughly [0, 1]: 1.0 means every labelled pixel belongs to a
    colour-pure label of exactly max_cell_size.

    Args:
        results:              list of result dicts from predict_sequence / predict_frame
        frames_multi:         [T, C, H, W] raw multi-channel frames
        max_cell_size:        px area of a plausible cell — the "reasonable" cap. At
                              0.497 um/px a ~8 um T cell is ~200 px in 2D projection, so
                              300 is a generous single cell. Calibrate on your own data.
        purity_threshold:     dominant-colour fraction required to count as one cell
                              (default 0.8)
        background_percentile: per-channel background subtracted before assigning colours
                              (default 25; purity is meaningless without it — see
                              score_segmentation)
        min_coloured_pixels:  labels with fewer bright/coloured pixels than this cannot be
                              judged, and are treated as impure (they still count in the
                              denominator, so they are not free)
        flows:                optional list of T [H, W, 2] dense (u, v) fields — e.g.
                              `flow.extract_dense_flow_pairs(multi_scale_flows, scale=1)` or
                              the `dense_flows` from `abm.compute_cell_flow_features`. When
                              given, a label must also be motion-coherent to count.
        coherence_threshold:  minimum vector coherence |Σv| / Σ|v| over a label's pixels
                              (default 0.5). 1.0 = every pixel moves the same way; near 0 =
                              the label straddles opposing motions, i.e. a boundary runs
                              through it. Pixels that barely move are ignored, since their
                              direction is noise.
        verbose:              print per-frame diagnostics

    Returns:
        scalar score in ~[0, 1], higher is better
    """
    frames_arr = np.asarray(frames_multi, dtype=np.float32)
    if frames_arr.ndim == 3:
        raise ValueError("frames_multi must be [T, C, H, W], not [T, H, W]")
    T, C = frames_arr.shape[0], frames_arr.shape[1]

    frame_scores = []

    for t, result in enumerate(results):
        if t >= T:
            continue
        instances = result['instances']
        frame = frames_arr[t]

        # Per-pixel confetti colour: brightest channel after background removal. Dim
        # pixels get no colour so background cannot make a label look pure.
        bg = np.percentile(frame.reshape(C, -1), background_percentile, axis=1)
        sub = np.clip(frame - bg[:, None, None], 0, None)
        bright = sub.max(axis=0)
        colour = sub.argmax(axis=0)
        colour[bright <= np.percentile(bright, 60)] = -1

        # Fixed, image-derived denominator: how much colour-carrying tissue there is to
        # account for. Independent of the labelling, so missing a cell cannot be free.
        total_coloured = int((colour >= 0).sum())

        numerator = 0.0
        n_pure = n_impure = n_incoherent = 0
        pure_sizes = []

        for label, sl in enumerate(ndimage.find_objects(instances), start=1):
            if sl is None:
                continue
            mask = instances[sl] == label
            size = int(mask.sum())

            cols = colour[sl][mask]
            cols = cols[cols >= 0]
            if len(cols) < min_coloured_pixels:
                n_impure += 1
                continue
            counts = np.bincount(cols, minlength=C)
            if counts.max() / counts.sum() < purity_threshold:
                n_impure += 1
                continue

            # Separation: does a motion boundary run through this label? Identity cannot
            # answer that when both cells happen to share a colour.
            if flows is not None and t < len(flows) and flows[t] is not None:
                uv = np.asarray(flows[t], dtype=np.float32)[sl][mask]      # [N, 2]
                mag = np.linalg.norm(uv, axis=1)
                moving = mag > max(1e-3, 0.25 * np.median(mag) if len(mag) else 0.0)
                if moving.sum() >= min_coloured_pixels:
                    resultant = np.linalg.norm(uv[moving].sum(axis=0))
                    coherence = resultant / (mag[moving].sum() + 1e-6)
                    if coherence < coherence_threshold:
                        n_incoherent += 1
                        continue

            n_pure += 1
            pure_sizes.append(size)
            numerator += min(size, max_cell_size) ** 2

        if total_coloured == 0:
            continue
        score = numerator / (max_cell_size * total_coloured)
        frame_scores.append(score)

        if verbose:
            med = float(np.median(pure_sizes)) if pure_sizes else 0.0
            print(f"  Frame {t}: {n_pure} counted / "
                  f"{n_pure + n_impure + n_incoherent} labels "
                  f"({n_impure} mixed-colour, {n_incoherent} mixed-motion) | "
                  f"median size {med:.0f}px (cap {max_cell_size}) | "
                  f"coloured area {total_coloured}px | score={score:.4f}")

    return float(np.mean(frame_scores)) if frame_scores else 0.0


def score_segmentation(results, frames_multi, min_cell_size=100,
                       purity_threshold=0.7, background_percentile=25,
                       junk_weight=0.05, count_penalty_weight=0.0,
                       verbose=False):
    """
    Score segmentation quality: count the plausible cells, penalise the junk.

    NOTE: prefer `score_label_size_confetti`. This one thresholds size into a binary
    good/fragment decision, so it is only weakly sensitive to over-segmentation; the other
    maximises label size subject to colour purity, which is what confetti can actually
    constrain. Kept because the tuning results recorded in docs/OPTIMIZATION.md were
    measured with it.

    Labels are classified per frame as:
      - good:       large (>= min_cell_size) AND dominant channel >= purity_threshold
      - merged:     large but impure — multiple channel types blended
      - fragmented: too small (< min_cell_size)

    Frame score = n_good - junk_weight * (n_merged + n_fragmented)
                  - count_penalty_weight * n_total

    averaged over frames. Higher is better; it can go negative. No ground-truth masks
    are involved — "good" is a confetti-purity proxy, not a verified cell.

    **This is a reward, not a ratio, on purpose.** It previously returned
    `n_good / n_large`, which a tuner maximises by *discarding* cells: tighten
    `affinity_threshold`, large cells drop into the (nearly free) fragment bin, and the
    surviving fraction looks purer. Measured on a real movie, that let the score climb
    0.429 -> 0.523 while the absolute number of good cells fell 140 -> 116, and
    `affinity_threshold` pinned to every upper bound it was given. In the reward form
    every way of losing a good cell costs at least 1, so there is no such gradient —
    `tests/test_optimize_objective.py` pins that. See docs/OPTIMIZATION.md.

    Args:
        results:              list of result dicts from predict_sequence / predict_frame
        frames_multi:         [T, C, H, W] raw multi-channel frames (uint8 or float)
        min_cell_size:        pixel threshold below which a label is "fragmented" (default 100)
        purity_threshold:     dominant-channel fraction above which a cell is "good"
                              (default 0.7 — meaningful only with background_percentile set,
                              see below)
        background_percentile: per-channel percentile subtracted before computing purity
                              (default 25; None = don't subtract, the legacy behaviour).
                              Purity is `max(mean_ch / mean_ch.sum())`, so it is floored at
                              1/n_channels; on background-inclusive intensities it sits just
                              above that floor and cannot discriminate. Measured over 326
                              large cells in a real movie: median 0.385 spanning 34% of the
                              usable range without subtraction, vs median 0.709 spanning 90%
                              with it. Without subtraction any purity_threshold >= 0.6 makes
                              the score identically zero.
        junk_weight:          cost per merged or fragmented label (default 0.05). This is the
                              recall/precision dial: 0 optimises raw cell count, large values
                              optimise cleanliness. Fragments dominate it in practice because
                              they dominate the counts.
        count_penalty_weight: extra penalty per label found, of any class (default 0.0 = off)
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

        if background_percentile is not None:
            # Remove each channel's background floor, otherwise every cell reads as an
            # even mix of all channels and purity collapses toward 1/C.
            bg = np.percentile(frame.reshape(C, -1), background_percentile, axis=1)
            frame = np.clip(frame - bg[:, None, None], 0, None)

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

        n_total = n_good + n_merged + n_fragmented
        junk_penalty = junk_weight * (n_merged + n_fragmented)
        count_penalty = count_penalty_weight * n_total

        score = n_good - junk_penalty - count_penalty
        frame_scores.append(score)

        if verbose:
            print(f"  Frame {t}: {n_total} total | "
                  f"{n_good} good | {n_merged} merged | {n_fragmented} fragmented | "
                  f"junk_penalty={junk_penalty:.2f} | count_penalty={count_penalty:.2f} | "
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
    background_percentile=25,
    junk_weight=0.05,
    count_penalty_weight=0.0,
    fixed_params=None,
    device=None,
):
    """
    CMA-ES optimization of LearnedAffinityInference parameters.

    Optimizes the 5 parameters in PARAM_NAMES to maximise the number of plausible cells
    found, net of a penalty on junk. Each label is classified as:
      - good:       >= min_cell_size px AND dominant channel >= purity_threshold
      - merged:     large but impure (multiple channel types blended)
      - fragmented: too small to be a real cell

    Score = n_good - junk_weight * (n_merged + n_fragmented) - count_penalty_weight *
    n_total, averaged over n_frames (see score_segmentation, the actual implementation,
    for why this is a reward rather than a ratio — a ratio is maximised by discarding
    cells).

    `junk_weight` is the dial that matters: 0 optimises raw cell count, large values
    optimise cleanliness. Fragments dominate it in practice — on a real movie 1951 of
    2277 labels were below min_cell_size — so it is effectively a fragmentation penalty.

    A flat objective still means the result is meaningless; this function warns when
    every evaluation scored the same. The usual cause used to be purity_threshold above
    the achievable range, which `background_percentile` fixes (see score_segmentation).

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
            background_percentile=background_percentile,
            junk_weight=junk_weight,
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

    # A flat objective returns whichever sample came first, which looks exactly like a
    # successful tune. Say so loudly instead: this is how a parameter set with merging
    # silently disabled came to be treated as "the tuned best".
    scores = [s for _, s in history]
    if scores and max(scores) == min(scores):
        flat_at = scores[0]
        print(
            f"\n*** WARNING: the objective never varied (every one of {len(scores)} "
            f"evaluations scored {flat_at:.4f}). ***\n"
            f"    These parameters are just the first sample evaluated, NOT an optimum.\n"
            + (f"    n_good was 0 everywhere: purity_threshold={purity_threshold} is above the\n"
               f"    achievable range for this data. Inspect the purity distribution and lower it\n"
               f"    (~0.4-0.5 on the confetti movies), then re-run.\n"
               if flat_at == 0.0 else
               "    Widen the bounds or change the objective; the search had nothing to climb.\n")
            + "    See docs/OPTIMIZATION.md -> Degenerate objectives."
        )

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
