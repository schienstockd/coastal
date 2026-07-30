# Optimization

> Update this file when you change CMA-ES params, bounds, or objective functions in
> `optimize.py`.

CMA-ES hyperparameter tuning (`optimize.py`, ~424 LOC) for both subsystems, via
`cma.CMAEvolutionStrategy` with a `BoundPenalty` boundary handler and an ask/tell loop.

## Segmentation tuning

- `optimize_segmentation_cma(...)` — searches the `LearnedAffinityInference` parameters listed in
  `PARAM_NAMES` (affinity thresholds, merge thresholds/distance, `prob_weight`) within
  `PARAM_BOUNDS`; anything else is passed through `fixed_params`.
- Objective: `score_segmentation(...)` — channel purity, no GT masks. The actual formula is
  `n_good / (n_good + n_merged) - count_penalty_weight * n_total`, averaged over `n_frames`.
  **Fragments are not in the denominator**, so with the default `count_penalty_weight=0` they cost
  nothing — the search has no reason to merge fragments, and merging can only lower purity. Set
  `count_penalty_weight > 0` to penalise fragmentation.

### Degenerate objectives

A flat objective is not a tuning failure you can see: `best_params` is `max(history, ...)`, and on
ties Python returns the **first** entry, so a completely flat run reports the first sample evaluated
as if it were an optimum. `optimize_segmentation_cma` now prints a loud warning when every
evaluation scored the same. Two ways it has actually happened here:

1. **`purity_threshold` above the achievable range.** The statistic is
   `max(mean_ch / mean_ch.sum())` over background-inclusive intensities, so it is floored near
   `1/n_channels` and compressed upward. Measured on a real confetti movie (344 large cells): median
   **0.395**, max **0.558**. So `purity_threshold` ≥ 0.6 gives `n_good = 0` for every candidate and
   the score is constant 0.0; 0.4 gives 160/344 and a usable gradient. The default (0.7) and the
   notebook's commented-out call (0.8) are both in the dead range — check the purity distribution on
   your data before trusting a run.
2. **A parameter with no effect inside its bounds.** `merge_max_distance < 1.0` disables merging
   entirely (`distance_transform_edt(~mask)` is 0 on the fragment and ≥ 1 off it, so a sub-1
   threshold selects only the fragment's own pixels, which are then excluded). The bound was
   `(0.5, 3.0)`, so a third of the range was a flat dead zone; it is now floored at **1.0**.

The `BEST_PARAMS` in `notebooks/pipeline_consensus.ipynb` came from a run hitting both, which is why
they had merging switched off and a `merge_affinity_threshold` (0.2261) below `affinity_threshold`
(0.5534) — a combination the `LearnedAffinityInference` docstring calls unusual. They are a starting
point, not a result.

## Tracking tuning

- `optimize_tracking_cma(...)` — searches `track_sequence` cost weights (`w_flow`, `w_color`,
  `chi2_gate`, `max_cost`, …) within `TRACKING_PARAM_BOUNDS`. (The removed cost terms — `w_ctx`
  and friends — are gone; see `DEAD_ENDS.md`.)
- Objective: `score_tracking_scalar(...)` — scalarised combination of `continuity` and
  `switch_rate` (see `docs/TRACKING.md` for the two metrics; the scalarisation is where the
  continuity/switch_rate trade-off is encoded — document any change to the weighting here).

## Conventions

- Param names + bounds are the single source of truth for what is tunable; keep
  `TRACKING_PARAM_BOUNDS` and the segmentation param list in sync with the actual
  `track_sequence` / `TwoPassSegmentationInference` signatures.
- `cma` is a real runtime dependency (declared in `pyproject.toml`).
- Scoring pulls in `coastal.segment`, `coastal.abm`, `coastal.track` — keep those import edges
  in mind when refactoring (optimize sits downstream of both subsystems).
