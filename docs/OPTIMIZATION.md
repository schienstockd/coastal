# Optimization

> Update this file when you change CMA-ES params, bounds, or objective functions in
> `optimize.py`.

CMA-ES hyperparameter tuning (`optimize.py`, ~424 LOC) for both subsystems, via
`cma.CMAEvolutionStrategy` with a `BoundPenalty` boundary handler and an ask/tell loop.

## Segmentation tuning

- `optimize_segmentation_cma(...)` — searches the `TwoPassSegmentationInference` parameters
  (seed sizes, affinity thresholds, blur sigmas, merge thresholds, prob threshold, …).
- Objective: `score_segmentation(...)` — channel-purity / tracking-derived score (no GT masks).

## Tracking tuning

- `optimize_tracking_cma(...)` — searches `track_sequence` cost weights (`w_flow`, `w_color`,
  `w_ctx`, gate, …) within `TRACKING_PARAM_BOUNDS`.
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
