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

### The objective is gameable — fix it before re-tuning (2026-07-30)

Making the objective non-flat is necessary but **not sufficient**. Re-tuned with a working threshold
(`purity_threshold=0.4`, `count_penalty_weight=0.0012`, 5 TRAIN eval frames), CMA-ES converges but
`affinity_threshold` pins to whatever the upper bound is — 0.6, then 0.8 when the bound was widened.
Measured on a held-out TEST movie:

| params | total | large | **n_good** | frag | `n_good/n_large` (the score) |
|---|---|---|---|---|---|
| shipped, aff 0.553 | 2277 | 326 | **140** | 1951 | 0.429 |
| tuned, aff 0.600 | 1891 | 251 | **122** | 1640 | 0.486 |
| tuned, aff 0.800 | 1714 | 222 | **116** | 1492 | 0.523 |

The score climbs while the absolute number of good cells **falls**. Because the denominator is
`n_large` only, stricter growing moves large cells into the fragment bin — which costs almost nothing
at a sane `count_penalty_weight` — and the surviving fraction looks purer. The search is rewarded for
finding *fewer* cells. So do not adopt tuned parameters from this objective, and do not widen
`affinity_threshold`'s bound to chase the wall.

Two fixes to make first, both measured:

1. **Subtract background before computing purity.** The statistic is
   `max(mean_ch / mean_ch.sum())` on raw intensities, so it is floored at `1/n_channels` and sits just
   above it. On a real movie (326 large cells): median **0.385**, spanning **34%** of the usable
   `[1/3, 1]` range. Subtracting each channel's 25th percentile first: median **0.709**, spanning
   **90%**. Most large cells *are* strongly single-channel-dominant — the current metric hides it
   behind background, which is also why any threshold ≥ 0.6 zeroes the objective out. Fix this and the
   function's own default (0.7) becomes meaningful again, rather than needing to be lowered to 0.4.
2. **Stop maximising a ratio over a subset.** Put fragments in the denominator (which is what
   `optimize_segmentation_cma`'s docstring claimed the score was, before it was corrected to match the
   code), or optimise `n_good` against a penalty on junk, so discarding a good cell always costs.
   Under `n_good / n_total` the same three runs score 0.0615 / 0.0645 / 0.0677 — far less pathological,
   though still not rewarding absolute recall.

Note that fragmentation, not merging, is where the quality problem lives: **1951 of 2277** detections
on that movie are below `min_cell_size=100`.

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
