# Optimization

> Update this file when you change CMA-ES params, bounds, or objective functions in
> `optimize.py`.

CMA-ES hyperparameter tuning (`optimize.py`, ~424 LOC) for both subsystems, via
`cma.CMAEvolutionStrategy` with a `BoundPenalty` boundary handler and an ask/tell loop.

## Segmentation tuning

- `optimize_segmentation_cma(...)` — searches the `LearnedAffinityInference` parameters listed in
  `PARAM_NAMES` (affinity thresholds, merge thresholds/distance, `prob_weight`) within
  `PARAM_BOUNDS`; anything else is passed through `fixed_params`.
- Objective: `score_segmentation(...)` — a confetti-purity proxy, no GT masks:

  ```
  frame_score = n_good - junk_weight * (n_merged + n_fragmented)
  ```

  averaged over `n_frames`, where `good` = at least `min_cell_size` px **and**
  dominant-channel fraction ≥ `purity_threshold`, `merged` = large but impure, `fragmented` = too
  small. Higher is better and it can go negative.

  Two design points, both the result of the failures recorded below:

  - **It is a reward, not a ratio.** Any ratio over a subset (the old `n_good / n_large`) is
    maximised by *discarding* cells. In the reward form every way of losing a good cell costs at
    least 1, so there is no "find fewer cells" gradient. `tests/test_optimize_objective.py` pins
    this behaviourally: dropping a cell, shrinking cells below `min_cell_size`, splitting one into
    fragments, and over-merging two must each lower the score.
  - **Purity is computed after per-channel background subtraction** (`background_percentile=25`;
    `None` restores the legacy behaviour). Purity is `max(mean_ch / mean_ch.sum())`, so it is
    floored at `1/n_channels`; on background-inclusive intensities it never leaves that floor.

  `junk_weight` is the recall/cleanliness dial — 0 optimises raw cell count, large values optimise
  cleanliness. Fragments dominate it in practice (1951 of 2277 labels on a real movie were below
  `min_cell_size`), so it behaves mostly as a fragmentation penalty. `count_penalty_weight` is still
  there but penalises good cells too; prefer `junk_weight`.

### How this objective failed before — three distinct ways (2026-07-30)

Kept because each is easy to reintroduce, and none of them looks like a failure from the outside: a
broken objective still returns confident-looking "best parameters".

1. **Flat because `purity_threshold` was unreachable.** Purity is floored at `1/n_channels`, and
   without background subtraction it stays there. Measured over 326–344 large cells in a real movie:
   median **0.385**, max **0.558**, spanning only **34%** of the usable `[1/3, 1]` range. Any
   `purity_threshold ≥ 0.6` therefore gave `n_good = 0` for *every* candidate, so the score was
   constant 0.0. And `best_params` is `max(history, ...)`, which on ties returns the **first** entry —
   so the run reported its first random sample as the optimum. That is where the notebook's
   `BEST_PARAMS` came from, including a `merge_affinity_threshold` (0.2261) below
   `affinity_threshold` (0.5534). Fixed by subtracting background (median → **0.709**, **90%** of the
   range), which also makes the documented `purity_threshold=0.7` default usable. A synthetic case in
   `tests/test_optimize_objective.py` shows four perfectly single-channel cells scoring 0 at every
   threshold with background left in, and 4/4 with it removed.
2. **Flat because a parameter had no effect inside its bounds.** `merge_max_distance < 1.0` disables
   merging entirely (`distance_transform_edt(~mask)` is 0 on the fragment and ≥ 1 off it, so a sub-1
   threshold selects only the fragment's own pixels, which are then excluded). The bound was
   `(0.5, 3.0)` — a third of it a dead zone. Now floored at **1.0**, pinned by a test.
3. **Gameable because the score was a ratio over a subset.** With the threshold fixed, the objective
   was no longer flat but `affinity_threshold` pinned to whatever the upper bound was (0.6, then 0.8
   when widened). Measured on a held-out TEST movie:

   | params | total | large | **n_good** | frag | `n_good/n_large` |
   |---|---|---|---|---|---|
   | shipped, aff 0.553 | 2277 | 326 | **140** | 1951 | 0.429 |
   | tuned, aff 0.600 | 1891 | 251 | **122** | 1640 | 0.486 |
   | tuned, aff 0.800 | 1714 | 222 | **116** | 1492 | 0.523 |

   The score climbed while the absolute number of good cells **fell**: stricter growing pushes large
   cells into the near-free fragment bin and the survivors look purer. Fixed by switching to the
   reward form above. `affinity_threshold`'s bound was left at 0.6 rather than widened to chase the
   wall.

**Checklist before trusting a tuning run:** did the score vary (the function warns if not)? Did any
parameter land exactly on a bound? Did absolute `n_good` improve, or only a ratio? Is
`purity_threshold` inside the measured purity distribution for *your* data?

Fragmentation, not merging, is where the quality problem lives: **1951 of 2277** detections on that
movie are below `min_cell_size=100`. `junk_weight` is the knob that expresses how much that matters.

### What the fixed objective revealed: these 5 parameters are not the lever (2026-07-30)

With the objective working, a tune was run per `junk_weight` (200-eval budget each, 5 TRAIN eval
frames, `purity_threshold=0.7`, `background_percentile=25`). Good cells found on a **held-out TEST**
movie, 5 frames:

| `junk_weight` | good | merged | fragmented | total | params pinned at a bound |
|---|---|---|---|---|---|
| *shipped params* | **171** | 155 | 1951 | 2277 | — |
| 0.0 | 167 | 133 | 2280 | 2580 | `merge_affinity_threshold`, `prob_weight` |
| 0.02 | **175** | 163 | 2142 | 2480 | `prob_weight` |
| 0.05 | 170 | 134 | 1850 | 2154 | `affinity_threshold`, `merge_affinity_threshold` |
| 0.1 | 168 | 134 | 1838 | 2140 | `affinity_threshold` |
| 0.3 | 144 | 108 | 1640 | 1892 | `affinity_threshold`, `merge_contact_*`, `prob_weight` |

**`n_good` is essentially invariant** — 167–175 across every setting except the extreme 0.3. What the
parameters actually trade is the *fragment* population (1640–2280). They do not change how many real
cells are findable.

The headroom can be read off directly: at `junk_weight=0` the score **is** `n_good` per frame, and
across that search it spanned **16.6–22.2 good cells/frame** with the shipped parameters already at
**20.8**. So tuning these five parameters is worth at most ~7% on this metric, and the shipped values
are within 7% of the best found. That is why no new `BEST_PARAMS` were adopted from this sweep.

What that points at instead: the prob map and embeddings decide which cells are findable, so the
lever is **training** (and the oversegmentation that yields ~1900 fragments per 5 frames), not the
inference thresholds. `junk_weight` also flips which direction `affinity_threshold` is pushed —
down at 0.0–0.02, up at ≥0.05 — which is a genuine trade-off rather than the one-way gaming the old
ratio produced.

## Tracking tuning

- `optimize_tracking_cma(...)` — searches `track_sequence` cost weights (`w_flow`, `w_color`,
  `chi2_gate`, `max_cost`, …) within `TRACKING_PARAM_BOUNDS`. (The removed cost terms — `w_ctx`
  and friends — are gone; see `DEAD_ENDS.md`.)
- Objective: `score_tracking_scalar(...)` — scalarised combination of `continuity` and
  `switch_rate` (see `docs/TRACKING.md` for the two metrics; the scalarisation is where the
  continuity/switch_rate trade-off is encoded — document any change to the weighting here).
- The three failure modes above are not segmentation-specific. `score_tracking_scalar` is also a
  scalarised combination, so it is worth checking whether it too can be improved by producing
  *fewer* tracks — the same "ratio over a subset" trap.

## Conventions

- Param names + bounds are the single source of truth for what is tunable; keep
  `TRACKING_PARAM_BOUNDS` and the segmentation param list in sync with the actual
  `track_sequence` / `TwoPassSegmentationInference` signatures.
- `cma` is a real runtime dependency (declared in `pyproject.toml`).
- Scoring pulls in `coastal.segment`, `coastal.abm`, `coastal.track` — keep those import edges
  in mind when refactoring (optimize sits downstream of both subsystems).

