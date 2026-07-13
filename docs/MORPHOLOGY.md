# Morphology

> Update this file when you change `morphology.py`.

`morphology.py` now holds **only** the polygon + 2D shape-feature readout. The boundary-state HMM
that once lived here was built as a tracking/identity signal, **failed** for that purpose, and has
been **removed** (recorded in [`DEAD_ENDS.md`](DEAD_ENDS.md) with a git ref to revive it).

## What it does (retained)

- `labels_to_polygons` — Z-max-project labels, extract polygon outlines
  (`skimage.measure.find_contours` → `shapely` `Polygon`/`MultiPolygon`, `unary_union`).
- `extract_shape_features` / `extract_cell_morphology` — per-cell 2D shape descriptors
  (`SHAPE_FEATURE_NAMES`; regionprops + derived ratios).

These are a **standalone morphology/QC readout — not currently wired into the segmentation or
tracking pipelines**, kept for potential future use.

Deps: `skimage.measure` (`find_contours`, `regionprops`), `shapely`, `multiprocessing.Pool`.

## Removed: boundary-state HMM (why it failed)

The HMM path (`fit_boundary_hmm`, `assign_boundary_states`, `assign_boundary_hmm_features`,
`extract_boundary_features`, curvature/`fold_score` features, run-length smoothing) fit a
`hmmlearn.GaussianHMM` to per-cell boundary states. T cells are featureless round blobs, so every
boundary state came out identical — no discriminative signal to distinguish cells. This is an
instance of the general finding that **appearance is not discriminative** in this data (see
`FAQ.md`, `docs/TRACKING.md`). Full rationale + revival instructions: [`DEAD_ENDS.md`](DEAD_ENDS.md).
