# Morphology

> Update this file when you change `morphology.py`.

Polygon extraction + boundary-state HMM (`morphology.py`, ~537 LOC). Built as a tracking signal;
**failed for that purpose** and is not in the live tracking path. Documented here so it is not
re-attempted blindly and so the reusable parts are known.

## What it does

- `labels_to_polygons` — Z-max-project labels, extract polygon outlines
  (`skimage.measure.find_contours` → `shapely` `Polygon`/`MultiPolygon`, `unary_union`).
- `extract_boundary_features` / `extract_shape_features` — per-cell boundary + 2D shape
  descriptors (`SHAPE_FEATURE_NAMES`).
- `fit_boundary_hmm` / `assign_boundary_states` / `assign_boundary_hmm_features` — fit a
  `hmmlearn.GaussianHMM` (diag covariance) to boundary states; smoothing helpers
  (`median_filter_states`, `enforce_min_run_length`).

Deps: `skimage.measure.find_contours`, `shapely`, `hmmlearn`, `sklearn.StandardScaler`,
`scipy.ndimage.gaussian_filter1d`, `multiprocessing.Pool`.

## Why it failed (for tracking)

T cells are featureless round blobs — every HMM boundary state comes out identical, so the model
carries no discriminative signal to distinguish cells. See `FAQ.md` and `docs/TRACKING.md`. This
is an instance of the general finding that **appearance is not discriminative** in this data.

## Reusable parts

The polygon/shape extraction (`labels_to_polygons`, `extract_shape_features`) is generic and
could feed a morphology-readout / QC use case unrelated to tracking. The HMM boundary-state
fitting is the part that specifically didn't pan out. If neither use materialises, this module is
a deprecation candidate — see `docs/FUTURE.md`.
