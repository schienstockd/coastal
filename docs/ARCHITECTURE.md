# Architecture

> Update this file in the same change whenever you alter a pipeline stage, a module boundary, or
> an array-shape contract between stages.

Coastal is two subsystems sharing one data model: **segmentation** (flow → labels) and
**tracking** (labels → tracks). They are decoupled — tracking consumes the label volume that
segmentation produces and does not depend on the UNet.

## End-to-end pipeline

```
raw frames                                   [T, H, W]  or  [T, Z, H, W]
   │
   │  flow.py  — Farneback optical flow at scales [1,2,4,8] + derived metrics
   ▼
frame + ~14–16 flow metrics per pixel        [T, 1+M, H, W]
   │
   │  model.py + loss.py + train.py — UNetWithEmbeddings, 2-loss training (no labels)
   ▼
prob map + pixel embeddings                  [T, 1, H, W] + [T, D, H, W]  (D=64)
   │
   │  segment.py — two-pass region growing on embeddings + prob
   ▼
2D instances per frame                        [T, H, W]  (or per Z-slice)
   │
   │  utils.py — IOU label matching across Z (+ single-slice gap bridging)
   ▼
3D+T instance labels                          [T, Z, H, W]
   │
   │  track.py + abm.py — Kalman + LAP assignment, optional cost terms
   ▼
tracks                                        {track_id: {t: pos_um}}
```

## Module map

| Module | Role | Subsystem | Doc |
|---|---|---|---|
| `flow.py` | Farneback flow + multi-scale temporal/variance metrics; `prepare_data_for_unet` | segmentation | `SEGMENTATION.md` |
| `model.py` | `UNetWithEmbeddings` (prob head + embedding head) | segmentation | `SEGMENTATION.md` |
| `loss.py` | `IntensityLoss`, `TemporalMetricsLoss`, `VarianceMetricsLoss`, `WarpConsistencyLoss` | segmentation | `SEGMENTATION.md` |
| `train.py` | Dataset + augmentation, training loop (AMP, checkpointing), volume splits | segmentation | `SEGMENTATION.md` |
| `segment.py` | `LearnedAffinityInference`, `TwoPassSegmentationInference`, `Inference3D` | segmentation | `SEGMENTATION.md` |
| `utils.py` | `intersection_over_union`, `match_masks_3d`, `filter_small_cells` | segmentation | `SEGMENTATION.md` |
| `data.py` | `prepare_training_data`, `validate_training_data`; cecelia I/O contract | shared | `DATA.md` |
| `track.py` | `Track` dataclass, centroid/colour/intensity extraction, `ContextAssigner` | tracking | `TRACKING.md` |
| `abm.py` | `track_sequence` (Kalman+LAP), cost terms, `score_tracking`, breadcrumb ABM | tracking | `TRACKING.md` |
| `morphology.py` | polygon extraction + boundary HMM | tracking (failed) | `MORPHOLOGY.md` |
| `optimize.py` | CMA-ES tuning of segmentation + tracking params | shared | `OPTIMIZATION.md` |
| `viz.py` | overlays, RGB+segmentation plots, MP4 export | shared | — |

## Data contracts

- **Frames**: `[T, H, W]` (2D sequence) or `[T, Z, H, W]` (3D+T), float, normalised to [0,1].
- **Metrics tensor**: `[T, 1+M, H, W]` — channel 0 is the frame, the rest are flow metrics
  (M ≈ 14–16 depending on the metric set; see `docs/SEGMENTATION.md` and `QUICK_REFERENCE.txt`).
- **UNet output**: prob `[T, 1, H, W]` (sigmoid) + embeddings `[T, D, H, W]`, `D=64`.
- **Instances**: integer label array; 0 = background, contiguous positive labels per frame.
- **3D+T instances**: `[T, Z, H, W]`, labels consistent across Z within a timepoint after
  `match_masks_3d`.
- **Tracks**: `{track_id: {t: (z, y, x) in µm}}`. Physical units matter — Z=4.0 µm/px,
  XY≈0.48 µm/px; the anisotropy is load-bearing for tracking (see `docs/TRACKING.md`).

## The cecelia seam

Coastal is a standalone package; the "cecelia integration" is a data contract, not a code
dependency (no cecelia/zarr/tifffile import in the source). `data.py` operates on numpy arrays;
`train.py` accepts zarr/dask array-likes via duck typing (`.shape` + lazy slicing) without
importing them. cecelia currently **does not** depend on coastal (it "dropped coastal" — see
`docs/JULIA_PORT.md`). Details in `docs/DATA.md`.
