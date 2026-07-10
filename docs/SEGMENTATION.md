# Segmentation

> Update this file in the same change whenever you alter flow metrics, the UNet, a loss,
> training, inference, or 3D stitching.

Instance segmentation of cells with **no ground-truth labels** — training is guided entirely by
optical-flow structure. Pipeline: `flow.py` → `model.py` + `loss.py` + `train.py` →
`segment.py` → `utils.py`.

## 1. Flow metrics (`flow.py`)

Multi-scale Farneback optical flow (`cv2.calcOpticalFlowFarneback`, Gaussian variant) computed
at temporal scales `[1, 2, 4, 8]`, plus derived metrics. `prepare_data_for_unet(frames,
temporal_scales=[1,2,4,8])` returns the prepped frames + metrics. ~14–16 metrics per pixel:

- **4 multi-scale magnitudes** (`mag_1..mag_8`) — motion at increasing time gaps (noise ↓).
- **temporal consistency** — motion consistency across scales, acceleration, direction stability.
- **cumulative displacement** (`cumulative_mag`) — total movement over the window; alone
  separates fast/slow/stationary cells.
- **deformation** — divergence, vorticity, strain.
- **structural** — edge strength, motion-at-edges.

`VarianceMetricsConfig` + `compute_variance_metrics` provide an alternative variance-based metric
set. Farneback runs **CPU-only** (no CUDA OpenCV dep). Parallelised per-frame via joblib. Full
parameter/selection guide: `QUICK_REFERENCE.txt`.

## 2. Model (`model.py`)

`UNetWithEmbeddings` — a UNet with two heads:

```
Input:  [B, 1 + M, H, W]     frame + flow metrics
Output: [B, 1, H, W]         cell probability map (sigmoid)
        [B, D, H, W]         learned pixel embeddings, D=64
```

Pure `torch.nn` (ConvBlock, MaxPool, Upsample, ModuleList). ~107 LOC.

## 3. Losses (`loss.py`)

```
total_loss = intensity_weight * L_intensity + temporal_weight * L_temporal
```

- **`IntensityLoss`** — rewards bright pixels, local contrast, edge strength. No labels.
- **`TemporalMetricsLoss`** — hard contrastive: pixels with similar optical-flow metrics should
  be close in embedding space.
- **`VarianceMetricsLoss`**, **`WarpConsistencyLoss`** — alternative/auxiliary objectives.

Default weights: `intensity_weight=1.0, temporal_weight=2.0`. Raise `temporal_weight` (3.0–4.0)
to reduce oversegmentation.

## 4. Training (`train.py`)

`train_with_metrics(train_frames, train_metrics, num_epochs=50)` — Adam, mixed precision
(`amp.GradScaler` + `autocast`), gradient clipping, checkpointing. `TemporalDatasetWithAugmentation`
handles augmentation. Multi-movie pipeline:

```python
all_frames, all_metrics = prepare_data_for_unet_batch([movie1, movie2, ...])
train_frames, test_frames, train_metrics, test_metrics = train_test_split_per_movie(
    all_frames, all_metrics)
model, history = train_with_metrics(train_frames, train_metrics, num_epochs=50)
```

`extract_sequences_from_volume` / `prepare_data_for_unet_batch_4d` handle 3D+T volumes.

## 5. Inference (`segment.py`)

`LearnedAffinityInference` grows regions from seeds using embedding affinity (cosine/`cdist`) +
the prob map, then merges fragments. `TwoPassSegmentationInference` runs it twice:

- **Pass 1**: large seeds (32 px), low affinity threshold (0.2) → large cells.
- **Pass 2**: small seeds (8 px), high threshold (0.8) on the remaining space → fragments.

`Inference3D` applies this per Z-slice and stitches via `utils.match_masks_3d`.

### Default best parameters

```python
TwoPassSegmentationInference(
    model=model,
    seed_size_large=32, affinity_threshold_large=0.2, embedding_blur_sigma_large=1.5,
    merge_affinity_threshold_large=0.90,
    seed_size_small=8,  affinity_threshold_small=0.8, embedding_blur_sigma_small=1.5,
    merge_affinity_threshold_small=0.90,
    prob_threshold=0.3, max_iter=200, min_component_size=10, device='cuda')
```

## 6. 3D label stitching (`utils.py`)

`match_masks_3d(masks_3d, stitch_threshold, gap_tolerance=1, gap_iou_threshold=0.3)` — matches
labels across Z by sparse IOU overlap, then bridges chains broken by ≤ `gap_tolerance` bad slices
(`_bridge_label_gaps`). `intersection_over_union` builds the sparse overlap matrix;
`filter_small_cells(instances_4d, min_voxels)` drops sub-threshold labels per timepoint.

## Known issues

- **Y-cell splitting** — cells with a body + probing leading edge segment as two instances.
  Mitigate with `merge_affinity_threshold > 0.90`.
- **Oversegmentation** — raise `embedding_blur_sigma` (2.0–3.0) or `temporal_weight` in training
  (3.0–4.0).

---

## Appendix: flow-metric reference

Folded in from the former root-level `QUICK_REFERENCE.txt`. The full 16-metric set produced by
`prepare_data_for_unet(frames, temporal_scales=[1,2,4,8])`, and how to select a subset via
`metrics_to_tensor(temporal_metrics, selected_keys=...)`.

### The 16 metrics
- **4 multi-scale magnitudes** — `mag_1` (frame-to-frame, fine but noisy), `mag_2`, `mag_4`,
  `mag_8` (longer gaps, less noise, more persistent patterns).
- **3 temporal consistency** — `motion_consistency` (similarity across scales; 0 = noise,
  1 = coherent), `acceleration`, `direction_stability`.
- **1 cumulative displacement** — `cumulative_mag` (total movement over the window; on its own
  separates fast / slow / stationary cells).
- **3 deformation** — `divergence` (local expansion/compression), `vorticity` (rotation),
  `strain` (total deformation).
- **2 structural** — `edge_strength` (image gradient), `motion_at_edges`.
- **1 target** — `cell_boundary_likelihood`.

Why multi-scale: it turns "is this pixel moving *now*?" (noisy, ambiguous) into "what is this
pixel's motion *signature* over time?" (robust) — which is what makes cells separable without
bright reporters. See `FAQ.md`.

### Parameters
- `temporal_scales`: `[1,2,4]` faster (testing) · **`[1,2,4,8]` recommended** · `[1,2,3,4,5]`
  more granular / slower.
- `cumulative_window`: `3` fast · **`5` recommended** · `7` more temporal context.

### Metric-selection presets
- **Minimum (5)**: `mag_1`, `cumulative_mag`, `edge_strength`, `motion_at_edges`,
  `cell_boundary_likelihood`.
- **Recommended (12)**: the 4 `mag_*`, `motion_consistency`, `cumulative_mag`, `divergence`,
  `vorticity`, `strain`, `edge_strength`, `motion_at_edges`, `cell_boundary_likelihood`.
- **All (16)**: default (no `selected_keys`).

The UNet's `in_channels` must match the metric count chosen (plus the frame channel — see the
data contract in `docs/ARCHITECTURE.md`).

### Cost & troubleshooting
- Metrics are computed **once** and reused across all training epochs. Multi-scale is markedly
  slower than frame-to-frame; drop scale 8 (`[1,2,4]`) if too slow.
- All-zero metrics → check frame normalisation to [0,1]
  (`(f - f.min()) / (f.max() - f.min() + 1e-5)`).
- Out of memory → fewer temporal scales.
- No improvement → verify inputs; sweep `cumulative_window` (3/5/7) and `temporal_scales`.
