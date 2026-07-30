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

### Memory: the 4D path (`predict_temporal_volume`)

`predict_temporal_volume(volume, ...)` is the entry point for a whole `[T,C,Z,Y,X]` movie, and on
real confetti movies (`[180, 4, 15, 531, 586]` uint16 ≈ 6.7 GB) it is the one place in coastal where
RAM is the binding constraint. Three rules:

1. **Pass the volume lazily.** It reads `volume.shape` and then slices `volume[:, :, z, :, :]`, so a
   dask array straight from cecelia streams one z-slice at a time. `np.asarray(volume)` at the call
   site materialises the whole movie first and defeats this — the notebook regression that OOM-killed
   the kernel. `tests/test_segment.py` pins the contract with a stand-in whose `__array__` raises.
2. **`n_workers` is a memory multiplier, not just a speed knob.** One in-flight z-slice holds its
   flow fields — multi-scale + cumulative, ~1.8 GB at T=180, 531×586 — for the slice's lifetime.
   The 14 float32 metric planes per frame are built on demand (`flow.TemporalMetrics`), so only the
   current frame's are resident; when they were all precomputed this term was 3.1 GB on its own.
   Measured: **4.65 GB peak RSS** for a single slice including the Torch/CUDA context, down from
   ~7.0 GB. Cost is linear in `n_workers` — 4 slices plus the 3.4 GB label buffer ≈ 15 GB on a
   31 GB box (~6 is the ceiling), where 8 under eager metrics wanted ~40 GB and OOM-killed the
   kernel.
3. **Per-slice results are dropped by default.** Only `instances` (copied into the single
   `[T,Z,H,W]` int32 output buffer, then Z-stitched in place) and `num_cells` are kept; the prob maps
   and regionprops of each frame are released as the slice finishes. Passing `keep_results=True`
   retains them for inspection and costs a further ~2× the output buffer plus one regionprops list
   per frame (T×Z of them) — it returns `None` otherwise.

If RAM is still short, cut `T` (segment a temporal window) rather than raising `n_workers`. Note that
per-chunk temporal windows change the normalisation statistics — `normalize_and_project` and
`normalize_metric` take percentiles over the frames given — so labels from a chunked run are not
bit-identical to a whole-movie run.

### Speed: keep per-label work off the whole frame

Frames carry **~600 labels**, so any `for label in np.unique(instances)` loop that touches the whole
frame costs 600 × H×W. Profiling `predict_frame` on a real 531×586 frame found four such loops
accounting for ~90% of the 6.6 s/frame — whole-frame `binary_fill_holes` per label (45%),
whole-frame `distance_transform_edt` per fragment (30%), `instances == label` size/reindex passes
(9%), and a whole-frame `components == comp_id` per connected component in the seed-guarantee step.

All four are now scoped to per-label bounding boxes (`scipy.ndimage.find_objects`) or replaced by a
single `bincount` + lookup-table gather, giving **6.62 → 0.78 s/frame (8.5×)** with **bit-identical
labels** — one z-slice at T=180 went from ≳20 min to **142 s**. The rewrites and their equivalence
arguments are pinned in `tests/test_segment_localised.py`, which keeps the previous implementations
verbatim as the oracle. If you add per-label post-processing, follow the same rule: get the label's
box from `find_objects` and work inside it. Two things that legitimately need the whole frame:
`_compute_fragment_affinity` (means over every pixel of a fragment) and the embedding blur (one
`gaussian_filter` call with `sigma=(s, s, 0)` over `[H,W,D]`).

**`merge_max_distance < 1.0` disables fragment merging entirely.** `distance_transform_edt(~mask)` is
0 on the fragment and ≥1 off it, so a threshold below 1 selects only the fragment's own pixels, which
`& ~mask` then removes — the candidate set is always empty. The CMA-ES-tuned `BEST_PARAMS` used in
`notebooks/pipeline_consensus.ipynb` set it to **0.6198**, so with those params
`_merge_split_instances` only drops small components, and the `merge_affinity_threshold` mitigation
for Y-cell splitting is inert. Raise `merge_max_distance` to ≥1.0 before tuning any merge threshold.
(This was previously hidden by cost: the step still spent 30% of the runtime computing whole-frame
distance transforms that could never produce a candidate.)

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
  Mitigate with `merge_affinity_threshold > 0.90` — but **only after** raising `merge_max_distance` to
  ≥ 1.0, otherwise no merge candidate is ever generated and the threshold does nothing (see
  *Speed: keep per-label work off the whole frame*). The notebook's tuned `BEST_PARAMS` sit at 0.6198.
- **Oversegmentation** — raise `embedding_blur_sigma` (2.0–3.0) or `temporal_weight` in training
  (3.0–4.0).

---

## Appendix: flow-metric reference

Folded in from the former root-level `QUICK_REFERENCE.txt`. The 15-metric set produced by
`prepare_data_for_unet(frames, temporal_scales=[1,2,4,8])`, and how to select a subset via
`metrics_to_tensor(temporal_metrics, selected_keys=...)`. (List regenerated from
`flow.py::extract_temporal_metrics` — the emitted keys, not an idealised set.)

### The 15 metrics
- **4 multi-scale magnitudes** — `mag_1` (frame-to-frame, fine but noisy), `mag_2`, `mag_4`,
  `mag_8` (longer gaps, less noise, more persistent patterns).
- **2 temporal dynamics** — `acceleration` (change in flow magnitude across scales),
  `direction_stability` (cosine between the coarse- and fine-scale flow vectors, clipped to [0,1]).
- **1 cumulative displacement** — `cumulative_mag` (total movement over the window; on its own
  separates fast / slow / stationary cells).
- **3 deformation** — `divergence` (∂u/∂x + ∂v/∂y, expansion/compression), `vorticity`
  (∂v/∂x − ∂u/∂y, rotation), `strain` (symmetric strain-rate-tensor magnitude).
- **1 structural** — `edge_strength` (structure-tensor λ₁−λ₂ edge measure on the image).
- **3 flow↔image alignment** — `flow_structure_alignment` (|cos| of flow vs image gradient),
  `normal_flow` and `tangential_flow` (flow components ⟂ / ∥ to the image gradient).
- **1 target** — `cell_boundary_likelihood` (weighted blend used as the boundary prior).

Why multi-scale: it turns "is this pixel moving *now*?" (noisy, ambiguous) into "what is this
pixel's motion *signature* over time?" (robust) — which is what makes cells separable without
bright reporters. See `FAQ.md`.

### Parameters
- `temporal_scales`: `[1,2,4]` faster (testing) · **`[1,2,4,8]` recommended** · `[1,2,3,4,5]`
  more granular / slower.
- `cumulative_window`: `3` fast · **`5` recommended** · `7` more temporal context.

### Metric-selection presets
- **Minimum (5)**: `mag_1`, `cumulative_mag`, `edge_strength`, `tangential_flow`,
  `cell_boundary_likelihood`.
- **Recommended (11)**: the 4 `mag_*`, `direction_stability`, `cumulative_mag`, `divergence`,
  `vorticity`, `strain`, `edge_strength`, `cell_boundary_likelihood`.
- **All (15)**: default (no `selected_keys`).

The UNet's `in_channels` must match the metric count chosen (plus the frame channel — see the
data contract in `docs/ARCHITECTURE.md`).

### Cost & troubleshooting
- Metrics are computed **once** and reused across all training epochs — `prepare_data_for_unet`
  returns a lazy `TemporalMetrics` sequence, and the training entry points
  (`prepare_data_for_unet_batch_4d` / `prepare_data_for_unet_batch`) materialise it with `list(...)`
  so the Dataset is not recomputing planes every epoch. Single-pass consumers (segmentation
  inference) iterate it lazily instead — that is what keeps the 4D path in RAM. Multi-scale is
  markedly slower than frame-to-frame; drop scale 8 (`[1,2,4]`) if too slow.
- All-zero metrics → check frame normalisation to [0,1]
  (`(f - f.min()) / (f.max() - f.min() + 1e-5)`).
- Out of memory → fewer temporal scales.
- No improvement → verify inputs; sweep `cumulative_window` (3/5/7) and `temporal_scales`.
