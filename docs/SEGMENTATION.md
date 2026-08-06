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

### One frame of a window — `flow_metrics_for_frame`

`prepare_data_for_unet` computes flows for **every** frame because training consumes every frame.
A tiled segmentation run consumes **one**: it reads a window around timepoint `t`, segments `t`,
then moves on — and at scale 8 consecutive windows share 16 of 17 frames. `flow_metrics_for_frame`
computes only what `extract_temporal_metrics` actually indexes (one flow per scale, plus the
cumulative sum) — 9 Farneback calls against 53 on a 17-frame window.

It is an optimisation, not a second feature set: `tests/test_flow_metrics_for_frame.py` asserts it
equals `prepare_data_for_unet(window, ...)` at that frame plane-for-plane, at every position
including the truncated ends. That equality is load-bearing, because the metric keys are a silent
train/inference coupling (see *Known issues* and `tests/test_flow_metric_count.py`).

Pass `value_range=(lo, hi)` when tiling. Training scales intensities by the whole movie's min/max;
without the override each tile-window gets its own scale, so the same frame changes contrast
depending on which window read it — and the structure-tensor planes read the scaled frame directly.

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

- **`ForegroundLoss`** — brightness blurred to cell scale, p99-normalised. **The prob-head
  supervisor to use**, on confetti and non-confetti data alike; see *What confetti actually
  contributes* below for why it replaces both of the two options under it.
- **`IntensityLoss`** — rewards bright pixels, local contrast, edge strength. No labels. Its target
  has no cell-scale structure and trains the prob head toward speckle (measured: median blob size
  **1 px**). Superseded by `ForegroundLoss`.
- **`ConfettiForegroundLoss`** — the same objective as `ForegroundLoss` but keyed on dominant-colour
  confidence. Measured to produce the same target to r ≥ 0.99, so it buys nothing over brightness
  and requires confetti input channels that are zeros at inference.
- **`TemporalMetricsLoss`** — hard contrastive: pixels with similar optical-flow metrics should
  be close in embedding space.
- **`ConfettiBoundaryLoss`** — pushes embeddings apart across a colour boundary. Off by default;
  needs both crowded confetti data and a lower `min_confidence` to fire at all.
- **`VarianceMetricsLoss`**, **`WarpConsistencyLoss`** — alternative/auxiliary objectives.

Default weights: `intensity_weight=1.0, temporal_weight=2.0`. Raise `temporal_weight` (3.0–4.0)
to reduce oversegmentation. For a no-confetti run, prefer
`intensity_weight=0.0, foreground_weight=1.0`; when confetti metrics are available, pass them with
`variance_as_input=False` so they supervise without becoming dead input channels.

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

**`predict_temporal_volume`'s flow defaults do not match the training defaults, and the mismatch is
silent.** It defaults to `temporal_scales=[1,2,4], cumulative_window=2`, which yields **14** flow
metrics; `prepare_data_for_unet` (and therefore `prepare_data_for_unet_batch_4d`, i.e. training)
defaults to `[1,2,4,8], cumulative_window=5`, which yields **15** — the extra one is `mag_8`.

`predict_frame` stacks metrics in `sorted(key)` order, so a missing `mag_8` does not leave a hole at
its own position: every metric sorting after it (`normal_flow`, `strain`, `tangential_flow`,
`vorticity`) shifts down one channel into a slot the model learned as something else, and
`n_variance = max(0, num_metrics - len(metric_list))` then zero-fills one channel at the *end*. The
shapes match, so nothing raises — the model is simply fed misaligned inputs.

Always pass the values training used explicitly:

```python
inf3d.predict_temporal_volume(volume, ch_indices=[...],
                              temporal_scales=[1, 2, 4, 8], cumulative_window=5)
```

The metric count also depends on `T`: fewer than 9 frames drops `mag_8` regardless of the scales
requested, so a short test sequence silently trains/infers on 14 channels. Check the
`Input channels:` line the trainer prints against `len(metrics_dict)` at inference.

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

### The foreground is speckle, and it is the real bottleneck

The prob head resolves cells, but sits on a 1–3 px noise floor that also crosses
`prob_threshold`. Measured on a real frame (T=180 prep): `prob > 0.4` gives **4518 blobs,
median 3 px, 99% under 100 px** covering 21% of the frame, which region growing then carves into
~1000 labels. This — not the inference parameters — is why ~86% of detections are fragments.

Two fixes were measured:

- **Raising `prob_threshold` does not work.** 0.4 → 0.9 drops fragments only 88% → 58% while
  labels collapse 719 → 40: it discards real cells as fast as noise, because most cells' probability
  overlaps the background texture. (`prob_threshold` is also commented out of `PARAM_NAMES`, so it
  has never been tuned — but 0.4 is already the best of the values tested.)
- **`prob_blur_sigma` (new, default 0.0 = off) does suppress the speckle.** Cells (~15–20 px) and
  speckle (1–3 px) differ by scale, so a blur separates them where a threshold cannot: on the raw
  mask, σ=3 cuts blobs 3612 → 142 and raises median size 3 → 64 px with the count of cell-sized
  (≥100 px) blobs unchanged at ~57.

  End-to-end it is a **trade, not a free win**: median label size 24 → 89 px and fragments 88% → 58%,
  but coverage of the colour-carrying area falls 25% → 19% and the share of labels straddling two
  colours rises 39% → 54%, so `score_label_size_confetti` nets ~7% *down*. Whether that trade is
  worth it is a judgement about what tracking needs downstream.

  **Compare it at matched foreground area, not at matched threshold** (re-measured 2026-08-01 on
  the clean re-import, 4 movies × 2 z-planes × 3 frames). A blur lowers every prob value, so a
  fixed threshold silently runs it at a stricter operating point and flatters or damns it
  arbitrarily. At equal area σ=1 beats σ=0 outright — 70.5% vs 68.1% recall at 1% foreground with
  half the blobs (47 vs 104); σ=3 trades the low-area end (60.9%) for the high (90.7% vs 87.8% at
  5% area) with 16× fewer blobs.

- **Cleaning the INPUT beats cleaning the prob map, and the two compose.** Restoring the mean
  projection and applying it back as a per-pixel scalar gain
  (`denoise.denoise_preserving_ratio`) gives 75.9% recall at 1% foreground area vs raw's 68.1%,
  with 63 blobs vs 104 — and adding σ=1 on top is the best measured combination (77.2%, 42 blobs).
  Over 35 conditions at `prob_threshold=0.6` it is 89.6% recall / 145 blobs against raw's 85.7% /
  745. Most of that comes from *smoothing the projection* rather than from the restoration net —
  a plain `gaussian_filter(proj, 1)` reaches 87.5% / 158. Full measurements, and the reason the
  gain form is the one that keeps confetti identity intact, in
  [`docs/todo/DENOISE_PLAN.md`](todo/DENOISE_PLAN.md) → *Ratio-preserving restoration*.

The number that frames all of this: **labels cover only ~25% of the colour-carrying area even at
σ=0.** Three-quarters of the visible cell material is unlabelled, so coverage — not fragment
cleanup — is the deficit to attack. That points at the training signal: the prob-head target is
`0.5*bright + 0.3*local_contrast + 0.2*edge`, three grayscale texture statistics, with neither
confetti (identity) nor flow (separation) connected to it.

### Merging of touching cells — finding 3 RETRACTED, see the notice below

> **RETRACTION (2026-08-05).** Finding 3 below ("the embeddings are blind to cell boundaries",
> cosine 0.945 / 0.943 / 0.920, Cohen's d = 0.22, n=45) was measured on `crowdgen` synthetic scenes.
> Two sections further down, and commit `2e2f05d`, establish that **`crowdgen` cannot validate
> separation — superposed copies never interact.** Time-offset copies of the same movie are
> statistically independent crops: they never repel, never deform against each other, and there is no
> membrane interface. The flow field at a synthetic "contact" is two independent motions summed, which
> is not what a real contact is, so an embedding cosine measured there has no physical meaning.
>
> **What survives:** the *symptom* — segmentation merges **86.7% of 465 real** touching
> different-colour pairs (see *A real validation set*). That was measured on real movies and stands.
>
> **What does not:** the *mechanism*. There is no established evidence that the embeddings are
> boundary-blind, and therefore no basis for the conclusion that no post-processing rule could help,
> nor for the `ConfettiBoundaryLoss` post-mortem which explains that loss's failure with the same
> synthetic numbers (`d 0.22 → 0.40`). Combined with the unreachable `min_confidence` gate documented
> in the amendment above, there are two independent reasons to treat the boundary approach as
> **untested rather than refuted**.
>
> Settling it needs real confetti with real cell–cell repulsion, and more than 3 colours (with 3,
> ~270 cells share each — see `docs/FUTURE.md`). Findings 1 and 2 are also `crowdgen`-based and carry
> the same caveat; finding 2's stage attribution is about where merges occur mechanically, which is
> less sensitive to whether the contact is physical, but it has not been re-checked on real pairs.

Measured 2026-08-04 against synthetic crowded ground truth (`crowdgen`: spatially-shifted,
time-offset copies of real AF+drift confetti superposed, per-cell GT; 2 movies × 2 densities ×
3 frames). Three findings, in the order they were established:

1. **`confetti_blur_sigma` is a merge↔split dial, not a separation knob.** Each model at its own
   best `prob_threshold`: blur 2.0 → F1@.35 65.2% / 4.5% merged / 6.2% split / mask area 0.94× GT;
   blur 1.0 → 66.6% / 2.4% / 9.2% / 0.71×; no blur → 50.3% / 0.1% / 29.2% / 0.52×. Sharpening the
   target buys fewer merges by shattering cells and shrinking masks. **1.0 is now the default**
   (`loss.ConfettiForegroundLoss`, `train.train_with_metrics`) because it wins on F1 outright while
   roughly halving merges. A no-blur model is *not* broken — its prob map is clean and cell-shaped —
   but it never reaches 0.5, so a fixed threshold reports zero labels.

2. **Most merging is downstream of the prob head.** Replaying `predict_frame` stage by stage over
   137 adjacent GT pairs (45 merged): **56% of merges happen in `_merge_split_instances`, 33% in
   `_grow_regions_fast`, 11% at seeding** (one local maximum for the pair).

3. **The embeddings are blind to cell boundaries, so no post-processing rule can fix this.**
   `_compute_fragment_affinity` cosines *whole-fragment mean* embeddings; restricting the mean to a
   band at the contact — where a boundary should show as a step — made merging **worse** (5.1% →
   5.9% at band=2, and no band width helped). Setting `prob_weight=0`, which stops a bright contact
   from lowering the growing threshold, did not move merges either. Measuring the embedding field
   directly, with segmentation out of the way: cosine across a contact is **0.945 within one cell**,
   **0.943 between adjacent same-colour cells**, **0.920 between adjacent different-colour cells**
   (Cohen's d = 0.22 vs the within-cell ceiling, n=45 different-colour pairs). A colour boundary is
   nearly indistinguishable from cell interior.

   This follows from what supervises the embeddings. `VarianceMetricsLoss` is **negatives-only** and
   mines the k pixels *farthest* in metric space within a window — the easy case, cells that are
   already far apart — while `TemporalMetricsLoss` *pulls together* pixels with similar motion, which
   is exactly what two touching cells drifting as a pair have. Adjacent pixels straddling a contact
   are never presented as a negative. Fixing this is a training change, not an inference one.

**`crowdgen` cannot validate anything flow-dependent.** Superposed copies pass *through* each
other — no deformation, no slowing, no avoidance — so at a synthetic contact the flow field is
merely two independent motions overlaid. The interaction signature that would let flow separate
touching cells is absent by construction. Use it for target-shape comparisons (the blur ranking
below is one, scored on identical scenes) and never for judging separation. The scenes also reach
only ~4% GT area coverage, so absolute merge rates from them are optimistic — measured against
real contacts, by about 3x.

### A real validation set: touching cells of different colour

The synthetic scenes superpose *time-offset copies of the same movie*, so every touching pair
moves independently by construction, which looked like it would overstate how separable real
contacts are. It does not — checked directly, and the concern was backwards.

Wherever two differently-coloured cells genuinely touch in the confetti movies, **colour is free
ground truth and the model never sees it** (the 3 variance channels are zero-filled at inference).
Mining all 9 movies across every valid z-plane and three 20-frame windows yields **465 real
touching different-colour pairs** over 431 frames — real cells, real scanner, no synthesis and no
domain gap. (mem-TOM is not usable for this: different cells *and* a different microscope.)

|  | relative motion above the flow noise floor | co-moving |
|---|---|---|
| synthetic (`crowdgen`) | 68% | 32% |
| **real confetti pairs** | **75.7%** | **24.3%** |

Real touching cells move apart *more* than the synthetic copies do (median 2.10 px/frame against a
1.28 px/frame within-cell noise floor). This sets the ceiling on any flow-only boundary signal:
about **76% of real contacts are in principle separable**, and the remaining ~24% are co-moving,
where nothing in greyscale + flow distinguishes the contact from cell interior — colour is the only
cue and it is unavailable at inference. Confetti can supervise which *flow* patterns mean
"boundary"; it cannot smuggle colour into inference.

Scored on that set, the current model (confetti p99, blur 1.0, `prob_threshold=0.40`) **merges
86.7% of real touching different-colour pairs**:

| of all 465 real contacts | |
|---|---|
| correctly separated | 13.3% |
| **merged, but the motion IS resolvable** | **64.1%** ← recoverable |
| merged and co-moving | 22.6% ← hard floor |

Take that seriously against the synthetic numbers above, which put merges at 2.4% of predicted
labels (32.8% of adjacent GT pairs). The definitions differ — synthetic pairs are whole GT cells,
real pairs are thresholded colour components — but the gap is far too large to be definitional.
Real cells in contact interdigitate in a way randomly-pasted copies do not, so **the synthetic
scenes badly understate the merge problem** and should not be used to judge it.

The mined pairs were validated before drawing that conclusion, since an aggregate number is exactly
what hid the earlier broken GT: median size ratio between the two components 0.67 (no rim
satellites), median colour purity of the weaker member 1.00 (nowhere near the 1/3 tie point where a
noise-flipped dominant channel would sit), 0% excluded by either filter, and six examples plotted
and eyeballed. The 86.7% is unchanged under every filter.

### `ConfettiBoundaryLoss`: right mechanism, not enough data (2026-08-04)

Built to close exactly the gap above — push embeddings apart across a confetti-colour boundary,
pull them together within one colour, mining contacts explicitly rather than sampling windows and
hoping. Shipped **off by default** (`boundary_weight=0.0`). It made segmentation *worse*, and the
reason is worth recording because it is a data limit, not a design flaw.

Trained 80 epochs alongside the confetti p99 loss (blur 1.0); the term converged 0.199 → 0.008.
Scored on the 465 real pairs:

| | merged | correctly separated |
|---|---|---|
| baseline (confetti blur 1.0) | 86.7% | 13.3% |
| + boundary loss | **89.7%** | 10.3% |

The embedding measurement below is **`crowdgen`-derived and retracted** — see the retraction notice
under *Merging of touching cells*. The 86.7% merge rate above is real; this explanation for it is
not established, so "it made segmentation worse" stands as an observation without a mechanism.

The embedding measurement explains it. Under the real inference condition (variance channels
zero-filled) the boundary model **did** improve separation, d 0.22 → 0.40 — and this is *not* a
shortcut through the confetti input channels: supplying them changes d only 0.40 → 0.44. But the
gain is entirely in the wrong half:

| | within cell | different colour |
|---|---|---|
| baseline | 0.945 | 0.920 |
| + boundary loss | **0.965** | 0.926 |

Within-cell coherence tightened; different-colour cosine moved the *wrong way*. Tighter
within-cell coherence is precisely what lets region growing expand further and raises
`_compute_fragment_affinity`, so merging went up.

The negative half of the loss is starved. Counted over 42 real training frames exactly as
`forward` mines them:

| | |
|---|---|
| positive pairs (same colour, adjacent) | 112,256 |
| **negative pairs (different colour, adjacent)** | **123** |
| ratio | 913 : 1 |
| frames containing *any* negative pair | 11/42 (26%) |

For three-quarters of training steps the term carries no boundary signal at all and acts purely as
a coherence loss. **The sparse confetti movies contain ~0.1% of the signal this loss needs.** The
mechanism is sound and ready; genuinely crowded confetti data is the prerequisite, and it is also
the only data containing the cell–cell avoidance behaviour that would make the flow signature
learnable in the first place. Revisit when the crowded confetti mice are available.

#### Amendment (2026-08-04): it is not only a data limit — `min_confidence` is the larger lever

The conclusion above ("a data limit, not a design flaw") is half right. Re-measured on
`ccidDriftCorrected` mid-z, 6 frames, counting pairs exactly as `forward` mines them:

| `min_confidence` | negative pairs / frame (r0hufV) | negative pairs / frame (fXgbTl) |
|---|---|---|
| **0.5 (shipped)** | **0** | **0** |
| 0.4 | 22 | 0 |
| 0.35 | 77 | 30 |
| 0.3 | 160 | 107 |

**The default gate sits above the achievable confidence range.** `_colour` returns
`dominance_share × clamp(brightness/p99)`, and the dominance share is pinned near its `1/C` floor
(see *What confetti actually contributes* below), so the confidence maxes out at **0.556** on
r0hufV and **0.457** on fXgbTl — on fXgbTl the 0.5 gate therefore admits *nothing at all*, and the
term is an exact no-op rather than a starved one. Dropping the gate to 0.3 recovers 160 negatives
per frame on the same data: a ~50× swing from a parameter, against the ~3/frame that the 42-frame
count above attributed to data scarcity.

So both effects are real, and the gate is the one that is cheap to fix. Two things follow for when
the crowded confetti data lands:

1. **Do not turn `boundary_weight` on without lowering `min_confidence`** — at 0.5 the term is
   753 positives and 0 negatives per frame on real confetti, i.e. it *only* pulls embeddings
   together, which is the same fusion this loss was written to counteract in `TemporalMetricsLoss`.
   That is consistent with the measured outcome above (within-cell cosine tightened 0.945 → 0.965,
   different-colour barely moved) — the loss did what a positives-only objective must do.
2. **Rebalance the halves.** Even at `min_confidence=0.3` the ratio is 4089 positives to 160
   negatives; with `pos_weight=0.3` the pull still outweighs the push ~7.7:1. Crowded data raises
   the negative count but will not by itself fix a ratio that starts two orders of magnitude out.

A peakier, more local softmax helps the co-expressed data specifically (raw-channel dominance at
`softmax_temp=0.05, pool_radius=2`: 90 negatives/frame on fXgbTl vs 2 on r0hufV) because its
markers are spatially interleaved rather than sparse.

## What confetti actually contributes (2026-08-04)

Confetti plays **four separate roles** in this pipeline, and they were being reasoned about as one
thing. Separated and measured, three of them do no work:

| role | where | status |
|---|---|---|
| model **input** channels | `train.TemporalDatasetWithAugmentation` concatenates `softmax_ch_*` onto `frame_and_metrics` | **always zeros at inference** |
| prob-head **supervision** | `loss.ConfettiForegroundLoss` | colour term **inoperative** (r ≥ 0.99 vs colour-blind) |
| embedding **supervision** | `loss.ConfettiBoundaryLoss` | 0 negative pairs at the shipped gate (above) |
| embedding **contrastive** | `loss.VarianceMetricsLoss` | negatives-only, never presents a contact |

This matters because the next data is multi-marker, not confetti. `zolIMa/fXgbTl` is
SHG / nuc-GFP / mem-Tom / CD169-Kat, where **the confetti premise is simply false**: nuc-GFP and
mem-Tom label the nucleus and the membrane of the *same* cell, so a correct segmentation spans two
colours, and SHG is collagen rather than a cell at all.

### The input channels are zeros at inference (but that is not what costs accuracy)

`TwoPassSegmentationInference.predict_frame` — the production path — takes no `variance_metrics`
argument, and `LearnedAffinityInference` zero-fills whatever the model expects beyond what it was
given. Training presents each variance channel with `variance_dropout_p=0.5` as a plain mask with no
inverted-dropout rescale, so all three zero at once is a **12.5%** corner of the training
distribution but **100%** of inference.

`variance_as_input=False` (new) keeps confetti as *supervision only*, which is all three confetti
losses need — they read `variance_metrics_norm` directly, not the input tensor. Previously the two
were coupled: you could not have the supervision without the dead input channels.

Worth being precise about the benefit: removing the channels is right because a model should not be
trained on inputs it will never see, and it drops 3 of 19 input channels. It is **not** measurably
more accurate — measured, the opposite (see *Deconfounding* below). Do not justify it on accuracy.

### The colour term in `ConfettiForegroundLoss` does no work — even on confetti

Its target is `max_c softmax_ch_c`, which is `dominance × brightness`. Measured against the same
target built from **brightness alone** (`loss.ForegroundLoss`), on `ccidDriftCorrected` mid-z,
6 frames:

| data | pearson r | foreground IoU @0.4 |
|---|---|---|
| kSUFux/r0hufV — real confetti | **0.9993** | 96.3% |
| zolIMa/fXgbTl — co-expressed markers | 0.9986 | 94.1% |
| 4kS67f/3w4IY5 — two cell types | 0.9906 | 83.0% |

The mechanism: `compute_variance_metrics` softmaxes across channels at `temp=0.3` over
Gaussian-pooled (`pool_radius=5`) intensities, then rescales **each channel independently** with
`normalize_metric`, which undoes the cross-channel comparability the softmax just established. The
dominance share is left pinned near its `1/C` floor — in the foreground on real confetti (C=3,
floor 0.333): p5 0.356, median 0.397, p95 0.528, and **0% of foreground pixels are unambiguously
one colour** (dominance > 0.8). A near-constant multiplier cannot carry identity.

So the reported win of `ConfettiForegroundLoss` over `IntensityLoss` (2834 blobs → 87, median
36 → 90 px) is attributable to **the cell-scale blur and the p99 rescale**, not to confetti. Both now
live in the shared `loss._blob_target`, so the two losses cannot drift apart.

A related consequence: the same per-channel rescale corrupts the colour *label*. Against the argmax
of raw per-channel intensity, on pixels with an unambiguous raw colour, `ConfettiBoundaryLoss._colour`
agrees only **52.7%** of the time on r0hufV and **38.3%** on fXgbTl, with a systematic channel swap.
Its `min_confidence` gate is what rescues it — of the pixels it admits, 93–95% are correct — but see
the amendment above for how little that leaves.

### Trained models: `ForegroundLoss` is the no-confetti path

Trained on the one 16-bit crop available (`zolIMa/fXgbTl` `ccidDriftCorrected`, 3 z-planes × 31
frames = 93 frames, 72 train / 21 held out, 30 epochs), scored on 12 held-out frames **at matched
foreground area** — never at matched threshold, since at `prob_threshold=0.4` these arms sit at
34.5% / 9.9% / 6.2% foreground area and any statistic read there compares operating points:

| arm | @1% area: blobs / median px | @2% | @5% |
|---|---|---|---|
| `IntensityLoss` | 861 / 1.0 | 1473 / 1.0 | 2772 / 1.0 |
| **`ForegroundLoss`** | **32 / 18.0** | **59 / 11.5** | **118 / 7.0** |
| `ConfettiForegroundLoss` | 66 / 1.9 | 118 / 1.7 | 248 / 1.0 |

`IntensityLoss` reproduces its documented failure exactly — a median blob size of **1 px** at every
operating point, i.e. pure speckle. `ForegroundLoss` gives 27× fewer blobs and an 18× larger median
blob, and also recovers more of the cell material (58.5% vs 46.0% at 1% area).

#### Deconfounding: the confetti arm differed in three ways at once, and neither obvious story held

The confetti arm above carried the confetti prob-head loss **and** `VarianceMetricsLoss` on its
embeddings **and** three input channels that are zero at inference. The heads share a trunk
(`model.encode_decode`), so the embedding loss moves the prob map too, and the three could not be
told apart. Two further arms, identical except as named:

| arm | prob-head loss | `VarianceMetricsLoss` | confetti input channels | @1%: blobs / med / cov |
|---|---|---|---|---|
| `foreground` | ForegroundLoss | — | — | **32 / 18.0 / 58.5%** |
| `confetti` | Confetti | yes | yes | 66 / 1.9 / 55.6% |
| `confetti_novarin` | Confetti | yes | **no** | 83 / 3.8 / 42.5% |
| `foreground_varloss` | ForegroundLoss | yes | no | 60 / 8.8 / 36.6% |

Two things this refutes, both of which were plausible:

* **The input-mismatch story is not supported.** Removing the zero-at-inference channels
  (`confetti` → `confetti_novarin`) made the prob map *worse*, not better — 66 → 83 blobs, coverage
  55.6% → 42.5%. An earlier draft of this section attributed `ForegroundLoss`'s win to that
  mismatch; that attribution is **withdrawn**. Dropping the channels is still right on grounds of
  train/inference honesty and model size, but it is not what produces the gap.
* **`VarianceMetricsLoss` is the expensive term here.** Adding it to the foreground arm
  (`foreground` → `foreground_varloss`) cost more than any confetti manipulation: 32 → 60 blobs,
  median 18.0 → 8.8, coverage 58.5% → 36.6%.

Holding everything else fixed, the confetti prob-head loss versus brightness
(`confetti_novarin` vs `foreground_varloss`) is a wash rather than a win for either: fewer and
larger blobs for brightness (60 / 8.8 vs 83 / 3.8), slightly better cell-material coverage for
confetti (42.5% vs 36.6%). Consistent with the two targets being r = 0.9986 identical.

**So the best configuration on this data is the simplest one: `ForegroundLoss` +
`TemporalMetricsLoss`, with `intensity_weight=0.0` and `variance_weight=0.0`.** No confetti anywhere,
and one fewer loss than the incumbent.

#### Seed check: the margin is bigger than seed-to-seed variation

A single-seed margin is not a result, and this one is being proposed as the default. Three seeds per
configuration, same 12 held-out frames — `fg` = ForegroundLoss + Temporal, `conf` = the incumbent
(ConfettiForegroundLoss + VarianceMetricsLoss + confetti input channels):

| area | | blobs (spread) | median px (spread) | cell-material cov (spread) |
|---|---|---|---|---|
| 1% | **fg** | **36** (28–46) | **15.2** (9.2–23.5) | **59.4%** (59.4–59.4) |
| 1% | conf | 50 (44–55) | 6.6 (4.6–8.3) | 52.8% (46.9–56.1) |
| 2% | **fg** | **68** (51–82) | **11.8** (7.9–18.7) | **84.2%** (84.1–84.3) |
| 2% | conf | 99 (97–102) | 3.1 (2.8–3.4) | 78.6% (74.8–80.7) |
| 5% | **fg** | **132** (90–166) | **8.1** (4.2–14.7) | **98.6%** (98.4–98.8) |
| 5% | conf | 212 (196–224) | 2.0 (1.9–2.0) | 97.2% (96.3–97.8) |

**The median-blob-size ranges do not overlap at any operating point**, and coverage is higher for
`fg` at all three. Two honest caveats: `fg` is markedly *less stable* across seeds on median blob
size (9.2–23.5 vs 4.6–8.3 at 1% area), and blob-count spreads do touch at 1% (46 vs 44). Both
argue for re-checking on the full 16-bit set rather than treating these absolute numbers as final —
this is one crop, 3 z-planes, 30 epochs.

#### `prob_blur_sigma` becomes a choice rather than a repair

`prob_blur_sigma` exists because the prob head sat on a 1–3 px speckle floor that a threshold cannot
separate from cells. Supervising the target at cell scale removes most of that floor at source, so
the blur is no longer fixing a defect — it is trading blob count against blob size from an already
clean starting point. Same 12 held-out frames, at 1% foreground area:

| `prob_blur_sigma` | `ForegroundLoss`: blobs / median | `ConfettiForegroundLoss`: blobs / median |
|---|---|---|
| 0.0 | **32 / 18.0 px** | 66 / **1.9 px** |
| 1.0 | 21 / 36.2 px | 21 / 31.5 px |
| 3.0 | 13 / 91.0 px | 12 / 82.3 px |

The confetti arm *needs* the blur to reach a cell-shaped map at all (1.9 → 31.5 px from σ=1 alone);
the foreground arm starts at 18 px unblurred. Note the two converge once blurred — more evidence that
what the blur and the target-side blur do is the same work, applied at different ends of the network.
Cleaning the target is the better end: it shapes what the model learns instead of smoothing over
what it got wrong.

### What is genuinely lost without confetti

Only one thing: **boundary supervision**. Nothing in greyscale + flow says "these two touching
regions are different cells" for the ~24% of real contacts that are co-moving. That is what
`ConfettiBoundaryLoss` was for, and it needs crowded confetti data (plus the gate fix above) to
work. Everything else confetti was supplying, brightness at cell scale already supplies.

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

> **These presets are not backed by a measurement, and the first ablation contradicts them
> (2026-08-05).** Note also that `metrics_to_tensor(..., selected_keys=...)`, referenced above as the
> way to apply them, **does not exist** in the codebase — subset the metrics dicts directly.
>
> Ablated on `zolIMa/fXgbTl` mem-Tom + `coastal.smooth` (one channel, no confetti;
> `ForegroundLoss` + `TemporalMetricsLoss`, 30 epochs, seed 42, 12 held-out frames), scored at
> matched foreground area. `split ratio` = instances ÷ connected components of the same mask, so
> 1.0 would mean region growing never divided a component:
>
> | metric set | ch | mask comps | instances | split ratio | @10% median blob |
> |---|---|---|---|---|---|
> | none (image only) | 0 | 110 | 65 | 0.60 | 13.3 px |
> | **4 magnitudes** | 4 | **85** | 59 | **0.69** | **24.6 px** |
> | minimum 5 | 5 | 148 | 63 | 0.43 | 4.5 px |
> | recommended 11 | 11 | 166 | 61 | 0.37 | 2.1 px |
> | all 15 | 15 | 176 | 58 | 0.33 | 1.7 px |
>
> **More metrics gave a more fragmented prob map, and all five arms converged on ~58-65
> instances.** The metric set barely moved the final count; it moved how much region growing had to
> undo. Four magnitudes beat both ends — unexplained, and n=1 seed, so treat the ordering as
> provisional. But "more is better" is not supported, and `All (15)` should not be the default
> without a measurement.
>
> **The split ratio is below 1.0 in every arm**: region growing net-*merges* components and never
> divides them, so the embeddings are not separating touching cells on this data. Independent of the
> retracted `crowdgen` finding, and reached on real data — but one image, one seed, and mem-Tom here
> may simply not have many genuine contacts.
>
> Caveat on interpreting the `none` arm: with no metrics there is no `TemporalMetricsLoss`, so its
> embeddings are unsupervised. That it still reaches 65 instances says more about how little the
> embedding pathway contributes than about flow specifically. A cleaner test would keep
> `TemporalMetricsLoss` while removing the metrics from the *input* only, which needs the same
> input/supervision split that `variance_as_input` gave the confetti channels.
>
> Two bugs were fixed to make this ablation runnable at all, both reachable only with an empty
> metric set: `_contrastive_metric_loss` returned a float instead of a tensor, and
> `predict_frame` fabricated a zero metric channel when handed none.

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
