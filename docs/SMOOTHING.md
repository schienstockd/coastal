# Smoothing — `coastal.smooth`

Model-free input smoothing: a Gaussian in space, a running statistic in time, **one shared kernel
applied to every channel**. No model, no weights, no download.

Distinct from [`coastal.denoise`](../coastal/denoise.py), which holds the **learned** Cellpose-3
restoration (the CPnet port) plus the ratio-preserving gain. The split is deliberate: three
different operations were converging on the word "denoise" — a trained net, a per-pixel gain, and a
pair of local averages. Nothing in `smooth` estimates or models noise; it averages neighbours.

## What it is for

**Photon-limited** acquisition — a resonance scanner's short dwell time gives single-digit photon
counts, so a channel is a delta at zero plus a thin tail. Measured on `zolIMa/fXgbTl` (16-bit,
31×4×31×420×441, 15 s/frame): **86–95% of voxels per channel are zero**, and the observed maximum is
**522 of 65535** — the data occupies 0.8% of its range. A 16-bit re-import changed nothing, because
there were never more than ~500 photons. **Bit depth was never the constraint.**

That breaks any statistic assuming a background *population* exists to be found. cecelia's AF
correction derives its background with a triangle threshold, and on this data it lands **inside the
signal**: the reference channel kept **8.6%** of its signal past background subtraction.

| channel | raw bg / signal kept | after `smooth_channels` |
|---|---|---|
| nuc-GFP | 40 / 12.6% | 7 / 85% |
| mem-Tom | 47 / 46.3% | 14 / 100% |
| CD169-Kat | 44 / **8.6%** | 6 / **80%** |

## The three invariants

**One shared kernel across channels.** The consumers are cross-channel ratios — cecelia's AF weight
is `b_t^p / Σ b_i^p`, and confetti identity is the channel vector. A *per-channel* transform corrupts
them silently. This is what disqualified the learned net here: its `normalize99` runs per plane per
channel, dividing mem-Tom by ~81 and CD169-Kat by ~35, never undone.

**Spatial before temporal.** A temporal statistic alone keeps **8.5%** of the reference channel —
*worse than no smoothing* (15.4%) — because a median over 3 samples of mostly-zeros is zero. The
Gaussian has to fill the sparse counts first.

**Median in time, never in space.** Both statistics suppress noise; the median keeps masks tight
(area 140 vs 165 for the mean, object count 24 vs 21), because it rejects a cell that moved through
the window instead of averaging it in at partial occupancy. But a *spatial* median deletes the
signal — it is robust to sparse outliers, and here the signal **is** sparse positive counts:
`ball(1)` left the reference channel at **4.1%** kept, worse than raw; `ball(2)` drove background
variance to exactly zero. There is deliberately no spatial-median option.

## Use

```python
from coastal.smooth import smooth_channels

out = smooth_channels(arr, sigma=1.0, frames=3, stat="median",
                      channel_axis=1, time_axis=0)     # (T, C, Z, Y, X)
```

`time_axis` is **opt-in**: `None` disables the temporal term. Guessing it wrong smooths across Z and
silently flattens a stack into a slab, so it is never inferred. `sigma=0` and `frames<=1` are
no-op passthroughs, so a caller can disable either term without branching. Output is float32;
**absolute intensities are not preserved** — absolute-brightness measurements must read the
unsmoothed store.

`spatial_smooth` / `temporal_smooth` expose the halves. `gaussian_restorer`,
`temporal_mean_restorer` and `temporal_median_restorer` are the projection restorers for
`denoise.denoise_preserving_ratio`; they live here because they are model-free, and `denoise`
re-exports them so existing imports keep working.

## Lineage, and a bug not to port

Supersedes R cecelia's `cleanupImages/slidingWindowCorrect.R` + `py/sliding_window_correct.py`,
which took `np.median` over a T window per channel (output value_name `slidingWindow`).

Do **not** port its window arithmetic. `w_start = i - sw; w_end = i + sw; slice(w_start, w_end)` is
half-open, so the window was `2*sw` frames and **off-centre** — and at its default `sw=1` a "median"
of 2 samples is their mean. `frames` here is a full, centred, odd width, and an even value is
promoted rather than shifted.

## Effect on the segmentation prob map — measured, and NEGATIVE on confetti

Do not assume this helps segmentation. Measured 2026-08-04 through the existing sweep harness
(`~/Downloads/TMP/coastal-denoise-experiments/scripts/harness.py`), 4 confetti movies x 3 frames,
**at matched foreground area**, recall against seeds from the raw grey:

| arm | rec@1% | blobs | rec@2% | blobs | rec@5% | blobs |
|---|---|---|---|---|---|---|
| **raw** | **84.2%** | 65 | 90.1% | 219 | **96.1%** | 484 |
| gaussian σ=1 | 79.8% | 57 | 85.5% | 281 | 91.2% | 564 |
| gauss + temporal mean 3 | 75.6% | **42** | 85.0% | 195 | 92.7% | 467 |
| gauss + temporal median 3 | 75.4% | 49 | 83.2% | **189** | 93.3% | **264** |
| `denoise_preserving_ratio` + temporal mean | 81.6% | 50 | **90.8%** | **164** | 96.0% | 287 |

`smooth_channels` **costs 4-9pp of recall** and buys fewer blobs. It does not beat raw, and it does
not beat the ratio-preserving gain. **For the confetti segmentation path, the gain remains the
measured best.**

Two caveats on the run itself, both pointing at unfinished business:

1. **It contradicts `DENOISE_PLAN.md`**, which records `gaussian_filter(proj, 1)` at 87.5% recall
   against raw's 84.5%. This run gives the opposite ordering. One of the two methods is wrong and it
   has not been resolved — treat both numbers as provisional until it is.
2. **`harness.prob_maps` clips to `uint8` at 255 while the grey stack reaches ~937**, so each arm was
   clipped by a different amount (smoothing lowers the peak 937 -> 541). The table above normalises
   each arm to its own p99.9 first; without that, raw led by a further 4-12pp. Every earlier number
   measured through this harness carries the unfixed version.

## Scope caveat

Every number above comes from **resonance, photon-limited, intravital** data. Coastal's segmentation
and `denoise_preserving_ratio` were tuned on **galvo confetti** data, which has far better SNR and a
different constraint (identity *is* the ratio). Treat `smooth_channels` as a general primitive whose
defaults are justified on the photon-limited regime — not as a validated default for confetti until
measured there.

Full measurement record, including four rejected alternatives (the ratio-preserving gain, the
Cellpose net repaired and still dropped, 16-bit, spatial median):
`cecelia-feijoa/docs/todo/SMOOTHING_PLAN.md`.
