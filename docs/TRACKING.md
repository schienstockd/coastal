# Tracking

> Update this file when you change the tracker design, cost terms, or scoring. Log each
> **experiment** (params + result numbers) in [`../TRACKING_SESSION_SUMMARY.md`](../TRACKING_SESSION_SUMMARY.md)
> — that is the running experiment ledger; this file is the durable "how it works" reference.

Track T cells in 3D+T confetti fluorescent microscopy.
Input: `instances_4d [T, Z, H, W]` from segmentation. Output: `{track_id: {t: pos_um}}`.

## Design (`abm.py::track_sequence`)

Constant-velocity Kalman filter + Hungarian LAP assignment per consecutive frame pair. Cost
matrix (every term optional, weighted):

```
cost = dist_cost                          # Mahalanobis, chi²-gated
     [+ w_flow        * flow_warp_cost]   # dense Farneback flow deviation (geometric, no model)
     [+ w_color       * color_cost]       # confetti RGB cosine distance (geometric, no model)
     [+ w_ctx         * ctx_cost]         # ContextAssigner joint transformer (learned)
     [+ w_collective  * collective_cost]  # neighbour-velocity coherence
     [+ w_persistence * persistence_cost] # turn penalty
     [+ w_exclusion   * exclusion_cost]   # contact repulsion
```

Supporting pieces: `track.py` (`Track` dataclass, `compute_3d_centroids`, `extract_cell_colors`,
`extract_cell_intensities`, `ContextAssigner`), `abm.py` (`compute_cell_flow_features`,
`BreadcrumbField`/`CellAgent`/`ABMTracker` breadcrumb-density variant, `stitch_tracklets`).

### Anisotropy is load-bearing
Z = 4.0 µm/px, XY ≈ 0.48 µm/px (ratio 8.33×). An isotropic Kalman in raw pixel coords
underweights Z by 8.3× — cross-population cells 6 µm apart in Z look ~1.5 px apart. **Track in µm
coords**, not pixels. This was the single biggest cause of cross-population switches.

## Scoring (`abm.py::score_tracking`)

- **`continuity`** — fraction of active tracks surviving each consecutive frame pair (higher =
  less fragmentation). ⚠️ Global-ILP trackers (btrack) inflate this with interpolated ghost
  segments; coastal counts only detected cells, so compare like-for-like.
- **`switch_rate`** — fraction of transitions where the tracked cell changes confetti identity
  (lower = fewer wrong assignments).

`score_tracking(verbose=True)` also prints a `switch_breakdown` (within-run vs at-gap).

**Target: beat both `continuity` AND `switch_rate` simultaneously.** No method has yet.

## Current honest best

| Method | continuity | switch_rate |
|---|---|---|
| No-emb Kalman (loose gate) | 0.836 | 0.247 |
| Kalman chi2=4.6 | 0.725 | 0.203 |
| + flow-warp (w=0.3) | 0.729 | 0.201 |
| coastal seg, `w_color=0` | 0.847 | 0.244 |
| **coastal seg, `w_color=1.0`** | **0.848** | **0.214** |
| coastal seg, `w_color=2.0` | 0.825 | 0.167 |

Full table + btrack comparisons + phase log: `../TRACKING_SESSION_SUMMARY.md`.

## What works vs what doesn't

**Helps (geometric, no learning):**
- Dense flow-warp cost (`w_flow≈0.3`) — sample the flow field at the track's last position,
  penalise assignments that deviate from where it lands.
- Confetti colour cost (`w_color`) — L2-normalise per-cell 3-channel RGB, add `(1−cosine)/2` to
  the LAP, EMA-update the track colour. `w_color=1.0` improves switch_rate at ~zero continuity
  cost; higher trades continuity for switch_rate.

**Failed (learned / appearance-based)** — see also `MORPHOLOGY.md`:
- HMM morphology, PatchEncoder, FlowEncoder — cells are identical greyscale blobs, no signal.
- ContextAssigner (4 variants) — consistently trades switch_rate for continuity; softer gates,
  not better assignments.
- Intrinsic motion prior (Phase 8) — redundant with the Mahalanobis term already in Kalman.

## Key diagnostic

67% of identity switches happen **within consecutive frames** (wrong per-frame cost-matrix
assignment); 33% at gaps. So `stitch_tracklets` / global gap-closing can only address 33% — the
per-frame cost matrix is the bottleneck.

## Data properties (5 movies)
~800 cells/frame · 3 confetti channels (one dominant per cell, stable over time, but ~270 cells
share each channel) · 61 frames · pix_res ≈ 0.48 µm XY, 4.0 µm Z · morphologically homogeneous
round blobs (appearance non-discriminative in greyscale) · crossings in XY, small Z-velocity.

## Untried directions
See `docs/TODO.md` and `docs/FUTURE.md`: learned confetti-colour embedding (metric space over
RGB), global LAP / network flow over the full sequence, hard-negative mining, attention over
track history (not just frame pairs), more confetti channels (hardware change).
