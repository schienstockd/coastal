# Tracking Session Summary

## Goal

Track T cells in 3D+T confetti fluorescent microscopy. Input: `instances_4d [T, Z, H, W]`
from the coastal segmentation pipeline. Output: `{track_id: {t: pos_um}}`.

**Key data properties:**
- ~800 cells per frame, 3 confetti channels (RGB, one dominant per cell — stable over time)
- Cells are morphologically homogeneous (round blobs); appearance is not discriminative in greyscale
- Crossings happen in XY; Z-velocity is small
- 5 movies, 61 frames, pix_res ~0.48 µm XY, 4.0 µm Z

**Metrics:**
- `continuity`: fraction of active tracks that survive each consecutive frame pair (higher = less fragmentation)
- `switch_rate`: fraction of frame transitions where the tracked cell changes confetti identity (lower = fewer wrong assignments)

**Baselines:**
| Method | continuity | switch_rate |
|---|---|---|
| No-emb Kalman (loose gate) | 0.836 | 0.247 |
| Kalman chi2=4.6 (best params) | 0.725 | **0.203** |

Target: beat both simultaneously. No method has done this yet.

---

## What Was Tried

### Morphology / HMM
- Extracted polygon boundary features, fit Gaussian HMM to boundary states
- **Failed**: T cells are featureless round blobs — all HMM states are identical

### Patch embeddings (PatchEncoder)
- Small greyscale crops around each cell centroid, CNN-encoded
- **Failed**: all cells look the same in greyscale — patches carry no discriminative signal

### FlowEncoder
- Predict per-cell displacement from Farneback flow averaged over cell mask
- **Failed**: the flow prediction wasn't better than Kalman velocity

### ContextAssigner — joint transformer for assignment
Architecture: all M tracks × N detections simultaneously; track self-attention →
cross-attention → [M, N] logits.

Supervision: confetti cosine similarity between track and detection (same cell ≈ 1).

Iterations:
1. **Argmax color labels** (4-class): loss plateau at 0.967 — ~N/4 false positives per frame
2. **Exclusive spatial nearest-match**: loss plateau at 0.49 — same-channel cells still noisy
3. **Cosine similarity supervision** (current): continuity=0.743, switch_rate=0.220 — marginal improvement but worse switch_rate than baseline
4. **Confetti as model input features** (n_ch=3): continuity=0.735, switch_rate=0.240 — worse; had a bug (track ID ≠ cell label ID — fixed)
5. **Flow features as model input** (n_ch=6: u,v,mag,div,vort,strain): continuity=0.735, switch_rate=0.234 — still worse than baseline

**Pattern**: every CTX variant improves continuity slightly but worsens switch_rate. The model makes softer gates, not better assignments.

### Flow-warp cost (geometric, no model)
Sample the dense Farneback flow field at each track's last observed position (t),
compute where it naturally lands at t+1, penalise assignments that deviate from that.

```
flow_pred = last_pos_t + flow_field[t][last_pos_px]
flow_cost = dist(detection, flow_pred) / gate_um
dist_cost = (1 - w_flow) * mahalanobis + w_flow * flow_cost
```

**Result at w_flow=0.3**: continuity=0.729, switch_rate=0.201 — **marginal improvement on both**, best combined result so far.

### stitch_tracklets
Post-hoc gap closing by spatial distance. Minimal effect (~0.001 improvement).

---

## Diagnostic Findings

From `score_tracking(verbose=True)` on the best method:

- **67% of switches happen within consecutive frames** → wrong per-frame assignment (cost matrix problem)
- **33% at gaps** → stitching/re-init failure
- Conclusion: global LAP gap-closing would only help 33% of switches; the per-frame cost matrix is the bottleneck

---

## Current Code State

### `coastal/abm.py`
- `compute_cell_flow_features(frames, instances_4d, n_workers)`:
  - Single Farneback pass per Z-slice; returns per-cell 6-dim features `{t: {cid: [u,v,mag,div,vort,strain]}}` and dense flow field `{t: [H,W,2]}`
  - Replaces the old `compute_cell_flows` (now derived from first 2 dims of the 6-dim feature)
- `track_sequence`: has `dense_flow_fields`, `w_flow`, `w_ctx`, `cell_intensities`, `cell_flow_features` params
  - Flow-warp cost samples at `last_pos_t` (fixed — was wrongly sampling at Kalman pred t+1)
- `score_tracking`: now prints `switch_breakdown` (within-run vs at-gap) and has fragmentation stats (fragmentation is broken — groups by channel argmax, not unique cell ID)

### `coastal/track.py`
- `ContextAssigner(d_model, n_heads, n_ch)`: joint transformer, `n_ch` extra features appended to track/det vectors
- `_ctx_frame_features`: builds training tensors; supervision = highest-cosine-sim detection within `landing_radius_um` as exclusive positive per track; model input = confetti intensity vectors (L2-normalised)
- `train_context_assigner`: uses confetti intensities for both supervision and model input; `flow_features_per_movie` parameter removed

### `notebooks/tracking.ipynb`
- Section 7: single `compute_cell_flow_features` call; `cell_flows` derived from u,v dims
- Section 9: trains CTX with confetti intensities (n_ch=3)
- Section 10: runs with `cell_intensities`, `dense_flow_fields`, `w_flow=0.3`
- Section 11: comparison table

---

## Next Direction — Open

No clear winner. Needs a fresh perspective.

**What the data tells us:**
- 67% of switches are wrong per-frame assignments — the cost matrix is the bottleneck, not stitching
- The only signal that marginally helped is the dense optical flow field (geometric, no learning)
- Every learned model (CTX variants) consistently trades switch_rate for continuity without net gain
- ~270 cells share each confetti colour → supervision is inherently noisy with only 3 channels

**What has NOT been tried:**
- Using confetti directly in the LAP cost (no model — just colour distance between track and detection)
- Re-ID style: embedding each cell's confetti vector and using embedding distance as assignment cost
- Graph-based tracking (cells as nodes, optical flow edges)
- Increasing the number of confetti channels (new reporter system — hardware change)
- Any form of negative mining / hard negative training for the CTX model
- Attention over time (not just frame pairs) — track history as a sequence

### btrack (global hypothesis optimization — GLPK ILP)

btrack v0.7.0 replaces greedy per-frame LAP with a global ILP over all frames.
Key findings:

- **Per-population btrack** (3 independent runs, one per confetti color): continuity=0.882,
  switch_rate≈0 by construction (no cross-pop candidates). Used as GT source for 18091
  consecutive same-cell pairs across 5 movies.

- **Full-population btrack, pixel coords**: continuity=0.887, switch_rate=0.290 — global
  ILP fixed 88% of gap-related switches (vs 60% for Kalman). Remaining errors concentrated
  in within-frame assignment (encounter zones).

- **Root cause of cross-pop switches**: btrack given raw pixel coords where Z=4µm/px and
  XY=0.48µm/px (ratio 8.33×). Isotropic Kalman underweights Z by 8.3× — cross-pop cells
  6µm away in Z look only 1.5px apart.

- **Full-population btrack, µm coords, r=10µm**: continuity=0.899, switch_rate=0.246
  — **best inference-time result**. 31.4% reduction in cross-pop assignments.

- **Failure analysis** (803 confirmed cross-pop assignments):
  - dz AUC=0.679 (Z distance discriminates 67.7% of failures)
  - dxy AUC=0.494 (XY distance is flat — cells are in the same XY region by definition)
  - Multi-feature logistic regression AUC=0.707

- **Switch breakdown for btrack**: 88% within-run (within consecutive frames), 12% at gaps
  — global ILP solved the gap problem, concentrating errors in per-frame cost matrix.

---

## Quick Reference: Key Numbers

| Method | continuity | switch_rate | notes |
|---|---|---|---|
| No-emb Kalman | 0.836 | 0.247 | high continuity target |
| Kalman chi2=4.6 | 0.725 | 0.203 | best switch_rate target |
| + stitch | 0.726 | 0.205 | negligible |
| + flow (w=0.3) | 0.729 | 0.201 | best combined (Kalman) |
| + CTX (confetti vec) | 0.743 | 0.220 | trades switch for continuity |
| + CTX+flow | 0.735 | 0.234 | worse |
| **btrack per-pop** | **0.882** | **~0** | zero cross-pop by construction (confetti split) |
| **btrack full-pop µm r=10** | **0.899** | **0.246** | **best no-confetti result** |
| btrack confetti VISUAL | 0.918 | 0.184 | ceiling; inflated by ILP ghost tracks |
| coastal_seq coastal w=0 | 0.847 | 0.244 | honest baseline (no ghost tracks) |
| **coastal_seq coastal w=1** | **0.848** | **0.214** | **best honest result — Phase 9** |
| coastal_seq coastal w=2 | 0.825 | 0.167 | beats btrack VISUAL sr, lower cont |
| coastal_seq confetti w=1 | 0.822 | 0.136 | best sr with confetti seg |

**Target: beat continuity=0.848 AND switch_rate=0.214 simultaneously (current honest best).**
**Note**: btrack's higher continuity (~0.90+) includes ILP-interpolated ghost tracks that inflate the metric.

---

## Phase 8 — Intrinsic Motion Prior (complete — negative result)

**Result**: velocity-prediction AUC=0.621, group velocity AUC=0.656. Both are in the
"weak but additive" zone. `w_vpred` sweep in `track_sequence` produced no improvement
over baseline — redundant with Mahalanobis distance already in Kalman.

---

## Phase 9 — Confetti Color in LAP Cost (`pipeline_confetti_ceiling.ipynb`)

**Setup**: train confetti UNet (dropout=0) → confetti instances (~400–500 cells/frame).
Extract per-cell L2-normalised RGB intensity vector from each instance. Add cosine-distance
color cost to `track_sequence` LAP via `w_color` parameter.

**Key architectural note — btrack continuity inflation**: btrack's ILP gap-filler
interpolates straight-line segments across gaps. These phantom segments count as surviving
tracks in the continuity metric. coastal_seq only tracks actually-detected cells — its
lower raw continuity is more honest, not worse performance.

### btrack conditions (5-movie mean)
| Method | continuity | switch_rate | notes |
|---|---|---|---|
| coastal seg + btrack motion | 0.908 | 0.254 | best prior inference-time result |
| confetti seg + btrack motion | 0.915 | 0.198 | tighter segmentation, no color used |
| confetti seg + btrack VISUAL | 0.918 | 0.184 | confetti as btrack feature — ceiling |

### coastal `track_sequence` + `w_color` (5-movie mean)
| Instances | w_color | continuity | switch_rate | notes |
|---|---|---|---|---|
| confetti | 0.0 | 0.820 | 0.169 | honest baseline (~400 cells/frame) |
| confetti | 1.0 | 0.822 | 0.136 | color cost helps, real cells only |
| confetti | 2.0 | 0.788 | 0.078 | continuity starts dropping |
| confetti | 3.0 | 0.763 | 0.025 | extreme color gating |
| **coastal** | **0.0** | **0.847** | **0.244** | full coverage (~800 cells/frame) |
| **coastal** | **1.0** | **0.848** | **0.214** | **color cost, honest best** |
| coastal | 2.0 | 0.825 | 0.167 | beats btrack VISUAL sr, lower cont |
| coastal | 3.0 | 0.794 | 0.109 | aggressive gating |

**Key findings:**
- coastal_seq on coastal instances at w=0 matches btrack motion sr (0.244 vs 0.254) without ghost tracks
- Adding w_color=1.0 improves switch_rate to 0.214 at zero continuity cost — genuine improvement
- Confetti segmentation is not genuinely better at separating populations; just tighter (training variance)
- **No method beats both continuity=0.908 AND switch_rate=0.254 simultaneously with real (non-ghost) tracks**

### What `w_color` is doing
L2-normalise each cell's 3-channel confetti RGB. Add `(1 - cosine_similarity) / 2` as a
cost term in the Hungarian LAP. EMA-update the track's color estimate after each assignment.
Dim cells (below Otsu per-frame) zeroed out. Implementation in `coastal/abm.py::track_sequence`.

### What remains untried
- **Learned color embedding**: instead of raw 3-channel cosine distance, train a metric
  space that maps confetti RGB → embedding where same-cell distances are tight and
  cross-cell distances are large. Could generalise across brightness, bleaching, and
  within-channel population ambiguity (~270 cells share each channel).
- Global LAP / network flow over full sequence
- Hard negative mining in any learned cost term
