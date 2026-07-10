# Julia Port Assessment

> Living assessment of whether coastal can be ported to Julia. Modelled on cecelia's
> `docs/todo/julia-port-watchlist.md`. Recheck the watch-list items periodically — when a Julia
> drop-in matures, the linked component becomes portable.

## Verdict (2026-07-08): No — not now. Port when a need appears.

Applying cecelia's own bar (its 2026-07 verdict: *don't port; port when a need appears — a new
module to build or a dependency you actually want gone — not to chase "Julia-native" for its own
sake*):

- **Feasibility is not the question — need is.** A port *is* technically feasible end-to-end
  (only two hard blockers; §B). But it would **remove no dependency** (the Torch stack *is* the
  package; you'd swap torch for `Lux`, not drop it) and would **replace code that already
  works**. That's pure consolidation, and consolidation alone doesn't clear the bar.
- **There is no consumer pull today.** cecelia dropped coastal and its own verdict is "don't
  port." Nothing is asking for a native-Julia coastal right now.
- **The two blockers are the expensive kind.** The Torch DL stack (like cecelia's cellpose) and
  Farneback flow (no native Julia) are exactly the irreducible pieces cecelia decided to leave in
  Python.

**So the right move is to leave it and keep this watch list** for when the calculus changes.
Concretely, revisit a port only when **one** of these becomes true:
1. cecelia (or another consumer) actually needs coastal as a native in-framework module — then
   port the *tracking core* first (§D step 2), which needs neither blocker.
2. A native-Julia Farneback / DL-segmentation stack with parity matures (W1/W2 below).
3. You specifically want torch gone from this package (no reason to today).

The rest of this doc is the *feasibility map* to execute against **if** that trigger fires — not
a recommendation to start.

## Governing rules (read before porting anything)

1. **Port for consolidation, not to shrink dependencies.** Unlike cecelia (where the heavy
   Python — torch/cellpose/scanpy — stays regardless), coastal's Torch stack *is* the core, so a
   Julia port would replace it wholesale rather than sit beside it. The payoff is a
   single-language package that can live **inside cecelia** (which is Julia and has dropped
   coastal). Weigh every port step against that.
2. **The strategic reason to port is reunification.** cecelia is Julia and "dropped coastal"
   (`cecelia .../docs/SHIPPING.md`). A Julia coastal could be a cecelia segmentation/tracking
   module natively instead of a dropped Python sidecar. Mirror cecelia's Julia conventions from
   day one if this is pursued.
3. **Only port what is live.** Do not port `morphology.py` (failed tracking direction, see
   `MORPHOLOGY.md`) or the breadcrumb-ABM variant unless a real need appears — porting dead code
   is net complexity.
4. **Don't create a second variant.** If a port would duplicate a helper, keep that piece Python
   behind the seam until the whole subsystem moves (CLAUDE.md one-canonical-way rule).

---

## A. Per-module port status

| Module | Status | Julia target / note |
|---|---|---|
| `utils.py` (IOU, stitch, filter) | 🟢 **Clean** | `SparseArrays`, hand-rolled normalise. First thing to port. |
| `data.py` | 🟢 **Clean** | numpy → Julia arrays. Trivial. |
| `track.py` (dataclasses, centroids, colours) | 🟢 **Clean** | `ImageMorphology.label_components` + `regionprops`-style; structs. (`ContextAssigner` is Torch — see below.) |
| `abm.py` — LAP/Kalman core | 🟢 **Clean** | `Hungarian.jl` (3 `linear_sum_assignment` sites), hand-rolled Kalman, `Base.Threads`. ~1,900 LOC but mostly array logic. |
| `optimize.py` | 🟢 **Clean** | `CMAEvolutionStrategy.jl` (native, ask/tell + bounds). |
| `segment.py` — post-processing | 🟡 **Mostly clean** | `ImageMorphology.jl` (dilate/erode/label/EDT), `ImageFiltering.jl` (gaussian/maximum filter). The **embedding inference** step depends on the Torch model (see blocker 1). |
| `morphology.py` | 🟡 **Portable, low priority** | `Contour.jl`/marching squares, `LibGEOS.jl`/`GeometryOps.jl`, `HiddenMarkovModels.jl`. Failed direction — don't port speculatively (rule 3). |
| `viz.py` | 🟡 **Portable** | `Makie.jl`, `DataFrames.jl`, `VideoIO.jl` (MP4 export replaces `cv2.VideoWriter`). |
| `model.py` + `loss.py` + `train.py` | 🔴 **Blocker 1 — the Torch stack** | See W1. |
| `flow.py` — Farneback | 🔴 **Blocker 2 — dense optical flow** | See W2. |

Legend: 🟢 clean port · 🟡 portable, some friction · 🔴 hard blocker.

---

## B. Watch list — the two hard blockers

### W1 — The Torch DL stack (UNet + custom losses + AMP training)
- **Covers:** `model.py`, `loss.py`, `train.py`, and the embedding-inference step of
  `segment.py` (~1,000 LOC).
- **Julia target:** `Lux.jl` (preferred for explicit params) or `Flux.jl`. The UNet, the two
  custom losses, and `cdist` embedding distance are all straightforward to reproduce.
- **Friction, not a wall:**
  - **Mixed precision (AMP)** — `GradScaler`/`autocast` equivalents are less mature in Julia;
    may need to train fp32 or hand-roll loss scaling. Verify convergence parity.
  - **Training reproducibility** — must re-validate that a Julia-trained UNet reaches the same
    segmentation quality (golden validation on a fixed movie).
  - **Device abstraction** — `CUDA.jl`/`Metal.jl`; keep the cuda→mps→cpu selection (CLAUDE.md
    GPU rule) in Julia form.
- **Watch for:** nothing external — this is buildable today; the risk is effort + convergence
  re-validation, not ecosystem gaps.

### W2 — Dense Farneback optical flow (`cv2.calcOpticalFlowFarneback`)
- **Covers:** `flow.py` (the algorithmic heart of segmentation) and the flow-warp cost in
  `abm.py`.
- **Gap (verified 2026-07):** **no mature native-Julia Farneback / dense optical flow.**
  `ImageTracking.jl` has sparse Lucas-Kanade and some flow, but not a parity Farneback.
- **Options:**
  1. Wrap OpenCV via `OpenCV.jl` (binding exists; maintenance varies) or `PythonCall.jl` — keeps
     a C/Python dep, but Farneback is CPU-only anyway (CLAUDE.md rule), so it's a contained seam.
  2. Reimplement Farneback in Julia (polynomial expansion + iterative displacement) — nontrivial;
     needs golden validation against the OpenCV output on real frames.
- **Also here:** MP4 export (`cv2.VideoWriter`) → `VideoIO.jl` (clean).
- **Watch for:** a registered, maintained native-Julia dense-flow package with Farneback parity.
  Until then, option 1 (wrap) is the pragmatic path — it does not block a port, it just leaves
  one C-backed seam.

---

## C. Clean drop-ins (no blocker — reference)

| Python | Julia | Note |
|---|---|---|
| `scipy.optimize.linear_sum_assignment` | `Hungarian.jl` | 3 sites in `abm.py` |
| `cma` | `CMAEvolutionStrategy.jl` | native, ask/tell + bounds |
| `hmmlearn.GaussianHMM` | `HiddenMarkovModels.jl` | morphology only |
| `shapely` | `LibGEOS.jl` / `GeometryOps.jl` | morphology only |
| `scipy.ndimage` (dilate/erode/gaussian/EDT/label/maximum) | `ImageMorphology.jl` + `ImageFiltering.jl` | segment.py, morphology.py |
| `skimage.regionprops` / `find_contours` | `ImageMorphology.label_components` + custom / `Contour.jl` | some hand-rolling |
| `scipy.sparse` / `sklearn.normalize` | `SparseArrays` / hand-rolled | utils.py |
| `pandas` / `matplotlib` / `tqdm` / `joblib` | `DataFrames.jl` / `Makie.jl` / `ProgressMeter.jl` / `Threads` | viz.py, flow.py |

---

## D. Suggested phasing (if pursued)

1. **`utils.py` + `data.py`** — clean, self-contained, immediate golden-test target.
2. **`abm.py` tracking core** (Hungarian + Kalman + scoring) — high value, no DL, no flow (pass
   in precomputed flow fields). This alone could run tracking in Julia against Python-produced
   labels.
3. **W2 decision** — wrap vs reimplement Farneback; port `flow.py`.
4. **W1** — Lux UNet + losses + training, with convergence re-validation.
5. **`segment.py`** — once W1 lands, port inference + post-processing.
6. **`optimize.py`, `viz.py`** — last, downstream of everything.

---

## E. Housekeeping surfaced by the audit

- **Undeclared runtime deps** in the *current Python* package (fix regardless of any port):
  `cma`, `pandas`, `Pillow` are imported but were missing from `pyproject.toml`. (Added — see
  `docs/MILESTONES.md`.)

---

*Last assessed: 2026-07-08 (coastal @ v0.1.0; Julia ecosystem checked against cecelia's
2026-07 sweep). Re-audit W1/W2 on each ecosystem check.*
