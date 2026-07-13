# Coastal — Project Guide

**Coastal** is a Python package for **instance segmentation of cells in 2D/3D microscopy
images** using learned embeddings trained on optical-flow metrics, extended with a **cell
tracking module** for 3D+T confetti fluorescent microscopy.

Stack: **Python** · PyTorch (UNet) · OpenCV (Farneback flow) · scipy/skimage (post-processing).
No ground-truth labels are required for segmentation — training is guided by optical-flow
structure. Tracking supervision comes from confetti fluorescence (stable colour per cell).

This file is an **index + cross-cutting rules only**. Depth lives in `docs/<AREA>.md` — do not
duplicate it here; add a pointer instead.

See also:
- [`FAQ.md`](FAQ.md) — reader-facing "why" highlight reel: the *counterintuitive* decisions
  (no labels, flow-as-supervision, why every learned tracker lost to Kalman). Punch lines, not
  prose. New detail goes in the relevant `docs/` file; only promote a genuinely surprising
  one-liner up to the FAQ.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — end-to-end pipeline, module boundaries, the
  array-shape data contracts between stages, and the cecelia integration seam.
- [`docs/SEGMENTATION.md`](docs/SEGMENTATION.md) — the segmentation subsystem: flow metrics →
  UNet (prob + embeddings) → 2-loss training → two-pass inference → 3D IOU stitching. Known
  issues (Y-cell splitting, oversegmentation) and default best parameters.
- [`docs/TRACKING.md`](docs/TRACKING.md) — the tracking subsystem: Kalman + LAP assignment, the
  optional cost terms, scoring (`continuity`/`switch_rate`), and the current design state.
  Points to `TRACKING_SESSION_SUMMARY.md` for the running experiment log.
- [`docs/MORPHOLOGY.md`](docs/MORPHOLOGY.md) — polygon extraction + HMM boundary states
  (`morphology.py`). A tried-and-failed tracking direction; documents what is reusable.
- [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) — CMA-ES hyperparameter tuning for both
  segmentation and tracking (`optimize.py`): param names, bounds, objective functions.
- [`docs/DATA.md`](docs/DATA.md) — data format, the `prepare_data_for_unet` contract, array
  conventions, and how coastal takes input from / returns output to cecelia.
- [`docs/JULIA_PORT.md`](docs/JULIA_PORT.md) — Julia portability assessment: per-module status,
  the two hard blockers (Torch stack, Farneback flow), the watch list of Julia drop-ins, and
  the "port into cecelia" strategic angle. Modelled on cecelia's `julia-port-watchlist.md`.
- [`docs/TODO.md`](docs/TODO.md) — open work only; items are deleted when done (no hand-curated
  `## Fixed` changelog — shipped history lives in `docs/MILESTONES.md`).
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — throwaway forward goals, in phases. Consult before a
  new phase.
- [`docs/MILESTONES.md`](docs/MILESTONES.md) — append-only ledger of what landed.
- [`docs/FUTURE.md`](docs/FUTURE.md) — deliberately deferred ideas: what / why deferred / when
  to revisit.
- [`docs/DEAD_ENDS.md`](docs/DEAD_ENDS.md) — tried-and-failed approaches removed from the live
  code: what / why dropped / git ref to revive / what would rule them in. (Distinct from
  `FUTURE.md`, which is deferred-but-untried.)
- [`docs/todo/README.md`](docs/todo/README.md) — the parked-plan convention (`*_PLAN.md`).
- [`docs/DEV.md`](docs/DEV.md) — development workflow: branches, commits, PRs, CI, and the
  never-push-to-`main` + ask-before-every-commit + reservations-first rules.

**Keep the docs current — update the relevant file in the same change, not after.**

| Changed area | Update |
|---|---|
| Pipeline stages, module boundaries, array-shape contracts, cecelia seam | `docs/ARCHITECTURE.md` |
| Flow metrics, UNet architecture, losses, training, inference, 3D stitching | `docs/SEGMENTATION.md` |
| Kalman/LAP tracking, cost terms, scoring metrics | `docs/TRACKING.md` (+ log the experiment in `TRACKING_SESSION_SUMMARY.md`) |
| Polygon/HMM morphology | `docs/MORPHOLOGY.md` |
| CMA-ES param tuning (names, bounds, objectives) | `docs/OPTIMIZATION.md` |
| Input/output format, `prepare_data_for_unet`, cecelia integration | `docs/DATA.md` |
| Julia-portability status of any component | `docs/JULIA_PORT.md` |
| Deferring a known-better approach | `docs/FUTURE.md` |
| Removing a tried-and-failed approach (dead end) | `docs/DEAD_ENDS.md` |
| A decision becomes *surprising* to an outside reader | `FAQ.md` (one punchy Q&A; keep detail in `docs/`) |
| Branching, commits, PRs, CI (dev workflow) | `docs/DEV.md` |

## Cross-cutting rules

- **One canonical way; the second way is the bug.** Before adding a second variant of a
  cross-cutting thing (a cost term, a metric, a flow computation, a data-prep path), stop and
  say so — propose centralising it (one helper, used everywhere) rather than forking it. If two+
  rounds pass on one question without clear progress, surface it or park it in `docs/TODO.md`.
- **Cite sources for non-trivial algorithms.** Farneback params, Mahalanobis gating, CMA-ES
  bounds, HMM setup — leave a comment pointing at the paper/reference, and add golden-value
  validation to the test suite where a published algorithm is ported.
- **Any change to core functionality ships with a test in the same change.** Core = the label
  stitching / IOU matching (`utils.py`), flow-metric shapes (`flow.py`), the scoring functions
  (`abm.py::score_tracking`), and the data-prep contract (`data.py`). See `tests/README.md`.
- **GPU: PyTorch device abstraction only.** Select `cuda` → `mps` → `cpu` at runtime; never
  hard-code `cuda`. Do **not** introduce a CUDA-enabled OpenCV dependency — Farneback runs on
  CPU by design.
- **No ground-truth labels in the segmentation path.** The whole point is flow-guided training;
  do not add a code path that silently assumes GT masks exist.
- **Git: never commit or push to `main` — branch + PR.** Ask before every commit and push
  (explicitly, each time; a "do the work" yes is not a commit/push yes), and volunteer honest
  reservations first. `gh` isn't installed → push the branch and relay the PR-create URL + a
  paste-ready body. Note this package **cannot be run in this environment** (no torch/GPU/data), so
  most changes are unverified end-to-end — say so. Full workflow: `docs/DEV.md`.

## Running

Recommended — the self-contained **pixi** env (independent of any external conda env; also links
cecelia editable). See `docs/DATA.md` → *Independent dev environment (pixi)*:

```bash
pixi install                # build the env (Python 3.12, coastal + cecelia editable, Jupyter)
pixi run kernel             # register the "Python (coastal)" Jupyter kernel (once)
pixi run test               # pytest
pixi run lab                # JupyterLab rooted at notebooks/
```

Plain-pip fallback (what CI uses; run `scripts/link_cecelia.sh` for the notebooks' cecelia dep):

```bash
pip install -e .[dev]
pytest
jupyter notebook notebooks/tracking.ipynb       # tracking work
jupyter notebook notebooks/optical_flow.ipynb   # segmentation work
```

## Notebooks

`notebooks/pipeline_consensus.ipynb` is the **clean, current end-to-end workflow** (post-audit
API: cecelia load → flow-metric UNet → 3D+T labels → Kalman+LAP `w_flow`+`w_color` tracking →
scoring) — start here. The exploratory notebooks (`tracking.ipynb`, `optical_flow.ipynb`,
`pipeline_confetti_ceiling.ipynb`) carry the full experiment history and reference removed
approaches (see `docs/DEAD_ENDS.md`). Superseded prototypes (`treecell-v*`, `treemove-v*`,
`optical_flow-old`) are in `notebooks/archive/` — history, not instructions.
