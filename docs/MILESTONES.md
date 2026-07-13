# Milestones

**Append-only** ledger of what landed. Never edit or delete a past entry — add a new one. The
durable counterpart to the throwaway `docs/ROADMAP.md`.

Entry schema: `## <date> — <title>` + what landed / notes.

## 2026-07-08 — Doc skeleton + first tests
Adopted cecelia's Claude documentation skeleton:
- `CLAUDE.md` rewritten as a pure index + "changed area → update" routing table + cross-cutting
  rules; depth moved into `docs/`.
- Added area docs: `ARCHITECTURE`, `SEGMENTATION`, `TRACKING`, `MORPHOLOGY`, `OPTIMIZATION`,
  `DATA`, `JULIA_PORT`.
- Added `FAQ.md` (reader-facing "why"), lifecycle trackers (`TODO`, `ROADMAP`, `MILESTONES`,
  `FUTURE`), and the parked-plan convention (`docs/todo/README.md`).
- Added `tests/` (previously **zero** tests) with `tests/test_utils.py` covering
  `filter_small_cells` and `match_masks_3d`, plus `tests/README.md`.
- Declared previously-undeclared runtime deps (`cma`, `pandas`, `pillow`) in `pyproject.toml`.
- Produced the Julia-portability assessment (`docs/JULIA_PORT.md`): technically feasible but the
  verdict is **don't port now** (applying cecelia's own bar — port when a need appears, not to
  chase Julia-native); two blockers (Torch stack, Farneback flow); revisit only on a real
  consumer/dependency trigger.
- Repo cleanup: folded `QUICK_REFERENCE.txt` into `docs/SEGMENTATION.md` (Appendix) and deleted
  the root file; archived superseded prototype notebooks to `notebooks/archive/`, leaving the
  three live notebooks at the top level.
- Replaced the stale `README.md` (it documented a defunct "ablation study" with files that no
  longer exist) with a proper GitHub README: intro, pipeline diagram, install, quickstart
  (segmentation + tracking, verified against the real public API), docs table, layout, status.

## 2026-07-08 — Notebooks cut over to the installable `cecelia` package
- Dropped the `CECELIA_APP` / `sys.path` bootstrap in the notebooks; switched to
  `import cecelia.utils.*` against cecelia's new pip-installable package (built out on the
  cecelia side per `cecelia-pineapple/docs/todo/PY_PACKAGING_PLAN.md`).
- Repointed `BTRACK_CONFIG` at the vendored config via `cecelia.__file__` instead of an absolute
  path; added a `notebooks` extra (`btrack`); documented the `pip install -e <cecelia>/python`
  dev-link step in `docs/DATA.md`.
- Relicensed to GPL-3.0-or-later (matches cecelia).

## 2026-07-10 — Public repo + standards
Set up `github.com/schienstockd/coastal` and the contribution standards:
- Initialised the git repo and pushed to GitHub (initial commit was the agreed last direct-to-`main`
  push; everything since lands via feature branch + PR).
- Added CI (`.github/workflows/ci.yml`): Ubuntu, CPU-only torch + OpenCV system libs, `pip install
  -e .[dev]` → `pytest`. First run on merged `main` is **green** — the package's first real
  end-to-end execution (Claude's env has no torch/GPU/data).
- Documented the dev workflow in `docs/DEV.md`: never commit/push to `main`, feature-branch + PR,
  conventional commits with the `Co-Authored-By` trailer, and the agent rules (ask before every
  commit/push, state reservations first, `gh` absent → relay the PR URL + paste-ready body).
  Added the `docs/DEV.md` pointer + routing row + git cross-cutting rule to `CLAUDE.md`.
- Reframed `README.md` for a public research repo: 🚧 WIP banner (principles still being validated,
  not a working tool; here for transparency/organisation) and a "How this was built" section
  attributing the science to Dominik and the doc/test/tooling/repo engineering to Claude Code —
  explicitly noting Claude could not run or validate the package.
- Applied cecelia's TODO policy: `docs/TODO.md` tracks **open work only** (items deleted when done);
  the shipped `## Fixed` history moved to this ledger + git.

## 2026-07-13 — Self-contained pixi dev environment
- Added `pixi.toml` + `pixi.lock`: a reproducible env (Python 3.12 + coastal editable with
  dev/notebooks extras + cecelia linked editable + JupyterLab), independent of the old miniconda
  `r-cecelia-env`. Tasks: `pixi run kernel` / `test` / `lab` / `doctor`.
- `pyproject.toml` stays the single dep source; `pixi.toml` adds only Python + the editable cecelia
  link + Jupyter. `scripts/link_cecelia.sh` kept as the non-pixi fallback.
- **First real end-to-end execution of the package** happened here — `pixi run doctor` →
  torch 2.13+cu130 `cuda True`, `cv2`, editable `cecelia` all import; `pixi run test` green. This
  corrected the earlier assumption that the package could not be run in Claude's environment.

## 2026-07-13 — Audit cleanup (dead ends removed, numerical bugs fixed)
- **Numerical bugs fixed** (with golden tests): `utils.intersection_over_union` now computes true
  Jaccard (was L1 containment; matches cellpose stitch3D); `flow.py` deformation metrics
  (divergence/vorticity/strain) corrected swapped gradient axes via a testable `_flow_deformation`;
  `direction_stability` made a proper cosine. These change feature values / stitching → retrain +
  re-tune `stitch_threshold`.
- **Two crashes fixed**: `TwoPassSegmentationInference` (`prob_merge_weight_*` → `prob_weight_*`);
  `optimize_tracking_cma` (`track_physics` → `track_sequence`, bounds reconciled, x0-dim bug).
- **Dead ends removed → `docs/DEAD_ENDS.md`** (append-only ledger with git ref to revive): the ABM
  tracker, HMM boundary-state morphology (polygon/shape readout kept), and the tracking cost terms
  `w_app`/`w_collective`/`w_persistence`/`w_vpred`/`w_exclusion`/`w_breadcrumb` + their helpers.
  `track_sequence` now keeps Mahalanobis + `w_flow` + `w_color` only (~−1100 LOC net).
- Added `notebooks/pipeline_consensus.ipynb` — the clean current end-to-end workflow.
- Cruft + doc drift swept: unused imports, stale docstrings, χ² gate label + Mahalanobis (DeepSORT)
  citation, regenerated `SEGMENTATION.md` metric list, `TRACKING`/`ARCHITECTURE`/`OPTIMIZATION`/
  `MORPHOLOGY` drift.
