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
