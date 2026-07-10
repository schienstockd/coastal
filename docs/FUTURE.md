# Future

Deliberately **deferred** ideas — known options set aside on purpose. Each: **what**, **why
deferred**, **when to revisit**. Distinct from `docs/TODO.md` (active backlog) and
`docs/todo/*_PLAN.md` (designed-but-unbuilt features).

## More confetti channels (hardware change)
- **What:** the fundamental tracking ceiling is that ~270 cells share each of only 3 confetti
  channels. A reporter system with more channels would make colour genuinely discriminative.
- **Why deferred:** requires a wet-lab / imaging change, out of scope for the software.
- **When to revisit:** if new data with more channels arrives — then `w_color` and a learned
  colour embedding become far stronger.

## Julia port of the whole package
- **What:** rewrite coastal in Julia to live natively inside cecelia.
- **Why deferred:** two real blockers (Torch stack, Farneback flow) and no immediate consumer
  pull; the Python package works today.
- **When to revisit:** when reintegration with cecelia is prioritised, or when a native-Julia
  Farneback / DL-segmentation stack matures. Full analysis + phasing in `docs/JULIA_PORT.md`.

## Deprecate `morphology.py`
- **What:** remove the polygon/HMM morphology module.
- **Why deferred:** it failed as a tracking signal but the polygon/shape-feature extraction is
  generic and might serve a future morphology-readout / QC use.
- **When to revisit:** if no morphology-readout use appears by the next cleanup pass, delete it
  (keep only if something imports it). See `docs/MORPHOLOGY.md`.

## Global / batch tracking (network flow)
- **What:** replace greedy per-frame LAP with a global optimisation over the full sequence.
- **Why deferred:** the diagnostic shows only 33% of switches are at gaps (where global helps);
  67% are within-frame cost-matrix errors that a global formulation doesn't fix on its own.
- **When to revisit:** after the per-frame cost matrix is improved (Phase B), to mop up the
  at-gap residual — pair it with a better colour/motion cost, not as a standalone fix.
