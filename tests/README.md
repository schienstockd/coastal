# Tests

`pytest`-based. The repo had **zero** tests before 2026-07-08; this suite is the seed.

## Running

```bash
pip install -e .      # tests import `coastal`, which pulls torch/cv2 via the package __init__
pytest
```

## Conventions (mirrors cecelia's discipline)

- **Any change to core functionality ships with a test in the same change.** Core = label
  stitching / IOU (`utils.py`), flow-metric shapes (`flow.py`), scoring (`abm.py::score_tracking`),
  the data-prep contract (`data.py`).
- **Assert real invariants on tiny hand-built arrays**, not on large fixtures. `test_utils.py`
  builds 4×4 masks and checks label-continuity / disjointness — no data files needed.
- **Ported / published algorithms get golden-value validation.** When a Julia port lands (see
  `docs/JULIA_PORT.md`) or a paper algorithm is implemented, pin expected numbers in a test.
- **Keep any fixtures small** (hundreds of KB) and out of the user's data dir; document each one
  here.

## Current tests

- `test_utils.py` — `filter_small_cells`, `intersection_over_union`, `match_masks_3d`
  (label unification + disjointness across Z).

## Wanted (see `docs/TODO.md`)

- `score_tracking` continuity/switch_rate golden values on a hand-built track set.
- `prepare_data_for_unet` output shape/range.
- `validate_training_data` round-trip.
- CI wiring (none yet).
