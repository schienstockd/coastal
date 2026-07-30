# TODO

Open work for coastal. Tracks **open work only** — delete an item when it's done; don't keep a
hand-curated "fixed" changelog here (it drifts and duplicates history). The durable record of
what landed lives in `docs/MILESTONES.md` (and git history once this is a repo).

## Open

### Cecelia integration
- [ ] **When cecelia is published to PyPI**, replace the dev-time editable-link bridge
      (`scripts/link_cecelia.sh` / local path install) with a normal pinned dependency in
      `pyproject.toml` (`cecelia>=<x.y>`), and delete `scripts/link_cecelia.sh`. See
      `docs/DATA.md` → *Installing / keeping cecelia in sync*. Gated on the cecelia-side publish
      (`cecelia-feijoa/docs/todo/PY_PACKAGING_PLAN.md`, Decision 1 dist-name check).

### Repo structure / docs
- [ ] Backfill `TRACKING_SESSION_SUMMARY.md` numbers into `docs/TRACKING.md`'s table when they
      drift (the summary is the ledger; the table is a snapshot).

### Testing (biggest gap — repo had zero tests)
- [ ] Add a test for `abm.py::score_tracking` on a tiny hand-built track set (continuity +
      switch_rate golden values).
- [ ] Add a shape/range test for `flow.py::prepare_data_for_unet` output.
- [ ] Add a `data.py::validate_training_data` round-trip test.
- [ ] Wire `pytest` into CI (none exists yet).
- [ ] **Characterize `match_masks_3d` no-overlap relabeling.** At `stitch_threshold=0.0`, two
      zero-overlap objects sharing an input label stay the *same* label rather than being split.
      Confirm whether this is intended; pin it with a test either way (see note in
      `tests/test_utils.py`).

### Tracking (research — see docs/TRACKING.md + TRACKING_SESSION_SUMMARY.md)
- [ ] Learned confetti-colour embedding (metric space over RGB) — the top untried idea.
- [ ] Global LAP / network flow over the full sequence (addresses the 33% at-gap switches).
- [ ] Hard-negative mining for any learned cost term.
- [ ] Attention over track history (sequence, not frame pairs).

### Segmentation
- [ ] Reduce Y-cell splitting without over-merging (current mitigation: merge threshold > 0.90).
- [ ] **`compute_cumulative_displacement` re-runs Farneback that `compute_multi_scale_optical_flow`
      already computed.** For each center frame it calls `calc_flow_farneback_between_frames` over
      its whole window, so at `cumulative_window=2` it does ~2T consecutive-frame flows while
      scale 1 of the multi-scale pass has already computed all T−1 of them. Roughly a third of the
      flow time on the 4D path is redundant. The fix is to sum the existing scale-1 flows instead
      — but only valid when scale 1 is present and the window is consecutive-frame, so it needs a
      guard rather than an unconditional swap (and must not become a second flow path). Verify
      values are unchanged before/after with a golden test.
