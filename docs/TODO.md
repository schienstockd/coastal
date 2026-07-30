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

### Optimization
- [ ] **Stop tuning the 5 inference parameters; they are worth ≤7%.** Measured with the fixed
      objective: `n_good` on a held-out movie is 167–175 across every `junk_weight` setting, and the
      shipped params are already within 7% of the best found. See `docs/OPTIMIZATION.md` → *these 5
      parameters are not the lever*. Redirect to training / oversegmentation.
- [ ] Decide a `junk_weight` if the tuner is used again — it encodes how many fragments a real cell is
      worth, which is a scientific choice, not a default. It also flips which way
      `affinity_threshold` is pushed.
- [ ] Check whether `score_tracking_scalar` has the same "ratio over a subset" flaw — i.e. whether it
      can be improved by producing *fewer* tracks.
- [ ] Several parameters still pin at their bounds under the fixed objective (`prob_weight` at 0.0,
      `affinity_threshold` at 0.6). Low priority given the ≤7% headroom, but it means the bounds, not
      the data, are choosing those values.

### Segmentation quality (the actual bottleneck)
- [ ] **~86% of detections are fragments** (1951 of 2277 below `min_cell_size=100` on a real TEST
      movie), and inference parameters cannot fix it — they only shuffle the fragment population
      (1640–2280 across a full tuning sweep) while `n_good` stays flat. This is a training / model
      problem: the prob map and embeddings decide what is findable. Next lever, not the tuner.

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
