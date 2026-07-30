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
  cecelia side per `cecelia-feijoa/docs/todo/PY_PACKAGING_PLAN.md`).
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

## 2026-07-30 — Streaming the 4D segmentation path (OOM fix, lazy flow metrics, 8.5× faster frames)
- **The bug:** `notebooks/pipeline_consensus.ipynb` OOM-killed the kernel at
  `predict_temporal_volume` on the real `[180, 4, 15, 531, 586]` movies. Four compounding causes —
  two in the notebook, two in the package: `np.asarray(volumes[uid][0])` materialised the whole
  6.7 GB movie even though the function streams per z-slice; the tracking cell materialised the same
  movie a second time; `n_workers=8` ran eight in-flight z-slices, each holding 14 float32 metric
  planes × T (3.1 GB); and `predict_temporal_volume` retained every prob map and regionprops list.
- `Inference3D.predict_temporal_volume` now writes per-slice labels straight into the single
  `[T,Z,H,W]` int32 buffer and Z-stitches it **in place**, dropping each slice's prob maps and
  regionprops as it finishes (they were retained for all T×Z = 2700 frames, ~6.7 GB, and no caller
  ever read them). New `keep_results=False` default returns `None` for the second tuple element;
  pass `keep_results=True` for the old inspection behaviour. Docstring now states the memory
  contract.
- Notebook refactored around the streaming seam: lazy dask passed everywhere (`extract_cell_intensities`
  / `score_tracking` already index per timepoint), `grey` for Farneback built one z-slice at a time,
  `SEG_WORKERS`/`FLOW_WORKERS` split out as named knobs, and segmented labels cached to
  `CACHE_DIR` as `.npy` + reloaded with `mmap_mode='r'` so a kernel restart resumes at tracking
  instead of re-segmenting.
- **Then removed the ceiling itself: flow metrics are now lazy.** `prepare_data_for_unet` returns a
  `TemporalMetrics` sequence instead of a list of T dicts — same values, same indexing/slicing/
  iteration, but each frame's 14 float32 planes are built when asked for. That term was 3.1 GB per
  in-flight z-slice at T=180 while `predict_sequence` only ever reads one frame at a time. Kept to
  **one** path rather than a streaming fork: the single lazy type is what everything gets, and the
  two training entry points (`prepare_data_for_unet_batch_4d` / `_batch`) call `list(...)` because a
  Dataset indexes per sample per epoch. Deleted `compute_all_temporal_metrics` — it had become a
  second way to say `list(metrics)`, unused outside its own test.
- `extract_temporal_metrics` no longer rebuilds and renormalises the **whole stack on every call**
  just to read one frame out of it (O(T) full-stack copies per z-slice). It scales the one frame it
  needs, with the global min/max cached by `TemporalMetrics` via a new `frame_range` arg. Measured
  in isolation: **38.4 s → 0.07 s** per z-slice at T=180, 531×586 — ~10 min of pure waste per
  15-slice movie. `normalize_and_project` likewise selects channels *before* the float32 cast
  instead of converting all C and discarding some.
- **Then the CPU side: per-label loops no longer touch the whole frame.** Profiling `predict_frame`
  on a real 531×586 frame (~600 labels) put ~90% of its 6.6 s in four `for label in ...` loops doing
  whole-frame work: `binary_fill_holes` per label (45%), `distance_transform_edt` per fragment (30%),
  `instances == label` size/reindex passes (9%), and `components == comp_id` per component in the
  seed-guarantee step. All four now work inside per-label bounding boxes (`find_objects`) or, for
  `_remove_small_components`, a single `bincount` + lookup-table gather; the embedding blur became one
  `gaussian_filter(sigma=(s, s, 0))` call instead of 64 per-channel calls. **6.62 → 0.78 s/frame
  (8.5×)** with **bit-identical labels**, verified by replaying the pre-optimisation `segment.py`
  straight out of git against the new one over real frames. One z-slice at T=180: **≳20 min → 142 s**.
- **Found while profiling — the tuned `BEST_PARAMS` were never tuned.** Two independent degeneracies
  in `optimize_segmentation_cma`, both now fixed or guarded:
  - `merge_max_distance < 1.0` disables fragment merging outright (`distance_transform_edt(~mask)` is
    0 on the fragment and ≥1 off it, so a sub-1 threshold selects only the fragment's own pixels,
    which `& ~mask` removes). The search bound was `(0.5, 3.0)` — a third of it a flat dead zone —
    now floored at **1.0** and pinned by a test.
  - `score_segmentation`'s `purity_threshold` was above the achievable range. Measured on a real
    movie (344 large cells): purity median **0.395**, max **0.558**, floored near `1/n_channels`
    because `frame / ch_mean` never subtracts background. At the notebook's `purity_threshold=0.8`
    every candidate scores exactly **0.0**, and `max(history, ...)` returns the *first* tie — so
    `BEST_PARAMS` is the first sample CMA-ES drew, not an optimum. That explains both the disabled
    merging and a `merge_affinity_threshold` (0.2261) below `affinity_threshold` (0.5534), which the
    docstring calls unusual. `optimize_segmentation_cma` now **warns loudly** when the objective
    never varied, and its docstring no longer misstates the formula (it is
    `n_good / (n_good + n_merged)`, with fragments excluded from the denominator and free unless
    `count_penalty_weight > 0`).
  - Set `merge_max_distance = 1.5` in the notebook to un-break merging. Measured effect on 5 real
    frames: 2366 → 2277 cells (**−3.8%**), so this is *not* on its own the Y-cell-splitting fix.
- **Fixed the objective itself, then found it exonerates the parameters.** `score_segmentation` now
  returns a **reward** — `n_good - junk_weight * (n_merged + n_fragmented)` — with purity computed
  after per-channel background subtraction (`background_percentile=25`, `None` = legacy).
  `tests/test_optimize_objective.py` pins the anti-gaming properties behaviourally: dropping a cell,
  shrinking cells below `min_cell_size`, splitting one into fragments, over-merging two, or adding
  junk must *each* lower the score, and "find nothing" cannot win. A synthetic case shows four
  perfectly single-channel cells scoring 0 at **every** threshold with background left in, and 4/4
  with it removed — so `purity_threshold`'s documented 0.7 default is usable again (baseline good
  cells on the TEST movie: 171 at 0.7 subtracted, vs 140 under the old metric at a *lower* 0.4).
- **Then swept `junk_weight` ∈ {0, 0.02, 0.05, 0.1, 0.3}, 200 evals each — and adopted nothing.**
  `n_good` on the held-out TEST movie is **167–175 for every setting** except the extreme 0.3 (144);
  what the parameters actually trade is the fragment count (1640–2280). At `junk_weight=0` the score
  *is* `n_good`/frame, and it spanned **16.6–22.2** across the search with the shipped params already
  at **20.8** — so these five inference parameters are worth **≤7%** and the shipped values are
  already within that. Full table in `docs/OPTIMIZATION.md`. The bottleneck is training and
  oversegmentation (**~86%** of detections are fragments), not the tuner; recorded in `docs/TODO.md`.
- **Superseded the earlier ratio-based re-tune** (kept below for the record, since the gaming
  behaviour is the reason the objective changed).
- **Re-ran the tune with a working objective — and did not adopt the result.** With
  `purity_threshold=0.4` and a calibrated `count_penalty_weight`, CMA-ES gets a real gradient
  (0.02–0.28, no flat warning) and converges in ~2 min. But `affinity_threshold` pins to whatever the
  upper bound is (0.6, then 0.8 when widened), because the score is a ratio over large cells only:
  on a held-out TEST movie the score rose 0.429 → 0.523 while the absolute number of good cells
  **fell 140 → 122 → 116**. It improves by discarding cells into the near-free fragment bin. The
  bound was left at 0.6 rather than widened to chase the wall, and `BEST_PARAMS` keeps its shipped
  values apart from `merge_max_distance`. Two prerequisite objective fixes are measured and parked in
  `docs/TODO.md`: background-subtract purity (median 0.385 → 0.709, 34% → 90% of its range, which
  also makes the documented 0.7 default usable) and stop maximising a ratio over a subset. Both
  redefine "good segmentation", so the intent is Dominik's call.
- **Measured on the box** (not estimated): peak RSS for one in-flight z-slice at T=180 went
  **~7.0 GB → 4.65 GB** (incl. the Torch/CUDA context), which lifts the notebook's `SEG_WORKERS`
  from 2 to 4 on 31 GB — with ~6 as the ceiling. Together with the loop work a 15-slice movie goes
  from hours to **~10 min**. The
  four rewritten notebook expressions were verified **exactly equal** to the numpy ones they replace
  on real cecelia data (streamed `grey`, `extract_cell_intensities`, `score_tracking`, and tracking
  on a read-only mmap).
- Tests (80 total): `tests/test_segment_localised.py` keeps every pre-optimisation implementation
  **verbatim as the oracle** and compares against the rewrites over randomised label maps (holes,
  touching pairs, frame-edge cases, `merge_max_distance` 0.62–3.0), plus bit-identity of the
  argmax tie-break and the single-call embedding blur. `tests/test_segment.py` gains a 4D contract
  test with a stub UNet — shape/dtype,
  Z-consistent labels, `keep_results` parity — and pins the streaming guarantee with a lazy
  stand-in whose `__array__` raises, so re-introducing `np.asarray(volume)` fails the suite.
  `tests/test_flow_lazy_metrics.py` pins that the per-frame normalisation is **bit-identical** to the
  old whole-stack copy, that the cached `frame_range` changes nothing, that the sequence really is
  lazy and uncached, and that channel-selection order in `normalize_and_project` is neutral.
- Follow-up parked in `docs/TODO.md`: `compute_cumulative_displacement` re-runs consecutive-frame
  Farneback that the multi-scale pass already computed (~a third of the flow time).
