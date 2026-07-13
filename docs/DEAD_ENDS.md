# Dead ends — ruled-out approaches

Approaches that were **tried and did not work**, then **removed from the live code** to keep it
clean. This is not a wishlist (see [`FUTURE.md`](FUTURE.md) for deferred-but-untried ideas) — it is
a record of directions already explored, so that when a genuinely better solution appears they can
be **revived and formally ruled out** rather than silently re-tried.

The code is not lost: git history holds every removed approach. Each entry names **what it was**,
**why it was dropped**, **where to revive it from**, and **what "ruling it out" would now require**.

> Revival: the removals landed in the `chore/audit-cleanup` commit **`deb8be5`**. To bring one
> back, take the file from its parent **`deb8be5^`** (the pre-removal state) and lift the named
> symbols — e.g. `git checkout deb8be5^ -- coastal/abm.py`, or `git show deb8be5^:coastal/abm.py`.

---

## Tracking — learned & prior-based approaches (all lost to Kalman + LAP)

The settled tracker is Kalman + Hungarian LAP with three cost terms: the always-on Mahalanobis
position gate, `w_flow` (dense flow-warp), and `w_color` (confetti cosine). Everything below was an
additional cost term or an alternative tracker that never beat that baseline on both `continuity`
and `switch_rate`. Removed from `coastal/abm.py` and `coastal/track.py`.

### ABM (agent-based) tracker
- **What:** `BreadcrumbField`, `MotilityState`, `CellAgent`, `ABMTracker`, `track_abm` — a
  per-cell agent model (motility states, breadcrumb spatial memory) as an alternative to the LAP
  assignment.
- **Why dropped:** never beat the Kalman + LAP baseline; no result in `TRACKING.md` /
  `TRACKING_SESSION_SUMMARY.md` favoured it. The extra agent state added complexity without a
  switch-rate/continuity win.
- **Ruling it out would need:** a scenario where per-agent behavioural state carries identity
  signal the Kalman motion model can't — not the case for morphologically identical confetti blobs.

### Appearance / embedding cost (`w_app`)
- **What:** cosine cost on learned per-cell embeddings (`extract_cell_embeddings`, `track_emb_ema`
  EMA state) to disambiguate assignment by appearance.
- **Why dropped:** the cells are near-identical blobs — appearance is not discriminative (see
  `FAQ.md`). Added cost, no gain.
- **Ruling it out would need:** a reporter/appearance channel with genuine per-cell variation.

### Collective-motion cost (`w_collective`) + flow-smoothing helpers
- **What:** penalised assignments whose velocity diverged from the local cell-population flow; the
  `smooth_cell_flows` / `blend_flows` helpers (Gaussian-neighbour flow smoothing / convex blend)
  supported it. Removed with the cost term. (`compute_cell_flow_features` was **kept** — it also
  produces the dense field that feeds the retained `w_flow` flow-warp cost.)
- **Why dropped:** neighbourhood-flow coherence did not improve the assignment over per-track
  Kalman motion.
- **Ruling it out would need:** dense, reliable collective motion that the per-track model misses.

### Persistence / turn-penalty cost (`w_persistence`)
- **What:** penalised sharp direction changes relative to recent velocity (T-cell directional
  persistence prior).
- **Why dropped:** the momentum term in the Kalman prediction already captures directional
  continuity; the explicit turn penalty added no net win.

### Velocity-prediction cost (`w_vpred`) — Phase 8 negative
- **What:** Euclidean distance from a raw-velocity-extrapolated position, as a separate cost.
- **Why dropped:** redundant with the Mahalanobis gate on the Kalman prediction (logged as a Phase
  8 negative result).

### Contact-exclusion cost (`w_exclusion`)
- **What:** repulsion penalty for assigning to a detection that other tracks' predictions converge
  on (cells cannot overlap).
- **Why dropped:** did not improve switch-rate/continuity; the gate + LAP already resolve most
  contested detections.

### Breadcrumb spatial-history cost (`w_breadcrumb`)
- **What:** attraction toward frequently-visited paths via a decaying voxel `BreadcrumbField`.
- **Why dropped:** off by default, no logged win; path-history is a weak prior for dense fields.

### ContextAssigner (learned joint transformer) — removed earlier
- **What:** a transformer that jointly scored track/detection pairs (4 variants tried).
- **Why dropped:** consistently traded switch-rate for continuity — never won both (see `FAQ.md`).
  Already absent from the code; recorded here for completeness (docs still referenced it).

---

## Segmentation / morphology

### HMM boundary-state morphology
- **What:** `fit_boundary_hmm`, `assign_boundary_states`, `assign_boundary_hmm_features`,
  `extract_boundary_features`, curvature/`fold_score` features and run-length smoothing in
  `coastal/morphology.py` — a hidden-Markov model over cell-boundary shape states, explored as a
  tracking/identity signal.
- **Why dropped:** cells are morphologically identical blobs; boundary dynamics carried no usable
  tracking signal (`MORPHOLOGY.md`, `FAQ.md`). Never wired into the live segmentation or tracking
  path.
- **Kept from this module:** `labels_to_polygons`, `extract_shape_features`,
  `extract_cell_morphology` — retained as a standalone morphology/QC readout (not currently wired
  into any pipeline).
- **Ruling it out would need:** cells with real, trackable boundary-shape dynamics.
