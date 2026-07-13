# Cecelia napari-utils upstream

Status: **paused** — execute during the planned cecelia napari work (after cecelia's
`feat/umap-facet` lands and its tree is clean). This plan lives in the **coastal** repo but the work
happens in the **cecelia** repo; hand it to a cecelia session (see *Handoff*).

## Goal

Give cecelia a small, **generic** napari display layer — `add_image` / `add_labels` / `add_tracks`
that take **arrays + scale**, no project state — extracted from the logic currently inline in
`napari/napari_bridge.py`. This (a) de-duplicates cecelia's own bridge, and (b) pins the display
conventions in one documented place so coastal's viewers (`coastal/napari_viz.py`, shipped in
coastal PR #7) and cecelia's bridge render identically.

**Reference implementation:** `coastal/napari_viz.py` in this repo — `show_images`,
`show_segmentation`, `show_tracks`, `tracks_to_matrix`. It already encodes the target conventions
(docstring). Use it as the design; do not import it into cecelia.

## Context / why the import boundary matters

- coastal's package source **must not import cecelia** (coastal cross-cutting rule) — so
  `coastal/napari_viz.py` stays as coastal's own parallel implementation. Alignment with cecelia is
  by **shared conventions (documented), not shared code**.
- cecelia's **importable** package is the light IO tier (`pip install cecelia` pulls no napari —
  napari is an *environment* dep in cecelia's `pixi.toml`). So the new module ships in the wheel but
  must **lazily import napari** (napari provided by the pixi env), like cecelia's other heavy modules.
  See `../cecelia/cecelia-pineapple/docs/todo/PY_PACKAGING_PLAN.md`.

## Decisions (2026-07-13)

1. **Location:** `cecelia-pineapple:python/cecelia/utils/napari_utils.py`. Lazy `import napari`
   inside functions; module top imports only numpy. cecelia code style (2-space indent, module
   docstring like `dim_utils.py`).
2. **API (generic, array-level — no disk, no project state, no layer reconciliation):**
   - `add_image(viewer, data, *, scale, units=None, channel_axis=None, channel_names=None, colormaps=None, contrast=True, blending='additive')`
   - `add_labels(viewer, labels, *, scale, units=None, opacity=0.7, name='Labels')`
   - `add_tracks(viewer, tracks, *, scale, units=None, color_by='track_id', colormap='turbo', tail_width=4, tail_length=30, properties=None, name='Tracks')` — `tracks` is `[track_id, t, (z), y, x]` with vertices in **pixel** coords (scale supplies µm).
3. **Bridge keeps its brain.** `napari_bridge.py` retains ALL project logic — disk load of label
   zarr / label-props HDF5, populations, per-layer reconciliation + signature caching, colour-by
   columns (categorical Okabe–Ito / continuous viridis), timestamp, scale-bar, axis labels — and
   delegates only the final `self._viewer.add_*` calls to `napari_utils`. Do **not** move state into
   the generic layer.
4. **coastal stays parallel.** `coastal/napari_viz.py` is not changed to import cecelia (would break
   the no-cecelia-in-coastal rule). If cecelia's helpers drift from coastal's, align the conventions,
   not the code. (Maintainer's-call alternative: make coastal's viz notebook-only glue and switch
   the notebook to `from cecelia.utils.napari_utils import ...`; not recommended — it couples the
   coastal notebook's viz to cecelia.)

## Conventions to preserve (both repos)

- **scale** per-axis from `pix_res` — `[T,Z,Y,X]` array → `(1, z, y, x)` µm; pass to image, labels
  AND tracks so all layers align (napari warns + disables unit rendering if they mismatch).
- **units** set consistently across layers (cecelia reads them from OME-XML).
- **images:** one layer per channel (`channel_axis`), per-channel colormaps, `blending='additive'`,
  contrast from a sample.
- **labels:** `opacity=0.7`.
- **tracks:** `[track_id, t, z, y, x]` in **pixel** coords; `color_by='track_id'` → turbo;
  categorical colour-by → Okabe–Ito; continuous → viridis; `tail_width=4`, `tail_length=30`.

## Phases

1. **Add `napari_utils.py`** (generic helpers + a cecelia test — the napari-free parts, or headless
   with `QT_QPA_PLATFORM=offscreen`). cecelia DEV workflow: branch + PR, never push to `main`.
2. **Refactor `napari_bridge.py`** — `add_image` (~L126), `show_labels`' `add_labels` (~L273),
   `show_tracks`' `add_tracks` (~L656) delegate their `viewer.add_*` call to `napari_utils`; all
   surrounding disk/state/reconciliation logic unchanged. Verify the bridge still works
   (cecelia `test-py` + a manual viewer check).
3. **(Optional) reconcile coastal** — confirm `coastal/napari_viz.py` conventions still match; fix
   any drift. No cross-import.

## References

- `coastal/napari_viz.py` (reference implementation), coastal PR #7:
  https://github.com/schienstockd/coastal/pull/7
- cecelia bridge: `../cecelia/cecelia-pineapple/napari/napari_bridge.py`
  (`add_image` ~L126, `show_labels`→`add_labels` ~L273, `show_tracks`→`add_tracks` ~L656).
- cecelia packaging (light-tier / lazy-napari rationale):
  `../cecelia/cecelia-pineapple/docs/todo/PY_PACKAGING_PLAN.md`.

## Handoff

No live channel between Claude Code sessions — this file *is* the handoff. To execute, point the
cecelia session at this plan (sibling path `../coastal/docs/todo/CECELIA_NAPARI_UPSTREAM_PLAN.md`),
or copy it into `cecelia-pineapple/docs/todo/` once `feat/umap-facet` merges so cecelia finds it
natively. Promote to `cecelia-pineapple/docs/NAPARI.md` (or similar) once shipped.
