# Data

> Update this file when you change the input/output format, `prepare_data_for_unet`, or the
> cecelia integration contract.

## Format conventions

- **Frames** — `[T, H, W]` (2D sequence) or `[T, Z, H, W]` (3D+T), float, normalised to [0,1]:
  `(frames - frames.min()) / (frames.max() - frames.min() + 1e-5)`.
- **Metrics tensor** — `[T, 1+M, H, W]`, channel 0 = frame, channels 1.. = flow metrics.
- **Instances** — integer labels, 0 = background, contiguous positives per frame.
- Physical units: Z = 4.0 µm/px, XY ≈ 0.48 µm/px. Tracking works in µm (see `docs/TRACKING.md`).

## `data.py`

`prepare_training_data(...)` converts `prepare_data_for_unet` output (from `flow.py`) into the
training format (flows dict + normalised metrics); `validate_training_data(...)` checks shapes
and value ranges. `data.py` operates on **numpy arrays only** — no zarr/tifffile/cecelia import.

## The cecelia integration contract

The "cecelia integration" is a *data contract*, not a code dependency:

- Nothing in `coastal/` imports cecelia, zarr, or tifffile.
- `train.py` accepts zarr/dask array-likes by **duck typing** (`.shape` + lazy slicing) so large
  chunked volumes can be streamed, but it never imports those libraries — the caller supplies
  the array-like.
- cecelia currently **does not depend on coastal** — its `docs/SHIPPING.md` records that it
  "dropped coastal". A Julia port is the cleanest path to reintegrating (see `docs/JULIA_PORT.md`).

**When wiring coastal to cecelia:** keep the seam at the array boundary. coastal takes a
normalised frame array in and returns a label volume out; I/O (OME-ZARR, h5ad, napari) is
cecelia's job, not coastal's.

### Streaming: pass the lazy array, never `np.asarray(movie)`

A confetti movie is `[180, 4, ~13, ~530, ~570]` uint16 ≈ **6.7 GB**, so the duck-typed lazy input is
not an optimisation — it is the contract. Every coastal entry point that takes a whole movie slices
it itself, and the axis it slices on is the thing to know:

| Entry point | Slices | Materialises at a time |
|---|---|---|
| `Inference3D.predict_temporal_volume` | `volume[:, :, z]` | one z-slice, all T (~450 MB) |
| `prepare_data_for_unet_batch_4d` | one `[seq_len, C, H, W]` window per sequence | one 20-frame window |
| `track.extract_cell_intensities` / `extract_cell_colors` | `volume[t][ch_indices]` | one timepoint |
| `abm.score_tracking` | via `extract_cell_colors` | one timepoint |

So `np.asarray(volumes[uid][0])` anywhere in a notebook throws the streaming away and adds 6.7 GB —
pass `volumes[uid][0]` itself. The exception is `abm.compute_cell_flow_features`, which needs real
pixels for Farneback: build its `grey` `[T,Z,H,W]` float32 array by looping z and reading
`np.asarray(vol[:, :, z])[:, ch].mean(axis=1)`, so the raw `[T,C,Z,H,W]` movie is never resident.

Segmentation output is worth caching. `[T,Z,H,W]` int32 labels are ~3.4 GB per movie; writing them
with `np.save` and re-reading with `np.load(..., mmap_mode='r')` keeps them out of RAM (every
downstream consumer indexes per timepoint and is read-only) and lets a kernel restart skip
segmentation. `notebooks/pipeline_consensus.ipynb` does this via `CACHE_DIR`.

Per-worker cost during segmentation is the other half of the budget — see
`docs/SEGMENTATION.md` → *Memory: the 4D path*.

### Metrics are lazy; materialise only for training

`prepare_data_for_unet` returns its metrics as a `TemporalMetrics` sequence rather than a list. It
indexes, slices and iterates exactly like the list it replaced, and `m[t]` is exactly
`extract_temporal_metrics(..., t)` — the difference is only *when* the 14 float32 planes are built.
This matters because those planes are the largest term on the 4D path (14 × T × H × W × 4 B = 3.1 GB
at T=180) and `predict_sequence` only ever reads one frame at a time.

The rule for a new consumer: **single pass → iterate it lazily; multiple passes → `list(...)` once.**
Training is the multiple-passes case (a Dataset indexes metrics per sample per epoch), so
`prepare_data_for_unet_batch_4d` and `prepare_data_for_unet_batch` already materialise for you.
Nothing caches, by design — caching all T would reinstate the memory the laziness exists to avoid.

## Notebook data loading (glue only — not package code)

The live notebooks load microscopy data via cecelia's Python IO helpers. As of 2026-07-08 cecelia
ships these as an **installable package** (`cecelia`, top-level `cecelia-feijoa/python/`), so the
notebooks just `import cecelia.utils.*` — no `sys.path`/`CECELIA_APP` hack:

```python
# one-time, into this notebook's env:
#   pip install -e /path/to/cecelia-feijoa/python      # pulls only cecelia's light IO deps
import os
import cecelia.utils.zarr_utils as zarr_utils
import cecelia.utils.ome_xml_utils as ome_xml_utils
from cecelia.utils.dim_utils import DimUtils

im, _ = zarr_utils.open_as_zarr(im_path, as_dask=True)   # list of pyramid levels
dim_utils = DimUtils(ome_xml_utils.parse_meta(im_path), use_channel_axis=True)
dim_utils.calc_image_dimensions(im[0].shape)
pix_res = dim_utils.im_physical_sizes()                  # → track_sequence's pix_res dict

# btrack (used in tracking.ipynb / pipeline_confetti_ceiling.ipynb) — supply a btrack config path.
# NOTE: cecelia's vendored config is NO LONGER shipped in the `cecelia` package — it moved next to
# cecelia's tracking *task* (cecelia-feijoa/app/src/tasks/tracking/cell_config.json), which is not
# part of the installable IO library. Point BTRACK_CONFIG at that file in a cecelia checkout, or let
# btrack use its own default (it downloads one). These notebooks' btrack path is experimental anyway
# — coastal's own tracker is Kalman+LAP (see TRACKING.md / DEAD_ENDS.md).
import os
BTRACK_CONFIG = os.environ.get("BTRACK_CONFIG")   # e.g. <cecelia>/app/src/tasks/tracking/cell_config.json
```

These helpers are **stateless** (plain paths / numpy / dask; no cecelia project state), and this is
notebook glue — it does **not** violate the "no cecelia import in `coastal/`" rule, which applies to
the package source only. The `cecelia` install pulls only cecelia's light IO tier (zarr>=3, dask,
tifffile, ome-types, scipy, scikit-image, tqdm) — **not** torch/napari/scanpy. `btrack` is coastal's
own notebook dep (`pip install coastal[notebooks]`), not cecelia's.

Caveats: this needs `cecelia` installed in the notebook kernel's env — either a wheel
(`pip install <cecelia-feijoa>/python`) or an editable path install
(`pip install -e <cecelia-feijoa>/python`). The old flat `py.zarr_utils` / `CECELIA_APP`
bootstrap is gone. cecelia's packaging design is recorded in
`cecelia-feijoa/docs/todo/PY_PACKAGING_PLAN.md`. (Update: the installable `cecelia` package is now
the **IO library only** — task runners and the vendored btrack `cell_config.json` live beside their
task under `cecelia-feijoa/app/src/tasks/`, not in the wheel. So `importlib.resources` no longer
finds `cell_config.json`; supply `BTRACK_CONFIG` explicitly as above.)

### Independent dev environment (pixi)

The notebooks used to run in a shared miniconda `r-cecelia-env`. They now have a **self-contained
pixi environment** (`pixi.toml`) so coastal is independent of any external conda env. pixi pulls a
pinned Python + conda-level bits from conda-forge and uses uv underneath for the PyPI resolve; it's
the same tool cecelia uses, so one workflow spans both repos.

```bash
pixi install        # build .pixi/ : Python 3.12, coastal (editable, +dev +notebooks extras),
                    #                and cecelia (editable, from the sibling checkout)
pixi run kernel     # register the "Python (coastal)" Jupyter kernel (select it in each notebook)
pixi run lab        # JupyterLab rooted at notebooks/
pixi run test       # pytest
pixi run doctor     # prints torch/cuda + cv2 + cecelia import — a quick "did the stack resolve" check
```

`pyproject.toml` stays the single source of truth for coastal's Python deps; `pixi.toml` only adds
what pip can't (pinned Python, the editable **cecelia** link, the Jupyter stack). The editable
cecelia line packages `scripts/link_cecelia.sh`'s job into the env declaratively — `pixi install`
links cecelia and keeps it linked across re-solves. Non-pixi users (plain venv / uv) still run
`scripts/link_cecelia.sh`; a non-sibling cecelia checkout means editing the path in `pixi.toml` (or
`CECELIA_PYTHON=... scripts/link_cecelia.sh` outside pixi). `.pixi/` is git-ignored; `pixi.lock` is
committed for reproducibility.

### Installing / keeping cecelia in sync

cecelia isn't on PyPI yet, so today it's a local install. Three modes, from best-for-co-dev to
future:

1. **Editable link (recommended while co-developing cecelia)** — run once:
   ```bash
   scripts/link_cecelia.sh          # → pip install -e ../cecelia/cecelia-feijoa/python
   # or: CECELIA_PYTHON=/path/to/cecelia-feijoa/python scripts/link_cecelia.sh
   ```
   **No per-change step**: edit cecelia's `python/cecelia/*` and coastal picks it up on the next
   import / kernel restart. This *is* the update routine.

2. **Frozen wheel** (isolation / a fixed cecelia) — `pip install ../cecelia/cecelia-feijoa/python`.
   A copied snapshot; **goes stale after any cecelia edit** — re-run to refresh. (This is what a fresh
   coastal env currently has until you run the editable link above.)

3. **Long-term — once cecelia is published to PyPI** (a real, versioned package): drop the
   script/path hack entirely and declare a normal pinned dependency in `pyproject.toml`, e.g.
   `dependencies += ["cecelia>=<x.y>"]` (or a `cecelia` extra), then plain `pip install`. At that
   point `scripts/link_cecelia.sh` can be removed. Tracked in `docs/TODO.md`; the cecelia-side
   publish is gated on the dist-name check in `cecelia-feijoa/docs/todo/PY_PACKAGING_PLAN.md`.
