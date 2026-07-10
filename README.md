# 🚧 Coastal

[![CI](https://github.com/schienstockd/coastal/actions/workflows/ci.yml/badge.svg)](https://github.com/schienstockd/coastal/actions/workflows/ci.yml)
[![License: GPL v3+](https://img.shields.io/badge/License-GPLv3+-blue.svg)](LICENSE)

> 🚧 **Work in progress — this package does not work yet.** The methods here are research
> *principles still being developed and validated*, not a finished or installable tool. Expect
> incomplete paths, rough edges, and breaking changes, and note that **nothing here has been
> independently verified**. This repository is public for **transparency and to keep the work
> organised** — not for use. How it was built: [end of this README](#how-this-was-built).

**Instance segmentation of cells in 2D/3D microscopy — trained on optical-flow structure, no
ground-truth labels required.** Plus a cell-tracking module for 3D+T confetti fluorescent
microscopy.

The idea: separate cells from the *motion* in a movie rather than from hand-drawn masks. A UNet is
trained on multi-scale optical-flow metrics with two self-supervised losses, producing a
probability map and per-pixel embeddings that a two-pass region-grower turns into instance labels;
those labels then feed a Kalman + LAP tracker. Whether this *works* well enough to be useful is
exactly what's still being tested — see [`docs/TRACKING.md`](docs/TRACKING.md) for how open the
tracking side still is.

## Why it's built this way

- **No labels.** Segmentation supervision comes from optical-flow structure (bright/edgy pixels +
  a contrastive objective on flow signatures), not annotations.
- **Multi-scale temporal signatures.** Flow at time gaps `[1,2,4,8]` + cumulative displacement
  turns *"is this pixel moving now?"* (noisy) into *"what is this pixel's motion pattern?"*
  (discriminative) — enough to separate touching cells without bright reporters.
- **Two-pass inference** catches large cells and small fragments with different seed sizes and
  thresholds.
- **Honest tracking metrics.** Tracking is genuinely hard here (morphologically identical blobs;
  ~270 cells share each of 3 confetti channels), so coastal reports `continuity` /
  `switch_rate` on actually-detected cells — no ghost-track inflation.

See [`FAQ.md`](FAQ.md) for the counterintuitive design decisions.

## How it works

```
raw frames                       [T, H, W] or [T, Z, H, W]
  │  Farneback flow @ scales [1,2,4,8] + ~14–16 derived metrics   (flow.py)
  ▼
frame + flow metrics             [T, 1+M, H, W]
  │  UNetWithEmbeddings, 2-loss self-supervised training          (model/loss/train.py)
  ▼
prob map + pixel embeddings      [T, 1, H, W] + [T, 64, H, W]
  │  two-pass region growing on embeddings + prob                 (segment.py)
  ▼
2D instances → 3D+T labels       [T, Z, H, W]  (IOU stitch across Z, utils.py)
  │  Kalman + Hungarian LAP assignment, optional cost terms       (track.py, abm.py)
  ▼
tracks                           {track_id: {t: (z, y, x) µm}}
```

## Installation

```bash
git clone <repo-url> coastal && cd coastal
pip install -e .
```

Requires Python ≥ 3.9. GPU is optional (CUDA → Apple MPS → CPU auto-selected); Farneback optical
flow runs on CPU by design (no CUDA-OpenCV dependency).

## Quickstart

*The intended interface — shown for transparency. It is **not** guaranteed to run end-to-end today
(see the banner above); treat it as the shape the API is converging on, verified against the real
function signatures.*

### Segmentation

```python
import coastal

# 1. Frames → flow metrics, per movie (metrics computed independently per movie).
#    Each movie is [T, H, W]; frames are normalised to [0, 1] internally.
all_frames, all_metrics = coastal.prepare_data_for_unet_batch([movie1, movie2, ...])
train_frames, test_frames, train_metrics, test_metrics = coastal.train_test_split_per_movie(
    all_frames, all_metrics)

# 2. Train the UNet (no labels needed).
model, history = coastal.train_with_metrics(train_frames, train_metrics, num_epochs=50)

# 3. Segment a sequence (frames_prep + normalised temporal metrics for one movie).
segmentor = coastal.TwoPassSegmentationInference(
    model=model,
    seed_size_large=32, affinity_threshold_large=0.2, embedding_blur_sigma_large=1.5,
    seed_size_small=8,  affinity_threshold_small=0.8, embedding_blur_sigma_small=1.5,
    merge_affinity_threshold_large=0.90, merge_affinity_threshold_small=0.90,
    prob_threshold=0.3, min_component_size=10, device="cuda",
)
instances = segmentor.predict_sequence(test_frames[0], test_metrics[0])   # [T, H, W]
```

For 3D+T Z-stacks, use `coastal.Inference3D(...).predict_temporal_volume(volume)` — it segments
each Z-slice and stitches labels across Z with IOU matching. See
[`docs/SEGMENTATION.md`](docs/SEGMENTATION.md).

### Tracking

```python
import coastal

pix_res = {"x": 0.48, "y": 0.48, "z": 4.0}   # µm/px — anisotropy matters, tracking runs in µm

# instances_4d: [T, Z, H, W] labels from segmentation.
tracks = coastal.track_sequence(instances_4d, pix_res, search_radius_um=50.0)

# Score against confetti identity (volumes: raw movie; ch_indices: confetti channels).
scores = coastal.score_tracking(tracks, instances_4d, volumes, ch_indices, pix_res, verbose=True)
print(scores)   # continuity (↑ = less fragmentation), switch_rate (↓ = fewer wrong assignments)
```

Optional cost terms (dense flow-warp, confetti-colour distance, learned assigner, …) are all
weighted keyword args on `track_sequence`. What helps and what doesn't is documented in
[`docs/TRACKING.md`](docs/TRACKING.md) and [`TRACKING_SESSION_SUMMARY.md`](TRACKING_SESSION_SUMMARY.md).

## Documentation

`CLAUDE.md` is the index; depth lives in `docs/`:

| Doc | Covers |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Pipeline, module map, array-shape contracts, the cecelia seam |
| [`docs/SEGMENTATION.md`](docs/SEGMENTATION.md) | Flow metrics → UNet → 2-loss → two-pass → 3D stitch; default params; the flow-metric appendix |
| [`docs/TRACKING.md`](docs/TRACKING.md) | Kalman + LAP design, cost terms, scoring, current best results |
| [`docs/MORPHOLOGY.md`](docs/MORPHOLOGY.md) | Polygon + HMM boundary states (a tried-and-failed direction) |
| [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) | CMA-ES tuning of segmentation and tracking params |
| [`docs/DATA.md`](docs/DATA.md) | Format conventions and the cecelia data contract |
| [`docs/JULIA_PORT.md`](docs/JULIA_PORT.md) | Julia portability assessment (verdict: feasible, but not now) |

Living trackers: [`docs/TODO.md`](docs/TODO.md) · [`docs/ROADMAP.md`](docs/ROADMAP.md) ·
[`docs/MILESTONES.md`](docs/MILESTONES.md) · [`docs/FUTURE.md`](docs/FUTURE.md).

## Project layout

```
coastal/
├── coastal/          package source (flow, model, loss, train, segment, utils,
│                     track, abm, morphology, optimize, data, viz)
├── docs/             structured documentation (see table above)
├── notebooks/        live: optical_flow.ipynb, tracking.ipynb, pipeline_confetti_ceiling.ipynb
│   └── archive/      superseded prototypes
├── tests/            pytest suite (run: pip install -e . && pytest)
├── CLAUDE.md         documentation index + contributor rules
└── FAQ.md            the "why" behind the design
```

## Status & scope

**Work in progress — not a working tool.** This is a research exploration aimed at 3D+T confetti
T-cell microscopy (~800 cells/frame, 61 frames, 5 movies). Its *principles* are still being tested:

- The **segmentation** idea (flow-metric-trained UNet, no labels) is the more developed part but
  is not validated as a general tool.
- **Tracking** is an open research problem — no method yet beats both `continuity` and
  `switch_rate` simultaneously (see [`docs/TRACKING.md`](docs/TRACKING.md)).

The repo exists for **transparency and organisation**, not for others to install and run. It's a
standalone Python package designed to eventually slot into the **cecelia** image-analysis pipeline
at the array boundary (normalised frames in, label volume out); a possible Julia port is assessed —
and deferred — in [`docs/JULIA_PORT.md`](docs/JULIA_PORT.md).

## How this was built

**The research is Dominik's; the repository engineering was done with [Claude Code](https://claude.com/claude-code)
(Anthropic, Claude Opus 4.8), under his direction.** This split is different from a full
"AI wrote the software" project — worth stating plainly:

- **Dominik** owns the science: the segmentation/tracking *approach* (flow-as-supervision, learned
  embeddings, the confetti tracking problem), the model and pipeline code, the experiments, and all
  empirical judgement about what works. The core `coastal/` source and the research in the
  notebooks predate and sit outside the AI collaboration.
- **Claude Code** acted as an **engineer/librarian over that existing work**, not as its author. In
  these sessions it: structured the documentation (the `CLAUDE.md` index + `docs/` skeleton, `FAQ`,
  lifecycle trackers), wrote the first automated test suite (and surfaced a real behavioural quirk
  in the label-stitching while doing so), produced the Julia-portability assessment, did the
  cecelia integration/tooling (notebook data-loading, the packaging plan), and set up this
  repository (git, CI, conventions).
- **Not validated by the AI.** Claude's environment has no GPU, no PyTorch/OpenCV runtime, and no
  microscopy data, so it **could not run the package, train a model, or validate segmentation or
  tracking quality**. Everything empirical was — and must be — checked by Dominik. This is also why
  the package is marked not-yet-working above.

Scientific tools this pipeline builds on (each with its own license/citation): **PyTorch**,
**OpenCV** (Farneback optical flow), **scikit-image**, **scipy**, **btrack** (used from the
tracking notebooks for comparison), and the **cecelia** ecosystem it integrates with.

## License

GPL-3.0-or-later (aligned with cecelia, which coastal integrates with).
