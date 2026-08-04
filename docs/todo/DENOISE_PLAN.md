# Denoise: extract Cellpose 3 restoration → coastal, then repurpose temporal video denoising

Status: **planning** (2026-07-24)

## Goal

Give coastal a **denoising / restoration** module so that:

1. **cecelia can drop the `cellpose==3.1.1.2` pin** and move segmentation to Cellpose 4. Today
   the *only* thing holding cecelia on Cellpose 3 is the denoise task
   (`cecelia:python/cecelia/tasks/cleanupImages/cellpose_correct_run.py`, which imports
   `cellpose.denoise.DenoiseModel`, removed in v4). Segmentation (`.../tasks/segment/cellpose_run.py`)
   would benefit from v4.
2. coastal gains a **temporal denoiser for slow-cadence 3D+T intravital microscopy** — not by
   inventing a method, but by **repurposing an established one** (motion-compensated,
   self-supervised video denoising) to our scenario, the same way coastal already repurposed
   optical flow for segmentation and HMM for track clustering.

This is two deliverables at different maturities. **Part A (extraction) is the near-term unblock;
Part B (temporal) is a research track that Part A's scaffolding feeds into.**

---

## Why this shape — the decisive constraint

The blocker is **not** a torch/numpy version clash. `cellpose==3.x` and `cellpose==4.x` are the
**same PyPI distribution** — one environment can hold exactly one version. So "vendor Cellpose 3
denoise alongside Cellpose 4" is impossible *by construction*, at any version. The only path that
unblocks v4 is to remove the `cellpose` dependency from the denoise path entirely — i.e.
reimplement, with Cellpose 3 as a **reference, not a dependency**.

Confirmed from the installed metadata:

| | cellpose 3.1.1.2 (pinned now) | cellpose 4.2.1.1 (latest) |
|---|---|---|
| numpy | **`<2.1`, >=1.20** (the binding pin) | `>=1.20`, no upper bound |
| numba | `>=0.53` (more numpy pressure) | dropped |
| `DenoiseModel` | ✅ | **removed** |

---

## Decisions (2026-07-24)

1. **Own implementation, no `cellpose` dependency — ever.** Cellpose 3 is a reference. (Forced by
   the same-package mutual-exclusion above; also cleaner license/versioning story.)
2. **coastal owns the denoise module; cecelia calls it.** Reverses "cecelia does not depend on
   coastal" (`docs/JULIA_PORT.md`, ARCHITECTURE "cecelia seam"). Requires a real install story for
   coastal (see Decision 8).
3. **Keep the coastal seam clean.** `coastal.denoise` operates on **numpy / duck-typed
   array-likes**, imports no cecelia. cecelia's runner keeps doing all zarr I/O via its own
   `zarr_utils` and hands coastal in-memory planes/volumes. (Same contract as `data.py`.)
4. **Reuse coastal infra, don't fork it:** `device.resolve_device` (cuda→mps→cpu), the UNet
   conventions in `model.py`, the AMP/autocast pattern in `train.py`, and Farneback flow in
   `flow.py`. One canonical way per cross-cutting thing.
5. **Mixed precision is CUDA-gated only.** MPS is a known crash surface (cecelia removed
   torch-on-MPS entirely after `harmonypy` MPS segfaults). `train.py` already gates autocast to
   `device_type='cuda'`; the denoise inference path must do the same and keep a clean CPU fallback.
6. **Part B is repurposing, not invention.** Framing everywhere: "established video-denoising
   technique, transplanted to intravital + validated on our data + cadence requirement." No
   novel-method / first-ever claims.
7. **Standing acquisition requirement: ≤ ~15 s frame interval** (see measured basis below). Same
   contract that already governs coastal's optical-flow segmentation — a consistent package-wide
   assumption, not a new burden. We do **not** optimise for high-motion movies.
8. **coastal graduates from "dropped" to a shipping dependency** as a git or PyPI dep (not the
   editable non-git sibling-path that got it dropped — see cecelia `docs/SHIPPING.md` "coastal is
   dropped for now").
   **RESOLVED (2026-07-24):** coastal is **not rc-ready — dev-only for now**, so cecelia pulls it as a
   git dep **tracking `main`** (`coastal = { git = "https://github.com/schienstockd/coastal.git",
   branch = "main" }`); switch to a `tag` pin once coastal cuts a release. Local co-dev uses an
   editable shadow (`pixi run pip install -e ../../coastal`), which keeps `pixi run dev` unchanged.
   **Consequence:** making the *shipped* cecelia cleanup hard-depend on coastal + dropping the
   `cellpose==3` pin (Part A A4/A5, and the Cellpose-4 segmentation migration it unblocks) is **gated
   on a coastal release** — deferred, since it would tie cecelia releases to unreleased coastal. Until
   then the **`coastalDenoise` dev custom module** (sys.path shim to `~/cc-workspace/coastal`) is the
   way to use coastal denoise; no cecelia `pixi.toml` change / pin drop yet. Fallback if coastal never
   releases: vendor `coastal.denoise` into cecelia (reintroduces the duplication we avoided).

---

## Reference: what Cellpose 3 denoising actually is

Read from the installed source (`cellpose/denoise.py`, `core.py`, `resnet_torch.py`):

- **Network** = `CPnet` (`resnet_torch.py`) — the *same* residual-UNet-with-style used for
  segmentation, instantiated with `nout=1`, `nbase=[1,32,64,128,256]`. `resnet_torch.py` imports
  **only torch** (~290 lines, self-contained).
- **Inference chain**: `DenoiseModel.eval` → per-channel `_eval` → `core.run_net`. `run_net` does
  padding, 224-px tiling (`transforms.make_tiles`/`average_tiles`, 0.1 overlap), per-tile batching,
  and `_forward` (plain `net.eval()` + `torch.no_grad()`, **fp32, no autocast, no compile**).
- **Coupling is moderate.** The *inference* path needs only: `CPnet` + weight loader;
  `transforms.{normalize_img,resize_image,make_tiles,average_tiles,get_pad_yx,convert_image}`;
  `core.{run_net,assign_device,_forward}`; and `models.model_path` (weight download from
  `https://www.cellpose.org/models/<name>`). The training half of `denoise.py` is not needed.
- **Model names**: `denoise_cyto3` / `deblur_cyto3` / `upsample_cyto3` (+ nuclei variants). Weights
  are small (light UNet).
- **Output range** `[-1, 10]`; cecelia rescales to bit depth.

**Extractability verdict:** the runnable denoiser is a clean torch UNet + a normalize/tile/stitch
harness — a few hundred lines, no segmentation entanglement at inference time.

---

## Part A — Extraction (near-term deliverable)

### A1. `coastal/denoise.py` — the module — **DONE (2026-07-24)**
Reimplement CPnet forward + the normalize/tile/stitch harness in coastal's idiom. Route device
through `device.resolve_device`. Public API takes array-likes (Decision 3), e.g.:

```python
denoise_image(arr, model='denoise', diameter=None, device=None,
              batch_size=8, tile=224, tile_overlap=0.1) -> np.ndarray
```

Weights: load Cellpose's public checkpoints via a documented one-time conversion into coastal's
own state-dict layout, cached under a coastal models dir (respect an env override, mirror
`CELLPOSE_LOCAL_MODELS_PATH` behaviour). This keeps the BSD attribution chain while removing the
`cellpose` import.

### A2. Golden-value test (required — CLAUDE.md "cite sources") — **DONE (2026-07-24)**
`tests/test_denoise.py` asserts coastal's forward matches `cellpose.denoise.DenoiseModel.eval`.
Cellpose is a **test-time-only** dep (dev extra), never runtime. **Result: bit-identical** —
correlation 1.000000, max abs diff ~4e-7 (float32 noise floor), for both `diameter=None` and the
`diameter`/rescale path cecelia uses; the `(Z,Y,X)` stack call equals per-plane; verified
end-to-end on a real `ldYr8J` nuc-GFP plane on CUDA (auto device). `THIRD_PARTY.md` created.

### A3. Speed — **measured (2026-07-24)**
Benchmarked on real `ldYr8J` nuc-GFP planes (1082×1100), CUDA, vs the current cellpose per-plane
path. Output stays **bit-identical** (corr 1.00000) throughout:

| variant | speedup vs current |
|---|---|
| cellpose per-plane fp32 (current cecelia path) | 1.00× |
| coastal fp16 (autocast, CUDA-gated) | 1.27× |
| coastal fp16 + `torch.compile` | **1.93×** |

~2× lossless, with a one-off ~34 s compile cost (negligible amortized over a movie's thousands of
planes). **Correction to the original A3 assumption:** stack-batching in *fp32* was **not** a win
(≈0.89×, slightly slower than per-plane) — the real levers are **fp16 + `torch.compile`**, both
CUDA-only (Decision 5). `torch.compile` is opt-in (`DenoiseModel(..., compile=True)`).

Original opportunity list (for reference):
The reference path leaves easy wins on the table:
- **Mixed precision** — biggest single win (~1.5–2× + lower VRAM). `torch.autocast`, **CUDA only**
  (Decision 5).
- **`torch.compile`** — CPnet is static-shape per tile; good candidate (~1.2–1.5× on CUDA).
- **Stack-level batching** — cecelia currently calls `eval` on **one 2-D plane at a time**, so the
  batch machinery only ever batches tiles within a plane. Accepting whole Z/T stacks lets coastal
  batch across planes — large throughput gain on small tiles.
- Tiling (fixed 224, 0.1 overlap, CPU assembly) — leave as-is initially.

Order: fp16 → stack batching → `torch.compile`. Benchmark each vs the Cellpose 3 baseline.

### A4. cecelia integration
`cellpose_correct_run.py`: swap `from cellpose import denoise` → `from coastal import denoise`,
preserve the per-plane output contract (`[-1,10]` → bit-depth rescale) or hand coastal whole
stacks (A3). The Julia handler `cleanupImages/cellpose_correct.jl` is unchanged (thin `run_py`).
No QC change — perceptual denoising is the sanctioned QC exemption.

### A5. Env + shipping
- Add coastal as a cecelia dependency (Decision 8): git/PyPI dep, not editable sibling path.
- **Drop `cellpose==3.1.1.2`** from `cecelia:pixi.toml`.
- **Separately** migrate `segment/cellpose_run.py` to Cellpose 4 (this plan *unblocks* it; the v4
  model-zoo/default-flow migration is its own work).
- Update cecelia `docs/SHIPPING.md` (both the cellpose-pin rationale **and** the "coastal dropped"
  note) and the FAQ cellpose-v3 line.

---

## Part B — Temporal denoising (repurpose, validated on our data)

### The gap we're filling — and why the incumbents don't fill it
Temporal fluorescence denoisers (DeepCAD-RT, DeepVID, SRDTrans) assume **consecutive frames show
the same signal, only the noise differs** — true for two-photon calcium (neurons don't move),
false when cells migrate (the pixel change *is* signal). Documented failure: on moving content they
**over-smooth or produce fixed-pattern artifacts**. The 2024–25 follow-ups escape this by
*dropping* the temporal axis (spatial-only / spatial-angular redundancy) rather than solving
motion. And the repos are effectively dormant (DeepCAD-RT last code push May 2025; SRDTrans
Aug 2024) — never established an ecosystem. So the "gated middle" — *use* temporal redundancy for
moving cells at a sane cadence — is unoccupied. We fill it by **transplant, not invention**:
motion-compensated / flow-aligned self-supervised video denoising is standard in general CV
(flow-aligned blind-spot networks; motion-compensated video denoising) — we apply it to intravital.

### Why it works here — measured on real data (`zolIMa/ldYr8J`, driftCorrected)
Image: 181 T × 4 C × 43 Z × 1082 × 1100, uint8, 0.331 µm/px, channels `SHG` (0, static —
second harmonic, exclude), `nuc-GFP` (1), `mem-TOM` (2), `CD169-Kat` (3). Cadence 15 s.

Motion measured at **30 s gaps (conservative upper bound; 15 s ≈ half)**, foreground, Z-max-proj:

| Channel | p50 | p90 | p99 | frac fg > 2 px |
|---|---|---|---|---|
| nuc-GFP | 0.31 µm (0.9 px) | 0.89 µm (2.7 px) | 1.77 µm (5.3 px) | 19.5% |
| mem-TOM | 0.39 µm (1.2 px) | 0.87 µm (2.6 px) | 1.54 µm (4.6 px) | 20.8% |
| CD169-Kat | 0.22 µm (0.7 px) | 0.72 µm (2.2 px) | 1.31 µm (3.9 px) | 12.6% |

Raw consecutive-frame similarity: **SSIM ≈ 0.89–0.93, PSNR ≈ 28–29 dB**, mean |Δ| ≈ 0.02–0.03
on [0,1]. → At 15 s the *median* cell moves sub-pixel-to-~1 px/frame. **Temporal redundancy is
real and exploitable; the DeepCAD failure mode does not bite at this cadence.** This is exactly
why Decision 7 (≤15 s) is the enabling precondition, not a hand-wave.

**B0 confirmed set-wide (2026-07-24)** — swept all 12 images of set `OLifi6` ("MERTK", project
`zolIMa`), driftCorrected, at the real **15 s** cadence (consecutive frames, mid-Z slab):

| Channel | p50 | p90 | p99 | fg > 2 px | SSIM |
|---|---|---|---|---|---|
| nuc-GFP (n=12) | 0.03 µm (0.1 px) | 0.45 µm (1.4 px) | 1.30 µm (3.9 px) | 5% | 0.908 |
| mem-TOM (n=12) | 0.16 µm (0.5 px) | 0.69 µm (2.1 px) | 1.42 µm (4.3 px) | 12% | 0.856 |

`ldYr8J` sits on the *higher-motion* end — the set is if anything more redundant than the single-image
test suggested. Two takeaways that harden the design: **(i)** mem-TOM moves ~2–3× more than nuc-GFP
(membrane protrusion vs nucleus) and two movies (`2yvS9D`, `YOXLrK`) carry a fat tail (mem-TOM p99
~2.5–2.8 µm, 28% fg > 2 px) — so the motion-comp branch must be **gated per-region/per-movie**, a
fixed global warp would be wrong; **(ii)** validate Part B on **mem-TOM + `2yvS9D`/`YOXLrK`** (the
stress cases), not just the easy nuclear channel.

### Two design facts, also measured (not speculation)
1. **Global flow warping *hurts* when motion ≈ 0** — in the low-motion slice, warping every frame
   dropped PSNR ~1.6 dB (Farneback adds warp noise where there's nothing to fix). So motion
   compensation must be **gated by flow magnitude** — applied only to the moving tail (~15–20% of
   foreground moving > 2 px/frame), leaving the static bulk untouched.
2. **Memory is a non-issue at this cadence** (PhenoCycler explicitly out of scope). Per
   (channel, timepoint) volume ≈ 51 MB uint8 / 204 MB float32; a 5-frame window ≈ 1 GB float32,
   trivially streamable — the cleanup path already streams timepoint-by-timepoint.

### Design (transplant + gate)
- **Training signal**: self-supervised (Noise2Noise across neighbouring frames). No clean ground
  truth exists for intravital; measured SSIM ~0.9 makes neighbouring frames valid noisy targets of
  the same signal — well-posed on this data.
- **Architecture**: 3D-aware (exploit the 43 Z slices too) spatiotemporal denoiser, reusing
  coastal's UNet conventions (`model.py`) and AMP training (`train.py`, `TemporalDatasetWithAugmentation`).
- **Motion-compensation branch (gated)**: use coastal's existing Farneback flow (`flow.py`) to
  warp only the fast-moving tail before temporal pooling; leave the static bulk to plain
  spatiotemporal pooling. Degrades gracefully instead of over-smoothing moving cells (the incumbent
  failure) and avoids warp-corrupting the static majority (the global-flow failure).
- **Cadence contract**: document ≤ ~15 s as a requirement (Decision 7).

### Methodology — heed coastal's own dead ends (`docs/DEAD_ENDS.md`)
This is an **experiment, not a scheduled feature** — treat it like the tracking work, which killed a
long list of elaborate methods (ABM tracker, learned `w_app` embeddings, collective-motion cost, HMM
boundary states) because each **added complexity without beating a simple robust baseline**. Apply
the same discipline here:
- **Baseline-first.** Stand up the simplest self-supervised temporal denoiser and the two trivial
  baselines (per-frame denoise; Cellpose-3 restoration) *before* any motion-compensation cleverness.
  Motion comp (B2) must *earn* its place against the plain temporal baseline, or it's ruled out.
- **One credible signal to build on:** the tracking work *kept* `w_flow` (dense flow-warp) — flow is
  reliable enough to warp frames — so flow-guided temporal alignment is a reasonable bet, but still
  gated on measured gain.
- **Honest eval without clean ground truth.** No clean target exists, so PSNR-vs-GT is out. Use:
  (1) **held-out-frame self-supervised loss** (predict a left-out frame from neighbours — the N2N
  validation signal); (2) a **downstream proxy** — does denoising improve segmentation/tracking
  scores (coastal already has `score_tracking`)?; (3) qualitative check that **moving cells aren't
  smeared** (the exact failure mode of DeepCAD-RT et al.). A method that wins (1) but smears cells
  (fails 3) is ruled out.
- **Log negatives.** Keep a running experiment log (like `TRACKING_SESSION_SUMMARY.md`) so a tried
  approach is *recorded as ruled out*, not silently re-tried.
- **Expectations:** no miracle on a one-week horizon; the deliverable of B1 is a *measured* baseline
  + a go/no-go read, not a shipped denoiser.

### Part B phases
- **B0 — DONE (2026-07-24)** — ~1 px/frame redundancy confirmed across all 12 `OLifi6` images
  (see the set-wide table above); locked as the design assumption.
- **B1** — self-supervised spatiotemporal baseline (**no** motion comp). Reuse coastal's training
  infra (`train.py`, `TemporalDatasetWithAugmentation`, AMP); train on `zolIMa`/`OLifi6` moving
  channels (validate on the stress cases: mem-TOM, movies `2yvS9D`/`YOXLrK`). Compare vs per-frame
  denoise + Cellpose-3 restoration on the eval signals above. **Output: a go/no-go, not a feature.**
- **B2** — *only if B1 shows temporal redundancy is being left on the table* — add the flow-gated
  motion-compensation branch; ablate the gate (measure the tail win, confirm no static-region
  regression — global warp HURT PSNR at near-zero motion, per B0).
- **B3** — *only if B1/B2 clear the bar* — promote to a permanent `coastal/docs/` area doc; wire an
  optional temporal mode into cecelia's cleanup task.

### B2 measured ahead of B1 — motion compensation works, but does not displace the spatial gain

Run directly (no training) on the clean re-import: warp each neighbour into the target frame with
`flow.calc_flow_farneback_between_frames`, then average. 4 movies × 2 z-planes × 3 frames,
windows 3–15 (`diag/e11`, `e12`).

| | n=3 | n=5 | n=9 | n=15 |
|---|---|---|---|---|
| plain average — noise | −5% | −20% | −36% | −45% |
| **motion-compensated — noise** | **−13%** | **−27%** | **−41%** | **−49%** |
| plain — identity | 93.8% | 92.7% | 91.5% | 90.4% |
| **motion-compensated — identity** | **94.2%** | **92.9%** | **91.9%** | **91.0%** |

**Motion compensation wins at every window on both axes** — worth roughly a 1.4× shorter window
for the same noise (MC at n=5 ≈ plain at n≈7). That is a real, reproducible effect and it
validates the B2 idea.

Three things it also settles, two of them against the plan as written:

- **The premise "plain averaging gets worse the longer the window" is wrong as stated.** Plain
  averaging keeps *reducing noise* monotonically (−5 → −45%); what degrades with window length is
  **identity** (93.8% → 90.4%). The earlier table said "worse" because it was reading a purity
  column measured on already-denoised data. The real shape is a trade-off, not a turning point,
  and motion compensation shifts the trade-off rather than removing it.
- **It does not help the original errand.** On prob-map speckle, MC temporal is *worse* than a
  cheap spatial smooth inside the gain wrapper: at prob 0.6, `mc9` gives 87.4% recall / 167 blobs
  / 3.3% area against the spatial gain's 88.2% / 138 / 2.6%. Segmentation wants the projection
  cleaned spatially; temporal averaging is the tool for noise *inside a channel*, which a scalar
  gain cannot touch.
- **Both temporal variants cost far more identity than the gain wrapper** (≈92% at n=5 against
  99.8%), because warping resamples and mixes. For confetti, that is the expensive axis.

**Methodological trap, worth keeping:** there is no single noise metric that is fair to both
families. Temporal sd is mechanically driven to zero by temporal averaging; high-frequency spatial
residual is mechanically driven to zero by spatial smoothing. Each flatters its own family. The
numbers above use high-frequency residual, which if anything *under*-credits the spatial arm — and
the spatial arm still wins on the downstream task. Only the task metric (recall vs foreground
area) is non-circular; prefer it for any cross-family comparison.

**Verdict on B2:** the mechanism is confirmed — but see the correction below. It does not improve
segmentation, and for this pipeline it actively hurts.

### Correction — an invalid comparison, and what it hid (`diag/e13`, `e14`)

The first prob-map run substituted the denoised plane into an otherwise-**raw** 20-frame stack.
The UNet is conditioned on flow metrics computed *between* frames, so that frame's metrics came
from optical flow between a temporally-averaged image and 19 noisy neighbours, while the spatial
arm had every frame denoised. That produced a bloated foreground (6.7% vs raw's 4.3%) and made
temporal denoising look far worse than it is. Redone with all 20 frames denoised (`e13`), recall
at matched foreground area:

| arm | 2% | 3% | 5% |
|---|---|---|---|
| raw | 73.5% | 83.9% | 89.5% |
| `gaussian_restorer(1)` | 85.3% | 88.2% | 96.5% |
| **plain temporal mean, n=3** | **86.9%** | **93.6%** | **99.7%** |
| motion-compensated, n=3 | 79.1% | 87.4% | 94.1% |
| motion-compensated, n=9 | 84.6% | 83.8% | 94.9% |

Two reversals:

- **Plain temporal averaging is the best restorer**, ahead of every spatial option.
- **Motion compensation is consistently WORSE than plain averaging** for segmentation — the
  opposite of the plan's expectation. The mechanism is specific to this pipeline and should have
  been predictable: coastal's segmenter is **flow-supervised**. Warping frames into alignment is
  precisely the operation that removes the inter-frame motion the UNet is conditioned on. Motion
  compensation improves the *image* (e12: less noise, cells stay sharp instead of smearing) while
  degrading the *features the model reads*.

**Adopted:** `temporal_mean_restorer(3)` inside the ratio-preserving gain. It matches averaging
the channels directly on segmentation (92.3% vs 92.3% at 3% area) while keeping identity at
**99.5% against 97.6%** — the wrapper is free here. Default stays `gaussian_restorer` only
because `denoise_preserving_ratio` also accepts single timepoints, where a time axis does not
exist.

**Verdict on B1:** weaker than when the plan was written. A three-frame box mean already captures
most of the available temporal redundancy; training a spatiotemporal net has to beat that, not
just beat raw. And the one axis a learned model would plausibly win — clean per-channel
intensities — is not what bottlenecks segmentation. Revisit only if a downstream consumer needs
them.

---

## Attribution (Step 5)

Cellpose is **BSD-3** (Stringer & Pachitariu, HHMI). Cellpose 3 restoration derives
methodologically from **CARE / CSBDeep** (Weigert et al. 2018) and Noise2Void/Noise2Self-style
self-supervision — preserve that chain even in a clean reimplementation.

- **coastal**: create `THIRD_PARTY.md` (does not exist yet) with a BSD-3 entry for the adapted
  CPnet architecture + weights, naming the exact weights version (`cellpose 3.1.1.2`); inline BSD
  header in `denoise.py`; cite CARE/CSBDeep + the transplanted video-denoising references for Part B.
- **cecelia**: add the same BSD-3 / weights-version entry to the existing `THIRD_PARTY.md`.

---

## Risks / reservations
- The v3→v4 **segmentation** migration is real, separate work this only unblocks (v4 is SAM-backed,
  different model zoo/default flow).
- coastal-as-shipping-dependency needs the git/PyPI install story resolved before the pin removal
  can ship (Decision 8).
- MPS remains an untested crash surface for any torch denoiser — needs explicit Apple-Silicon
  CPU-fallback testing (Decision 5).
- Matching Cellpose 3 *quality* depends on faithfully reproducing its **normalization**, not just
  the architecture (A2 golden test guards this).
- Part B's redundancy claim is validated on `ldYr8J`; B0 must confirm it set-wide before commit.

## References
- Cellpose (BSD-3): https://github.com/MouseLand/cellpose
- OpticalFlow3D, Lee et al., *J Cell Sci* 2026 — voxel LK flow for motion (context, not denoise):
  https://doi.org/10.1242/jcs.264851
- DeepCAD-RT, *Nat Biotech* 2022 (temporal-redundancy assumption + its failure mode):
  https://www.nature.com/articles/s41587-022-01450-8
- SRDTrans / spatial redundancy transformer, bioRxiv 2023 (drops temporal dependence):
  https://www.biorxiv.org/content/10.1101/2023.06.01.543361v1
- Flow-calibrated self-supervised video denoising, arXiv 2412.11820 (the transplanted technique).
- CARE / CSBDeep, Weigert et al., *Nat Methods* 2018 (upstream attribution).


---

## Measured provenance + denoise findings (2026-07-31)

Recorded because it was slow to recover and is needed the moment the re-processed project lands.

### What the current movies actually are

`ccidDriftCorrected.zarr` is **already cellpose-denoised** — the name does not say so. From the
per-image cecelia logs (`ANALYSIS/1/<uid>/log/`) for `fFnZOv`:

| step | when |
|---|---|
| `importImages.omezarr` — bioformats2raw 0.11.0, `.oir` → `ccidImage.ome.zarr` | 2026-06-18 12:21 |
| `cleanupImages.cellposeCorrect` | 2026-06-18 15:58 |
| `cleanupImages.driftCorrect` | 2026-06-19 11:43 |
| `importImages.remove` ×2 — intermediates deleted | 2026-06-19 12:09–12:10 |

So the order was **denoise → drift correct**, and the raw + `ccidCpCorrected` intermediates were
removed, leaving only the drift-corrected result. Settings used:

```
{'model': ['denoise_cyto3'], 'modelChannels': [0,1,2,3], 'modelDiameter': [10]}   # 4-channel
{'model': ['denoise_cyto3'], 'modelChannels': [0,1,2],   'modelDiameter': [10]}   # 3-channel
```

`modelDiameter=10` is worth revisiting: cells measure ~15–20 px across, so denoise was told to
expect objects about half their real size, which biases it toward preserving fine structure — i.e.
the speckle we are trying to remove.

**Consequence for every measurement in `docs/SEGMENTATION.md`:** the ~10% residual noise and the
4518-blob prob map are what *survives* cellpose, not raw sensor noise. Any second denoise pass
applied to `ccidDriftCorrected` (as measured below) is double-denoising.

### Sources

`.oir` originals live on the rclone Google Drive mount at
`/home/dominik/gdrive/Notebook/DATA/TCELL_TYPES/{20211125,20211128}/`, ~1.03 GB each plus sidecar
`_00001/`, `_00002/` chunk folders which hold the bulk of the pixels. The nine TRAIN/TEST movies map
to `M{1,2,3}-{1,2,3}-B6-naive-gBT-uGFP-OTI-CTV-P14-ubTomato-z{230,250,280,300}*.oir`; the exact
per-uid filename is in each image's `importImages.omezarr.*.log`, and also in the OME XML
(`<zarr>/OME/METADATA.ome.xml`, `Name="....oir"`).

### Why cellpose denoise is wrong for confetti

Measured on `fFnZOv` (a second pass over already-denoised data, so read the direction not the
magnitude): `denoise_cyto3` rescales per channel, per image to ~[0, 1] — background p25 6429 → 0.6,
dynamic range 4635 → 0.5 — which destroys absolute intensity **and** the inter-channel ratios that
carry confetti identity. Purity of the brightest 10% of pixels fell 0.700 → 0.660.

So coastal's denoiser has a hard requirement cellpose does not meet: **preserve per-channel absolute
intensity**, since colour identity is the ratio between channels.

### Intensity-preserving alternatives, measured

Middle frame of a 9-frame window, `fFnZOv` z=7. All preserve background within 2% (unlike cellpose):

| method | bg preserved | bg noise sd | purity | dominant-colour agreement |
|---|---|---|---|---|
| raw (`ccidDriftCorrected`) | 100% | 385 | 0.703 | 100% |
| spatial gaussian σ=1 | 100.8% | 218 (−43%) | 0.642 | 91.6% |
| temporal mean, 3 frames | 100.9% | 293 (−24%) | 0.640 | 90.7% |
| temporal mean, 5 frames | 101.3% | 238 (−38%) | 0.615 | 86.1% |
| temporal mean, 9 frames | 101.7% | 200 (−48%) | 0.589 | 81.3% |
| spatial median 3×3 | 100.3% | 377 (−2%) | 0.637 | 87.5% |

Every option costs purity, and **plain temporal averaging gets worse the longer the window** —
cells move enough between frames to smear across positions and mix colours. That is the argument for
Part B's motion compensation: it is the only route that trades noise for √N without spatial mixing,
and the Farneback fields are already computed in `flow.py`.

Caveat on the purity column: it is evaluated at pixels chosen as brightest in the *raw* frame, which
penalises any spatial redistribution, so those drops are an upper bound on the true cost. A per-cell
purity measure would be fairer.

### Next

Dominik is re-processing from the `.oir` originals in cecelia feijoa with the order reversed —
**drift correct first, denoise second** — keeping the intermediates this time. With raw +
drift-corrected + denoised all available for one movie (`fFnZOv`, which every measurement in this
session used), the denoiser can finally be tuned against a real before/after instead of a
double-denoised proxy.

---

## Ratio-preserving restoration — measured on clean data (2026-08-01)

The re-import landed: nine `kSUFux/jHMfOI` movies from the `.oir` originals on one shared 8-bit
window `[0, 1500]`, drift-corrected, valid boxes written, **not** previously denoised. This is the
first honest before/after; everything above this line was measured on double-denoised data.

Result: **`denoise.denoise_preserving_ratio`** — smooth the mean projection once, apply it back to
every channel as a per-pixel scalar gain. **The default restorer is a Gaussian, not Cellpose**:
the ablation below puts the net 0.7pp ahead, which does not justify making a weights download the
default path for confetti data. `cellpose_restorer()` stays available as an explicit opt-in. This
keeps the "get off Cellpose restoration" goal intact rather than re-entering by the back door.

Scripts, logs, per-row JSON and figures: `~/Downloads/TMP/coastal-denoise-experiments/`.

### The problem it solves

Restoring each channel independently runs three different nonlinear corrections on the three
numbers whose ratio *is* the confetti label. Measured: dominant-channel agreement with raw is
**78% of cell pixels**, and the wrapper is not the cause — normalising all channels through one
shared window instead of per-plane reaches only 81%. On mask-integrated colour (what tracking's
`w_color` consumes) per-channel restoration reaches 95.1%, against **99.8% for the gain**, with a
5.3× smaller colour shift (L1 on the normalised channel vector: 0.0564 → 0.0106).

Preservation is exact per pixel; mask-integrated colour is a gain-*weighted* mean, hence 99.8% not
100%. **Absolute intensity is not preserved** — the gain is what changes it. Absolute-brightness
measurements must read the raw store.

### Segmentation effect (the original errand: speckle in the prob map)

35 conditions — 6 movies × 3 z-planes × 2 timepoint windows — recall scored against cell seeds
taken from the *raw* grey, so no arm is favoured. At `prob_threshold=0.6`:

| | recall | fg area | blobs | cell mask px |
|---|---|---|---|---|
| raw | 85.7% | 3.7% | 745 | 184 |
| gain | **89.6%** | **2.5%** | **145** | 168 |

Better recall, less foreground, **5× fewer blobs**, slightly tighter masks. Better on 5 of 6
movies, tied on the sixth. At the instance level (E3) spurious labels drop ~4× at equal recall and
the merge rate is flat-to-lower — the tighter-but-fuller masks do **not** fuse neighbours.

### Three findings that bound the claim

- **The restorer matters far less than the idea.** Ablation (E9, prob 0.6): raw 84.5% / 596 blobs,
  `gaussian_filter(proj, 1)` 87.5% / 158, `denoise_cyto3` 88.2% / 138. Most of the win is *smooth
  the projection*; the net buys +0.7pp and ~13% fewer blobs over a free gaussian. `ratio_preserving_gain`
  therefore takes `restored` as an argument rather than owning a model. `denoise_nuclei` fails
  outright (60% of the frame foreground); cyto2 ≈ cyto3.
- **The advantage is largest on bright cells.** At matched foreground area (E10), gain leads by
  ~+6pp on p99 seeds, +2–7pp on p97, and −0.8 to +3.7pp on p95. Scored at a fixed *threshold*
  instead (E6) gain appears to lose dim cells by 11pp — that is an artifact of raw's larger mask,
  not sensitivity. **Compare denoising arms at matched area, never at matched threshold.**
- **`GAIN_CAP` is a guard, not a knob.** Swept 1.0 → 4.0: recall flat (~89%) at every value,
  everything ≥ 1.3 numerically identical. It binds on ~0.5% of pixels (those where the projection
  sits at the noise floor; measured max ratio 2.6e4). Clipping rescales the shared factor, so it
  does **not** break ratio preservation.

### It composes with `prob_blur_sigma`

Judged at matched foreground area rather than matched threshold, `prob_blur_sigma` is a legitimate
tool, and denoising the input does not replace it — the two stack:

| arm | recall @1% area | @2% | @5% | blobs @1% |
|---|---|---|---|---|
| raw | 68.1% | 80.5% | 87.8% | 104 |
| raw + σ=1 | 70.5% | 81.5% | 89.1% | 47 |
| gain | 75.9% | 86.3% | 93.7% | 63 |
| **gain + σ=1** | **77.2%** | **87.1%** | **93.8%** | **42** |

### Reservations

- The UNet (`coastal_model.pt`) was trained on the **old** double-denoised, per-channel-windowed
  movies. Every number here applies it out of distribution. A retrain on gain-restored input is
  untested and could move all of it.
- Per-channel temporal noise barely improves (−10%), far worse than a plain gaussian (−68%) or
  median 3×3 (−69%) — a scalar gain cannot remove noise that lives inside a channel. The
  segmentation win comes from the projection, not from cleaner channels.
- 3D stitching (E8): gain does not harm it and modestly helps (single-z labels 76.5% → 68.3%,
  median volume 50 → 69), but **both** arms under-stitch — median z-extent 1.0 where 10:1
  anisotropy (5 µm z vs 0.497 µm xy) and ~8–10 µm cells predict ~2. Pre-existing, worth its own
  look.
- Seeds come from local maxima of the raw grey, so "recall" is detection of *those* maxima, not of
  ground truth. There is no GT here.
- cecelia's `cleanupImages.cellposeCorrect` still denoises "each channel independently" — the path
  that produced the 78%/95% identity above. Adopting this upstream is a separate cecelia change.
