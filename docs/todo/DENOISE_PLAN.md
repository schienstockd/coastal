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
   dropped for now"). Blocks the pin removal shipping cleanly; resolve early.

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

### Part B phases
- **B0 — DONE (2026-07-24)** — ~1 px/frame redundancy confirmed across all 12 `OLifi6` images
  (see the set-wide table above); locked as the design assumption.
- **B1** — self-supervised spatiotemporal baseline (no motion comp), trained on `zolIMa` moving
  channels; compare vs per-frame denoise + vs Cellpose-3 restoration.
- **B2** — add the flow-gated motion-compensation branch; ablate the gate (measure the tail win,
  confirm no static-region regression).
- **B3** — promote to a permanent `coastal/docs/SEGMENTATION.md`-style area doc; wire an optional
  temporal mode into cecelia's cleanup task.

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
