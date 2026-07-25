# B1 experiment log — self-supervised temporal denoise baseline

Running log (per DENOISE_PLAN Part B "log negatives"). Newest first. Script: `scripts/b1_temporal_denoise.py`.

## 2026-07-25 — PARKED (Dominik). Cellpose is already a strong punctae-preserving baseline; temporal needs a design discussion.

**Decision.** Park the temporal-denoise experiment (B1/B2/B3) here. Ship cellpose denoise as coastal's
clean denoise capability (already on coastal main, `coastal.denoise`, denoise-only, bit-identical to
Cellpose 3). Temporal denoising is NOT ruled out but needs a **better design discussion on how to
actually use temporal information** before more code — the naive N2N transplant doesn't clear the bar.

**What settled it — the corrected cellpose baseline (ldYr8J cpCorrected, same acquisition as Dnm0rS):**
- The `cellpose` column in the first Dnm0rS montage was blank due to MY harness bug: cellpose's
  `DenoiseModel` outputs in its own range (~[-1,10] → rescaled, floors ~22), and I normalized it by the
  RAW image's scale (0–36), crushing it to black. Cellpose itself is fine.
- Real cellpose result (ldYr8J `cpCorrected`): removes the grain AND **preserves the small punctae**
  (the signal Dominik cares about) as crisp dots. Slight "posterized" membrane contours, but punctae
  intact. So cellpose (spatial, off-the-shelf) is a STRONG baseline, not a weak one.
- The learned N2N over-smoothed punctae into blobs → currently WORSE than cellpose on the feature that
  matters. Its "+2.5 dB" was vs raw, not vs cellpose-done-right. Plain L1 N2N penalizes the punctae
  (they flicker frame-to-frame, so predicting a neighbour averages them away).

**Implication for any future temporal work.** The bar is: preserve punctae ≥ cellpose AND remove noise
cellpose can't — beating *raw* is not enough. That needs a punctae/detail-preserving objective (not
plain L1), likely a residual/light-touch model, and a real answer to "what does temporal buy us over a
good single-frame denoiser" (candidate: temporally-consistent punctae that a single frame can't
disambiguate). Defer to the design discussion.


## 2026-07-24 — Data characterization flips the premise: raw 15 s data is NOISE-dominated (GO for N2N)

**Setup.** Switched to the correct-order data Dominik supplied: `obWDNS` / "MERTK crop" = image
`WcJXhC` in project `zolIMa`, **drift-corrected FROM RAW** (not cellpose-denoised-then-drift like the
old `OLifi6`/`ldYr8J`). 181 T × 4 C × 41 Z, 15 s, channels [SHG, nuc-GFP, mem-TOM, CD169-Kat]. Spatial
crop 297×329.

**Pre-flight characterization (should have been step 1 — B0 discipline).** Identical code, mid-Z,
global norm, gap-1..8:

| image | channel | RMSE_fg g1 | SSIM g1 | PSNR g1 | fg% | RMSE g1→g8 |
|---|---|---|---|---|---|---|
| WcJXhC (raw→drift) | mem-TOM | 0.42 | 0.43 | 15.2 dB | 9.5% | 0.42→0.44 (**flat**) |
| WcJXhC (raw→drift) | nuc-GFP | 0.53 | 0.42 | 15.1 dB | 5.5% | 0.53→0.55 (**flat**) |
| ldYr8J (dn→drift, B0 ref) | mem-TOM | 0.034 | 0.89 | 28.5 dB | 88% | 0.036→0.070 (2×) |

**Interpretation (decisive).** The gap curve is the noise-vs-motion test: noise is iid (gap-independent),
motion grows with gap.
- **Raw data: flat gap curve → the frame delta is almost entirely NOISE.** N2N is *well-posed*
  (neighbouring frames = same signal + independent noise) and there is a LOT to remove (PSNR 15 dB).
- **B0's SSIM~0.9 "redundancy" was measured on already-cellpose-denoised data** — its rising gap curve
  shows what's left there is *motion*, not available redundancy. B0 overstated the redundancy of the
  RAW acquisition and understated the noise. Correcting the record: the temporal-denoise motivation is
  **stronger** on real raw data, not weaker.
- Caveat: `WcJXhC` is a small crop that is ~90% background (fg 5–9% vs 88% on the full `ldYr8J` FOV) —
  a poor training/eval substrate. A fuller-FOV raw movie is wanted.

**NEGATIVE logged — first learned run's metrics were invalid on raw data.** Trained a compact N2N UNet
(coastal ConvBlock/Encoder/Decoder, 1→1, depth 3), input x_t → target adjacent frame, L1, AMP, 1500 it.
Two eval bugs, both from calibrating on *denoised* data:
1. **Per-frame percentile norm** put consecutive frames on different scales → injected artificial
   frame-to-frame variance. Fixed to a single **global** (lo,hi). (raw static residual barely moved,
   0.415→0.398 — because the real story is noise, not scaling.)
2. **Metrics assume the raw frame ≈ clean signal**, false on raw data: (a) residual-to-noisy-neighbour
   over ~90% background rewards blur; (b) Sobel "no-smear" reads raw noise as sharpness, so a denoiser
   removing noise shows a big Sobel drop (−86%) that CANNOT be distinguished from smear. Also normalized
   the Cellpose baseline by the raw scale → squashed it (−97% sharpness), obviously wrong.
   → The learned net's fg-residual "win" (+37% vs raw, +5% vs cellpose) is consistent with real
   denoising but **not trustworthy as a quality verdict**. Inconclusive on quality.

**GO/NO-GO.** GO on the *premise* (raw 15 s data is noise-dominated → temporal N2N well-posed and
worthwhile). Model quality UNPROVEN — blocked on a valid no-GT quality metric + a better substrate.

**Next (proposed, pending Dominik):**
1. **Valid quality eval without clean GT:** build a pseudo-reference by temporal median/mean over a
   *low-motion* window (gate frames where Farneback flow < ~0.5 px so cells don't smear the reference),
   then score each denoiser by SSIM/PSNR to that pseudo-GT + a visual QC panel (raw | cellpose | N2N |
   pseudo-GT). This distinguishes noise-removal from structure-smear, which Sobel cannot on raw data.
2. **Better substrate:** train on a fuller-FOV raw movie (more foreground, more cells) rather than this
   90%-empty crop — ask Dominik whether one exists / can be exported (the copy-image feature #347 can
   duplicate a raw version into a fresh set for a clean drift→train pipeline).
3. Re-run the learned N2N with fg-weighted patch sampling once the metric + substrate are sound.
