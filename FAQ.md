# Coastal — FAQ

The counterintuitive "why" behind coastal. Highlights only — detail lives in `docs/`.

**Q: How do you segment cells with no ground-truth labels?**
You don't need them. The UNet is trained on optical-flow *structure*, not annotations:
bright/edgy pixels (`IntensityLoss`) plus a contrastive objective that pulls together pixels
with similar flow signatures (`TemporalMetricsLoss`). The supervision is the physics of motion,
not a human tracing outlines. See `docs/SEGMENTATION.md`.

**Q: Why compute optical flow at four time scales instead of frame-to-frame?**
Instantaneous frame-to-frame flow is too noisy to separate touching cells that lack bright
reporters — different cells produce the same instantaneous motion. A *multi-scale temporal
signature* ([1,2,4,8]-frame gaps + cumulative displacement) turns "is this pixel moving now?"
into "what is this pixel's motion pattern over time?", which *is* discriminative.

**Q: Why two segmentation passes?**
One threshold can't catch both large cells and small fragments. Pass 1 uses large seeds + a low
affinity threshold for big cells; pass 2 uses small seeds + a high threshold on the leftover
space for fragments. See `docs/SEGMENTATION.md`.

**Q: You have confetti colours — isn't tracking easy?**
No. ~270 cells share each of only 3 channels, so colour is weakly discriminative at best. And
the cells are morphologically identical round blobs — greyscale appearance carries *no* signal.
So every appearance-based idea failed (HMM morphology, patch encoders, flow encoders). See
`docs/MORPHOLOGY.md` and `docs/TRACKING.md`.

**Q: Every learned tracker lost to a plain Kalman filter. Why?**
Because the discriminative signal genuinely isn't there in appearance. Four ContextAssigner
variants all traded `switch_rate` for `continuity` — softer gates, not better assignments. The
only things that helped were *geometric*: dense flow-warp cost and a raw confetti-colour cost in
the LAP. Lesson: when appearance is non-discriminative, don't spend model capacity on it.

**Q: btrack reports ~0.90 continuity — why isn't that the answer?**
Its global ILP interpolates straight-line "ghost" segments across gaps that count as surviving
tracks, inflating continuity. Coastal only tracks *actually-detected* cells; its lower raw
continuity is more honest, not worse. Compare like-for-like. See `TRACKING_SESSION_SUMMARY.md`.

**Q: Could this be Julia instead of Python?**
*Could*, yes — *should*, not now. It's technically feasible (two hard parts: the Torch DL stack
→ Lux/Flux, and `cv2.calcOpticalFlowFarneback` with no native Julia equivalent; everything else
has a mature package). But a port removes no dependency (torch *is* the package) and replaces
working code — pure consolidation, with no consumer asking for it. Same bar cecelia used on
itself: port when a need appears, not to chase "Julia-native". See `docs/JULIA_PORT.md`.
