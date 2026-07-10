# Roadmap

**Throwaway** forward goals, in phases — rewrite freely as priorities change. Durable "what
landed" history goes in `docs/MILESTONES.md`, not here. Consult this before starting a new phase.

## Phase A — Repo hygiene (in progress)
- Doc skeleton on the cecelia model ✅ (2026-07-08)
- First tests + CI wiring
- Notebook archive cleanup; fold `QUICK_REFERENCE.txt` into `docs/`

## Phase B — Tracking: crack the per-frame cost matrix
The diagnostic says 67% of switches are within-frame. Priorities:
- Learned confetti-colour embedding (metric space over RGB)
- Global LAP / network flow (for the remaining 33% at-gap)
Goal: beat continuity=0.848 **and** switch_rate=0.214 simultaneously (nothing has).

## Phase C — Segmentation robustness
- Y-cell splitting without over-merging
- Cross-movie generalisation of the flow-metric UNet

## Phase D — Julia port (conditional)
Only if reintegration with cecelia is prioritised. Follow the phasing in `docs/JULIA_PORT.md`
(utils/data → tracking core → Farneback decision → Torch stack → inference).

## Backlog (post the above)
- Deprecate or repurpose `morphology.py` (see `docs/FUTURE.md`)
- Package for distribution
