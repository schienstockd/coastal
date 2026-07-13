# Parked plans

This directory holds **parked plans**: standalone design documents for a feature too big to
capture as a `docs/TODO.md` item, but not yet built (or built in phases). Each is a `*_PLAN.md`
with the design work done up front — decisions, architecture, a phased build sequence — so the
thinking survives a context break and anyone (human or agent) can pick it up cold.

## What belongs here vs. the other trackers

| Doc | Holds | Shape |
|-----|-------|-------|
| `docs/TODO.md` | the backlog | numbered items, one line → one paragraph |
| `docs/todo/*_PLAN.md` | **parked plans** | full design doc: decisions + phases + architecture |
| `docs/FUTURE.md` | deferred *known-better* ideas | what / why deferred / when to revisit |
| `docs/ROADMAP.md` / `docs/MILESTONES.md` | forward phase goals / shipped ledger | high-level |
| `docs/<AREA>.md` | how a **built** subsystem works | permanent reference |

## When to create a parked plan

Create a `*_PLAN.md` when **any** of these is true:
- The feature needs **multiple locked decisions** and a **phased build sequence** — more than a
  TODO paragraph can hold.
- A topic is being **paused** but the design must be preserved.
- **Code needs a stable pointer** to the rationale (`# see docs/todo/X_PLAN.md (Decision 3)`).

If it fits in a paragraph and needs no design, it's a `docs/TODO.md` item, not a parked plan.

## Conventions

- **Name**: `<FEATURE>_PLAN.md`, SCREAMING_SNAKE (`COLOR_EMBEDDING_PLAN.md`, `JULIA_PORT_PLAN.md`).
- **Top matter**: a one-line `Status:` (planning / paused / in-progress) and a `## Goal`.
- **Locked decisions**: a dated `## Decisions` section, numbered so code/docs can cite them.
- **Phases**: an independently-checkpointed build sequence.
- **References**: use repo-relative paths (`docs/todo/<FILE>.md`), never absolute/`~`, so the
  pointer survives a checkout anywhere.
- **Promotion**: once the feature ships, move the durable "how it works" content into a permanent
  `docs/<AREA>.md` and either delete the plan or mark it historical at the top.

## Current parked plans

- [`CECELIA_NAPARI_UPSTREAM_PLAN.md`](CECELIA_NAPARI_UPSTREAM_PLAN.md) — extract coastal's napari
  viz helpers (`coastal/napari_viz.py`) into a generic `cecelia/utils/napari_utils.py` and have
  cecelia's `napari_bridge.py` delegate to it. **Work happens in the cecelia repo** — hand this
  file to a cecelia session (see its *Handoff*). Paused until cecelia's `feat/umap-facet` lands.

The Julia port lives as an assessment in `docs/JULIA_PORT.md`; promote it to a `JULIA_PORT_PLAN.md`
here if/when a concrete build is scheduled.
