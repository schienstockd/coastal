# Development workflow

How we change this repo: branches, commits, pull requests, CI. For *what* the code does, see the
per-area docs linked from [`CLAUDE.md`](../CLAUDE.md); for how to run things see `CLAUDE.md` →
**Running**.

Repository: `git@github.com:schienstockd/coastal.git` (default branch **`main`**).

> Note: coastal is a **work-in-progress research repo** — the methods are principles still being
> validated, not a working tool (see the README banner). The workflow below still applies: keep
> `main` clean and reviewable.

## Golden rule — never commit or push to `main`

`main` is protected by convention. **All work lands via a feature branch + pull request**, even
docs and one-line fixes. Never `git commit`/`git push` directly onto `main`. (The initial
repo-bootstrap commit was the agreed last direct-to-main push.)

Agents (Claude Code): **ask before every commit and before pushing a branch / opening a PR —
explicitly, each time.** Do not commit or push proactively, even mid-task or after a general
"go ahead": approval to *make a change* is not approval to *commit or push* it — and `git push`
needs its own explicit yes. First show the file list + proposed commit message(s) + branch, then
wait. This is not the same as re-asking every turn: finish the work, report status, and let
Dominik call the commit. If the current branch is `main`, branch first.

## Branches

Branch off the latest `main`, named with a conventional-commit-style prefix:

```
feat/<slug>      # new capability        e.g. feat/color-embedding
fix/<slug>       # bug fix               e.g. fix/match-masks-relabel
docs/<slug>      # documentation         e.g. docs/tracking-update
chore/<slug>     # deps, tooling, infra  e.g. chore/repo-standards
refactor/<slug>  # behaviour-preserving cleanup
test/<slug>      # tests only
```

```bash
git switch main && git pull
git switch -c <type>/<slug>
```

Keep a branch scoped to one logical change.

## Commits

Conventional-commits style: `<type>(<scope>): <imperative summary>`, `type` ∈
`feat | fix | docs | chore | refactor | test | perf`. When authored by Claude Code, end the message
with:

```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

Ship the test in the same commit as core code, and update the relevant doc in the same change
(see `CLAUDE.md` → cross-cutting rules and the *Keep the docs current* table).

### State reservations before committing (agents)

Every time you're told to commit or push — **including** "what's the PR url?" (that request is
itself the go-ahead to commit + push, so don't stall on extra `git status` round-trips) — **first
volunteer your honest reservations about the change**, in the same turn, before running the commit.
A short, prioritized list separating:

- **Unverified — "go look":** what you did *not* actually exercise. What "unverified" means depends
  on where the agent is running:
  - *In a bare sandbox* (no torch/GPU/data): the package can't be imported or run at all — tests are
    reasoned through or stubbed, and CI (or Dominik) is the first real execution.
  - *In Dominik's workspace* (pixi + GPU available, as verified 2026-07-13): the agent **can**
    `pixi install`, `pixi run test`, and `pixi run doctor` — so import, the GPU torch build, `cv2`,
    the editable cecelia link, and the `tests/` suite are genuinely exercised. Say so when you did.
  - **Still unverified either way:** anything needing **real microscopy data** — models are never
    trained, and segmentation/tracking *quality* is never run on real images. That empirical
    judgement remains Dominik's.
- **Real limitations:** edge cases not handled, silent no-ops, perf, stale-state paths.

If any reservation is material, pause for Dominik's call; if there are genuinely none, say
"no reservations" and proceed.

## Pull requests

Open a PR against `main`; **Dominik reviews and merges**. An agent **asks first** (golden rule)
before pushing the branch.

- The `gh` CLI is **not installed** in the agent environment. An agent therefore **pushes the
  branch and relays the PR-creation URL**
  (`https://github.com/schienstockd/coastal/compare/main...<branch>?expand=1`, or the
  `pull/new/<branch>` link git prints) — it does not attempt `gh pr create`.
- **Always relay a complete, paste-ready PR body** — for every branch — inside a fenced
  ` ```markdown ` code block (so links survive copy-paste), ending with:

  ```
  🤖 Generated with [Claude Code](https://claude.com/claude-code)
  ```

```bash
git push -u origin <type>/<slug>
# relay the compare/PR URL git prints
```

## CI

Every push / PR runs [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) on Ubuntu:
`pip install -e .[dev]` (CPU-only torch, plus the OpenCV system libs via apt) → `pytest`. It
verifies the package imports and the `tests/` suite is green. Keep it green before requesting a
merge. CI deliberately tests the **plain-pip** install path (not pixi), so a fresh `pip install -e .`
stays viable for anyone without pixi; the local dev env uses pixi (below).

## Tests

One `pytest` suite under [`tests/`](../tests). Run it via the dev env — `pixi run test` — or, in a
plain-pip env, `pip install -e .[dev] && pytest`. Any change to core functionality ships with a test
in the same change; see [`tests/README.md`](../tests/README.md) for scope and conventions.

## Local dev environment

`pixi.toml` defines a self-contained env (Python 3.12 + coastal editable + editable cecelia +
Jupyter), independent of any external conda env: `pixi install`, then `pixi run kernel` / `lab` /
`test` / `doctor`. Details in [`docs/DATA.md`](DATA.md) → *Independent dev environment (pixi)*.
`pixi.lock` is committed; `.pixi/` is git-ignored.
