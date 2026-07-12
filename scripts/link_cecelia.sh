#!/usr/bin/env bash
# Editable-link cecelia's Python helper package into coastal's active env.
#
# NB: if you use the pixi env (pixi.toml — the recommended setup), cecelia is ALREADY linked
# editable by `pixi install` (it's a [pypi-dependencies] path dep), and stays linked across
# re-solves. This script is for the non-pixi path (a plain venv / uv), or to (re)link cecelia
# from a NON-sibling checkout via CECELIA_PYTHON. See docs/DATA.md.
#
# Why: coastal's notebooks import `cecelia.utils.*` (zarr/OME/dim helpers) and locate the vendored
# btrack config from the installed `cecelia` package. Installing cecelia EDITABLE means your edits
# to cecelia's Python (in cecelia-pineapple/python/) are picked up live — no reinstall per change.
# (A plain `pip install <wheel>` copies a frozen snapshot instead — stale after any cecelia edit.)
#
# Usage:
#   scripts/link_cecelia.sh                 # default: ../cecelia/cecelia-pineapple/python
#   CECELIA_PYTHON=/path/to/cecelia-pineapple/python scripts/link_cecelia.sh
#
# Run once; thereafter cecelia edits are live. Re-run only to refresh cecelia's own deps.
#
# Lifespan: this is a DEV-TIME BRIDGE while cecelia is unpublished. Once cecelia is on PyPI, drop
# this script and declare a normal pinned `cecelia>=<x.y>` dependency in pyproject.toml
# (see docs/TODO.md → Cecelia integration, and docs/DATA.md → Installing / keeping cecelia in sync).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cecelia_python="${CECELIA_PYTHON:-$here/../../cecelia/cecelia-pineapple/python}"

if [[ ! -f "$cecelia_python/pyproject.toml" ]]; then
  echo "error: no cecelia package at: $cecelia_python" >&2
  echo "       (is cecelia-pineapple checked out on a branch that has python/? — it landed on main" >&2
  echo "        in 2026-07). Set CECELIA_PYTHON to the right path and re-run." >&2
  exit 1
fi

python -m pip install -e "$cecelia_python"
echo "✓ cecelia linked editable from: $(cd "$cecelia_python" && pwd)"
echo "  cecelia Python edits are now live in this env — no reinstall needed per change."
