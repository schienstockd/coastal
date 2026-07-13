"""Runtime torch device selection.

Per the project GPU rule (CLAUDE.md): pick ``cuda`` → ``mps`` → ``cpu`` at runtime; never
hard-code ``cuda``. All entry points default ``device=None`` and route it through
``resolve_device`` so a CPU/MPS-only machine works out of the box.
"""

import torch


def resolve_device(device=None):
    """Resolve a concrete torch device string.

    - ``None`` or ``'auto'`` → auto-select ``cuda`` if available, else ``mps``, else ``cpu``.
    - An explicit ``'cuda'`` that is **not** available falls back to ``mps``/``cpu`` (no crash).
    - Any other explicit value (``'cpu'``, ``'mps'``, ``'cuda:1'``, …) is returned unchanged.
    """
    def _auto():
        if torch.cuda.is_available():
            return 'cuda'
        mps = getattr(torch.backends, 'mps', None)
        if mps is not None and mps.is_available():
            return 'mps'
        return 'cpu'

    if device in (None, 'auto'):
        return _auto()
    if device == 'cuda' and not torch.cuda.is_available():
        return _auto()
    return device
