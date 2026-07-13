"""Tests for coastal.device.resolve_device — the cuda→mps→cpu runtime selection.

Covers the GPU rule (CLAUDE.md): entry points default device=None and must resolve to a real
device, and an explicit 'cuda' on a machine without CUDA must fall back rather than crash.
"""

import torch

from coastal.device import resolve_device


def test_resolve_device_auto_returns_valid_device():
    for dev in (None, 'auto'):
        assert resolve_device(dev) in ('cuda', 'mps', 'cpu')


def test_resolve_device_explicit_cpu_is_returned_as_is():
    assert resolve_device('cpu') == 'cpu'
    assert resolve_device('cuda:1') == 'cuda:1'


def test_resolve_device_auto_matches_cuda_availability():
    # On a CUDA box → 'cuda'; otherwise never 'cuda'.
    got = resolve_device(None)
    assert (got == 'cuda') == torch.cuda.is_available()


def test_resolve_device_cuda_falls_back_when_unavailable():
    if not torch.cuda.is_available():
        assert resolve_device('cuda') in ('mps', 'cpu')   # no crash, sensible fallback
    else:
        assert resolve_device('cuda') == 'cuda'
