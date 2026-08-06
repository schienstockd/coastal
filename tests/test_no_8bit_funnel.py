"""Nothing in coastal may funnel image data through 8 bits.

History, because it explains why this is a test and not a preference. Images were imported as
8-bit on the assumption that it made them meaningfully smaller; changing the zarr compressor
showed it did not, and 16-bit is now the norm. The 8-bit machinery outlived its reason and cost
real money in the meantime:

  * ``np.array(frames, dtype=np.uint8)`` in three flow entry points **wrapped** rather than
    clipping, so on 16-bit stores (channel maxima 176-1302) 305 became 49 -- the *brightest*
    voxels came out dark, which is what Farneback then tracked. Recorded as trap 3 in cecelia's
    ``docs/todo/SEGMENTATION_OPEN_PROBLEM.md``.
  * ``abm.py`` rescaled **each frame by its own min/max**, which breaks the brightness-constancy
    assumption Farneback rests on: identical structure at a different frame-wise range reads as
    motion.

Both are gone, and the reason neither needed a clever fix is that **OpenCV's Farneback accepts
float32 directly** -- measured against known shifts on real 16-bit intravital data. The 8-bit step
was a lossy round-trip that bought nothing.
"""
import ast
import pathlib

import numpy as np

from coastal.flow import (calc_flow_farneback_between_frames,
                          compute_multi_scale_optical_flow,
                          normalize_and_project)

PKG = pathlib.Path(__file__).resolve().parent.parent / "coastal"


def _ramp16(t=3, n=64):
    """16-bit frames with values far above 255, i.e. the case that used to wrap."""
    rng = np.random.default_rng(0)
    f = np.zeros((t, n, n), dtype=np.uint16)
    for i in range(t):
        f[i, 8:24, 8:24] = 40           # dim
        f[i, 30:46, 30:46] = 305        # a bare uint8 cast sends this to 49
        f[i, 48:60, 48:60] = 1302       # ...and this to 22
        f[i] += rng.integers(0, 5, size=(n, n)).astype(np.uint16)
    return f


def test_farneback_takes_float32_so_no_8bit_step_is_needed():
    a = _ramp16()[0].astype(np.float32)
    b = np.roll(a, 2, axis=1)
    u, v = calc_flow_farneback_between_frames(a, b)
    assert np.isfinite(u).all() and np.isfinite(v).all()


def test_16bit_ordering_survives_the_flow_entry_points():
    """The brightest region must stay brightest all the way in — the wrap inverted it."""
    f = _ramp16()
    flows = compute_multi_scale_optical_flow(f, scales=[1], verbose=False)
    assert len(flows[1]) == 2
    assert all(np.isfinite(fl["u"]).all() for fl in flows[1])


def test_normalize_and_project_is_not_quantised():
    seq = (_ramp16()[:, None] * 1).astype(np.uint16)          # [T, C=1, H, W]
    multi, proj = normalize_and_project(seq)
    assert multi.dtype == np.float32 and proj.dtype == np.float32
    assert multi.max() <= 255.0 + 1e-3, "the 0-255 range is a downstream contract"
    # quantised output would have <=256 distinct values; float32 keeps the gradations
    assert len(np.unique(multi)) > 3


def test_no_8bit_casts_of_image_data_anywhere_in_the_package():
    """Detector. AST, not text, so the prose above does not trip it.

    Label/mask casts are legitimate (a mask IS 0/1) and are excluded by name.
    """
    def _is_uint8(node):
        return (isinstance(node, ast.Attribute) and node.attr == "uint8"
                and isinstance(node.value, ast.Name) and node.value.id == "np")

    IMAGEY = ("frame", "vol", "img", "im", "seq", "arr", "data", "stack")

    def offending(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            # np.array(x, dtype=np.uint8) / np.zeros(..., dtype=np.uint8)
            if any(k.arg == "dtype" and _is_uint8(k.value) for k in node.keywords):
                names = [a.id.lower() for a in node.args if isinstance(a, ast.Name)]
                return any(any(w in n for w in IMAGEY) for n in names)
            # x.astype(np.uint8)
            if node.func.attr == "astype" and node.args and _is_uint8(node.args[0]):
                tgt = node.func.value
                if isinstance(tgt, ast.Name):
                    return any(w in tgt.id.lower() for w in IMAGEY)
        return False

    offenders = [
        f"{p.name}:{n.lineno}"
        for p in sorted(PKG.glob("*.py"))
        if p.name not in ("viz.py", "napari_viz.py", "morphology.py")   # rendering + mask code
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8")))
        if offending(n)
    ]
    assert not offenders, (
        f"8-bit cast of image data at {offenders}. Farneback takes float32 directly and images are "
        "16-bit — do not funnel them through uint8."
    )
