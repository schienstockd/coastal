"""The tuning objective must not be improvable by finding FEWER cells.

`score_segmentation` used to return `n_good / n_large`. A ratio over a subset is
maximised by discarding cells: tighten region growing, large cells drop into the
(nearly free) fragment bin, and the surviving fraction looks purer. On a real movie that
let the score climb 0.429 -> 0.523 while the absolute number of good cells fell
140 -> 116, and CMA-ES pinned `affinity_threshold` to every upper bound it was given.

These tests pin the properties that make the reward form immune to that, using synthetic
frames where each degradation is applied deliberately. They are behavioural requirements
on the objective, not golden values.
"""

import numpy as np
import pytest

from coastal.optimize import score_segmentation

H = W = 60
CELL = 10                      # 10x10 = 100 px, exactly min_cell_size
MIN_CELL = 100


BACKGROUND = 500.0    # every channel, everywhere
CELL_EXCESS = 150.0   # one channel, inside a cell


def _frames(n_ch=3):
    """[1, C, H, W]: one dominant channel per cell, on top of a dominant background.

    The background:signal ratio is chosen to reproduce the real data. With BACKGROUND
    left in, a perfectly single-channel cell reads as purity ~0.39 — matching the 0.385
    median measured over 326 large cells in a real movie, and barely above the 1/3 floor.
    Subtract the background and the same cell reads ~1.0. Without a background-dominated
    frame these tests would not exercise the thing that was broken.
    """
    f = np.full((1, n_ch, H, W), BACKGROUND, dtype=np.float32)
    for i, (y, x) in enumerate([(5, 5), (5, 25), (25, 5), (25, 25)]):
        ch = i % n_ch
        f[0, ch, y:y + CELL, x:x + CELL] += CELL_EXCESS
    return f


def _labels(layout):
    """Build a [1, H, W]-style label map. layout: list of (y, x, h, w, label)."""
    inst = np.zeros((H, W), dtype=np.int32)
    for y, x, h, w, lab in layout:
        inst[y:y + h, x:x + w] = lab
    return [{'instances': inst}]


FOUR_CELLS = [(5, 5, CELL, CELL, 1), (5, 25, CELL, CELL, 2),
              (25, 5, CELL, CELL, 3), (25, 25, CELL, CELL, 4)]


def _score(results, **kw):
    # 0.7 is the library default and, with background subtraction, a clean cell reaches
    # ~1.0 while a two-colour blob reaches ~0.5 — so the threshold separates them.
    params = dict(min_cell_size=MIN_CELL, purity_threshold=0.7, junk_weight=0.05)
    params.update(kw)
    return score_segmentation(results, _frames(), **params)


def test_four_clean_cells_are_all_good():
    """Baseline: with background subtracted, single-channel cells read as pure."""
    s = _score(_labels(FOUR_CELLS))
    assert s == pytest.approx(4.0), 'expected 4 good, 0 junk'


def test_dropping_a_cell_lowers_the_score():
    """Detecting less must never help — the core failure of the old ratio."""
    full = _score(_labels(FOUR_CELLS))
    three = _score(_labels(FOUR_CELLS[:3]))
    assert three < full


def test_splitting_a_cell_into_fragments_lowers_the_score():
    """Oversegmentation: one good cell becomes 4 sub-min_cell_size pieces."""
    full = _score(_labels(FOUR_CELLS))
    h = CELL // 2
    split = FOUR_CELLS[1:] + [(5, 5, h, h, 11), (5, 5 + h, h, h, 12),
                              (5 + h, 5, h, h, 13), (5 + h, 5 + h, h, h, 14)]
    assert _score(_labels(split)) < full


def test_shrinking_cells_below_min_size_lowers_the_score():
    """The exact gaming move: stricter growing turns good cells into fragments."""
    full = _score(_labels(FOUR_CELLS))
    shrunk = [(y, x, CELL - 3, CELL - 3, lab) for y, x, _, _, lab in FOUR_CELLS]
    assert _score(_labels(shrunk)) < full, 'shrinking cells into fragments must not pay'


def test_merging_two_cells_into_an_impure_blob_lowers_the_score():
    """Over-merging: two differently-coloured cells become one impure label."""
    full = _score(_labels(FOUR_CELLS))
    merged = FOUR_CELLS[2:] + [(5, 5, CELL, CELL + 20, 1)]   # spans cells 1 and 2
    assert _score(_labels(merged)) < full


def test_adding_pure_junk_fragments_lowers_the_score():
    extra = FOUR_CELLS + [(45, 5 + 3 * i, 3, 3, 90 + i) for i in range(6)]
    assert _score(_labels(extra)) < _score(_labels(FOUR_CELLS))


def test_junk_weight_zero_ignores_fragments():
    """junk_weight is the documented recall/cleanliness dial."""
    extra = FOUR_CELLS + [(45, 5 + 3 * i, 3, 3, 90 + i) for i in range(6)]
    assert _score(_labels(extra), junk_weight=0.0) == _score(_labels(FOUR_CELLS),
                                                             junk_weight=0.0)
    assert _score(_labels(extra), junk_weight=0.5) < _score(_labels(FOUR_CELLS),
                                                            junk_weight=0.5)


def test_empty_segmentation_cannot_win():
    """The degenerate 'find nothing' strategy must not beat a real segmentation."""
    nothing = [{'instances': np.zeros((H, W), dtype=np.int32)}]
    assert _score(nothing) <= 0.0 < _score(_labels(FOUR_CELLS))


# --------------------------------------------------------------------------- #
# Background subtraction                                                       #
# --------------------------------------------------------------------------- #

def test_background_subtraction_is_what_makes_purity_usable():
    """Without it, purity collapses toward 1/C and the default threshold scores 0."""
    results = _labels(FOUR_CELLS)

    # legacy behaviour: background left in -> nothing reaches the documented 0.7 default
    legacy = score_segmentation(results, _frames(), min_cell_size=MIN_CELL,
                                purity_threshold=0.7, background_percentile=None,
                                junk_weight=0.0)
    assert legacy == 0.0, 'legacy metric cannot reach 0.7 even on perfectly pure cells'

    # with subtraction the same cells are correctly identified as pure
    fixed = score_segmentation(results, _frames(), min_cell_size=MIN_CELL,
                               purity_threshold=0.7, background_percentile=25,
                               junk_weight=0.0)
    assert fixed == pytest.approx(4.0)
