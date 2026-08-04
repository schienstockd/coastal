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


# --------------------------------------------------------------------------- #
# score_label_size_confetti: "largest reasonable label size, preserving confetti"  #
# --------------------------------------------------------------------------- #

from coastal.optimize import score_label_size_confetti

CAP = 400          # "reasonable" cell area for these synthetic cells


def _sized_frames(n_ch=3, regions=((5, 5, 20, 20, 0), (5, 35, 20, 20, 1))):
    """Background-dominated frame with one dominant channel per region."""
    f = np.full((1, n_ch, H, W), BACKGROUND, dtype=np.float32)
    for y, x, h, w, ch in regions:
        f[0, ch, y:y + h, x:x + w] += CELL_EXCESS
    return f


def _inst(layout):
    inst = np.zeros((H, W), dtype=np.int32)
    for y, x, h, w, lab in layout:
        inst[y:y + h, x:x + w] = lab
    return [{'instances': inst}]


def _size_score(layout, frames=None, **kw):
    params = dict(max_cell_size=CAP, purity_threshold=0.8)
    params.update(kw)
    return score_label_size_confetti(_inst(layout), frames if frames is not None
                                     else _sized_frames(), **params)


TWO_CELLS = [(5, 5, 20, 20, 1), (5, 35, 20, 20, 2)]     # 400 px each, one colour each


def test_splitting_a_pure_label_lowers_the_score():
    """The failure purity alone is blind to."""
    whole = _size_score(TWO_CELLS)
    split = _size_score([(5, 5, 20, 10, 1), (5, 15, 20, 10, 3), (5, 35, 20, 20, 2)])
    assert split < whole


def test_merging_same_colour_pieces_of_one_cell_raises_the_score():
    """Two fragments of one cell should prefer being one label."""
    split = _size_score([(5, 5, 20, 10, 1), (5, 15, 20, 10, 3), (5, 35, 20, 20, 2)])
    whole = _size_score(TWO_CELLS)
    assert whole > split


def test_merging_two_different_colour_cells_lowers_the_score():
    """The confetti constraint: one label spanning two colours is a merge error."""
    separate = _size_score(TWO_CELLS)
    merged = _size_score([(5, 5, 20, 50, 1)])       # spans both coloured regions
    assert merged < separate


def test_tiling_a_region_at_cell_size_beats_one_giant_label():
    """'Largest REASONABLE size' — the cap is what stops runaway merging.

    With only 3 confetti channels shared by ~270 cells each, a big label can be colour-pure
    by accident, so the size reward has to saturate. One label over a whole single-colour
    region must therefore lose to several cell-sized ones covering the same area.
    """
    region = ((5, 5, 20, 60, 0),)                 # 1200 px of one colour = 3x the cap
    frames = _sized_frames(regions=region)
    giant = _size_score([(5, 5, 20, 60, 1)], frames=frames)
    tiled = _size_score([(5, 5, 20, 20, 1), (5, 25, 20, 20, 2), (5, 45, 20, 20, 3)],
                        frames=frames)
    assert giant < tiled
    assert giant == pytest.approx(CAP / 1200, abs=0.05)   # saturated at one cap^2


def test_dropping_a_label_lowers_the_score():
    """Missing a cell must cost — the denominator is image-derived, not label-derived.

    Normalising by labelled area instead would make this pass at 1.0 either way, i.e. a
    quality ratio that rewards segmenting one cell and ignoring the rest.
    """
    assert _size_score(TWO_CELLS[:1]) < _size_score(TWO_CELLS)


def test_perfect_cell_sized_pure_labels_approach_one():
    """Normalisation sanity: all labelled area pure and exactly at the cap -> ~1.0."""
    assert _size_score(TWO_CELLS) == pytest.approx(1.0, abs=0.05)


def test_impure_labels_still_count_in_the_denominator():
    """An impure label must cost, not merely fail to contribute."""
    pure_only = _size_score(TWO_CELLS[:1], frames=_sized_frames(
        regions=((5, 5, 20, 20, 0),)))
    plus_impure = _size_score(TWO_CELLS[:1] + [(40, 5, 20, 50, 9)], frames=_sized_frames(
        regions=((5, 5, 20, 20, 0), (40, 5, 20, 25, 1), (40, 30, 20, 25, 2))))
    assert plus_impure < pure_only


# --------------------------------------------------------------------------- #
# Flow = separation: the merges identity cannot see                            #
# --------------------------------------------------------------------------- #

def _flow(regions):
    """[1, H, W, 2] dense flow; regions is (y, x, h, w, u, v)."""
    f = np.zeros((1, H, W, 2), dtype=np.float32)
    for y, x, h, w, u, v in regions:
        f[0, y:y + h, x:x + w, 0] = u
        f[0, y:y + h, x:x + w, 1] = v
    return f


def test_flow_catches_a_same_colour_merge_that_confetti_cannot():
    """Two touching cells of the SAME colour moving apart.

    With ~270 cells per confetti channel this is common, and colour purity is blind to it:
    the merged label is perfectly single-coloured. Flow separates them because a motion
    boundary runs through the label.
    """
    # one 20x40 patch of a single colour = two adjacent same-colour cells
    frames = _sized_frames(regions=((5, 5, 20, 40, 0),))
    # left half moves left, right half moves right
    flows = _flow(((5, 5, 20, 20, -3.0, 0.0), (5, 25, 20, 20, 3.0, 0.0)))

    merged = [(5, 5, 20, 40, 1)]                      # one label over both
    split = [(5, 5, 20, 20, 1), (5, 25, 20, 20, 2)]   # correctly separated

    # Confetti alone cannot tell these apart — the merged label is colour-pure.
    assert _size_score(merged, frames=frames) > 0
    conf_only_merged = _size_score(merged, frames=frames)
    conf_only_split = _size_score(split, frames=frames)

    # With flow, the merged label is rejected as spanning a motion boundary...
    with_flow_merged = _size_score(merged, frames=frames, flows=flows)
    with_flow_split = _size_score(split, frames=frames, flows=flows)

    assert with_flow_merged < conf_only_merged, 'flow must penalise the same-colour merge'
    assert with_flow_split > with_flow_merged, 'the correct split must win once flow is used'
    # and the correct split is unaffected by adding the flow constraint
    assert with_flow_split == pytest.approx(conf_only_split)


def test_coherent_motion_does_not_penalise_a_real_cell():
    """A cell whose pixels move together must still count."""
    frames = _sized_frames(regions=((5, 5, 20, 20, 0),))
    flows = _flow(((5, 5, 20, 20, 2.0, 1.0),))
    one = [(5, 5, 20, 20, 1)]
    assert _size_score(one, frames=frames, flows=flows) == pytest.approx(
        _size_score(one, frames=frames))


def test_flow_constraint_is_opt_in():
    """flows=None leaves behaviour exactly as before."""
    frames = _sized_frames()
    assert _size_score(TWO_CELLS, frames=frames, flows=None) == _size_score(
        TWO_CELLS, frames=frames)
