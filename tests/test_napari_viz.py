"""Tests for the napari-free parts of coastal.napari_viz.

The viewer helpers need napari + a display, so they aren't exercised here; but the pure-numpy
track-matrix conversion (µm → pixel, the [track_id, t, z, y, x] layout napari wants) is, and the
module must import without napari installed (lazy import).
"""

import numpy as np

from coastal.napari_viz import tracks_to_matrix, _scale_tzyx


def test_scale_tzyx_puts_zyx_in_order_with_unit_time():
    assert _scale_tzyx({'z': 4.0, 'y': 0.48, 'x': 0.48}) == (1.0, 4.0, 0.48, 0.48)


def test_tracks_to_matrix_converts_um_to_pixels_and_layout():
    pix_res = {'z': 4.0, 'y': 0.5, 'x': 0.5}
    # track 7: t=0 at (z=8µm, y=10µm, x=20µm), t=1 at (z=4, y=5, x=5)
    tracks = {7: {0: (8.0, 10.0, 20.0), 1: (4.0, 5.0, 5.0)}}
    m = tracks_to_matrix(tracks, pix_res)
    assert m.shape == (2, 5)                       # [track_id, t, z, y, x]
    assert m[0].tolist() == [7, 0, 2.0, 20.0, 40.0]   # 8/4, 10/0.5, 20/0.5
    assert m[1].tolist() == [7, 1, 1.0, 10.0, 10.0]


def test_tracks_to_matrix_min_len_filter_and_empty():
    pix_res = {'z': 1.0, 'y': 1.0, 'x': 1.0}
    tracks = {1: {0: (0, 0, 0)}, 2: {0: (1, 1, 1), 1: (2, 2, 2)}}
    m = tracks_to_matrix(tracks, pix_res, min_track_len=2)
    assert set(m[:, 0].astype(int)) == {2}          # track 1 (len 1) dropped
    assert tracks_to_matrix({}, pix_res).shape == (0, 5)
