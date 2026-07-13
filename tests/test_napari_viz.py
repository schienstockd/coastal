"""Tests for the napari-free parts of coastal.napari_viz.

The viewer helpers need napari + a display, so they aren't exercised here; but the pure-numpy
track-matrix conversion (µm → pixel, the [track_id, t, z, y, x] layout napari wants) is, and the
module must import without napari installed (lazy import).
"""

import os

import numpy as np

from coastal.napari_viz import tracks_to_matrix, _scale_tzyx, _prep_image, _fix_qt_plugin_path


def test_fix_qt_plugin_path_drops_cv2_path_keeps_others():
    prev = os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH')
    try:
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/x/site-packages/cv2/qt/plugins'
        _fix_qt_plugin_path()
        assert 'QT_QPA_PLATFORM_PLUGIN_PATH' not in os.environ   # cv2 path removed → PyQt5 plugins win

        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/opt/qt/plugins'
        _fix_qt_plugin_path()
        assert os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] == '/opt/qt/plugins'  # legit path untouched
    finally:
        os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
        if prev is not None:
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = prev


class _NoMaterialise:
    """Array-like that refuses to become a numpy array — proves _prep_image never forces a load.

    A confetti movie is multiple GB; _prep_image must pass a lazy (dask) array straight through to
    napari, which slices on demand. If _prep_image ever calls np.asarray() again, __array__ fires.
    """

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)

    def __getitem__(self, idx):
        # emulate lazy channel slicing vol[:, [..]] -> narrower channel axis, still lazy
        ch = idx[1]
        new = list(self.shape)
        new[1] = len(ch)
        return _NoMaterialise(new)

    def __array__(self, *a, **k):
        raise AssertionError("_prep_image materialised a lazy array (called np.asarray)")


def test_prep_image_keeps_lazy_arrays_lazy():
    data, channel_axis = _prep_image(_NoMaterialise((180, 4, 12, 540, 516)), ch_indices=[1, 2, 3])
    assert channel_axis == 1
    assert data.shape[1] == 3                       # channel selection applied, still lazy


def test_prep_image_channel_axis_and_slice_numpy():
    data, channel_axis = _prep_image(np.zeros((2, 4, 3, 6, 6)), ch_indices=[1, 3])
    assert channel_axis == 1 and data.shape[1] == 2
    data, channel_axis = _prep_image(np.zeros((2, 3, 6, 6)), ch_indices=None)
    assert channel_axis is None and data.shape == (2, 3, 6, 6)


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
