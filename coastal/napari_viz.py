"""napari viewers for the pipeline stages: raw images, segmentation, tracks.

napari is an **optional, lazily-imported** dependency (install ``coastal[napari]`` or use the pixi
env) — this module is never imported at package import time, and calling a helper without napari
raises a clear message. Conventions mirror cecelia's napari bridge so coastal overlays look and
align the same as cecelia's project viewer:

  - anisotropic ``scale`` from ``pix_res`` (so Z / XY are physically correct),
  - one image layer per channel with per-channel colormaps + additive blending,
  - ``labels`` at ``opacity=0.7``,
  - tracks as an ``[track_id, t, z, y, x]`` matrix in **pixel** coords, with ``scale`` supplying the
    µm conversion (coastal tracks are in µm, so vertices = ``pos_um / [z, y, x]``),
  - ``color_by='track_id'`` with the ``turbo`` colormap.

Each ``show_*`` helper returns the napari ``Viewer`` (creating one if none is passed) so the stages
can be layered into a single viewer.
"""

import numpy as np

# Default confetti-ish per-channel colormaps (extend if a movie has >4 channels).
CHANNEL_COLORMAPS = ['red', 'green', 'blue', 'yellow']


def _require_napari():
    try:
        import napari
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "napari is required for coastal.napari_viz — install `coastal[napari]` or use the "
            "pixi env (it ships napari + pyqt)."
        ) from e
    return napari


def _scale_tzyx(pix_res):
    """napari scale for a [T, Z, Y, X] array (time unscaled, Z/Y/X in µm)."""
    return (1.0, float(pix_res['z']), float(pix_res['y']), float(pix_res['x']))


def _prep_image(image, ch_indices):
    """Return (data, channel_axis) for napari.add_image from a [T,C,Z,Y,X] or [T,Z,Y,X] array."""
    vol = np.asarray(image)
    if vol.ndim == 5:                       # [T, C, Z, Y, X]
        if ch_indices is not None:
            vol = vol[:, list(ch_indices)]
        return vol, 1
    if vol.ndim == 4:                       # [T, Z, Y, X] — single channel
        return vol, None
    raise ValueError(f"expected a [T,C,Z,Y,X] or [T,Z,Y,X] image, got shape {vol.shape}")


def show_images(image, pix_res, ch_indices=None, channel_names=None, viewer=None):
    """Show a raw movie — one layer per channel, anisotropic scale, additive blending.

    Args:
        image:        [T, C, Z, Y, X] (multi-channel) or [T, Z, Y, X] (single) array.
        pix_res:      {'z','y','x'} µm/pixel.
        ch_indices:   channels to display (multi-channel input); None = all.
        channel_names: optional per-layer names.
        viewer:       reuse an existing napari.Viewer; None creates one.

    Returns the napari.Viewer.
    """
    napari = _require_napari()
    v = viewer if viewer is not None else napari.Viewer()
    data, channel_axis = _prep_image(image, ch_indices)
    scale = _scale_tzyx(pix_res)
    if channel_axis is not None:
        n_ch = data.shape[channel_axis]
        v.add_image(data, channel_axis=channel_axis, scale=scale,
                    name=channel_names, colormap=CHANNEL_COLORMAPS[:n_ch] or None,
                    blending='additive')
    else:
        v.add_image(data, scale=scale, name=(channel_names or 'image'))
    return v


def show_segmentation(image, instances, pix_res, ch_indices=None, channel_names=None, viewer=None):
    """Overlay instance labels on the raw movie (image channels + a Labels layer at opacity 0.7).

    ``instances`` is a [T, Z, Y, X] integer label array (0 = background).
    """
    napari = _require_napari()
    v = show_images(image, pix_res, ch_indices=ch_indices, channel_names=channel_names, viewer=viewer)
    v.add_labels(np.asarray(instances), name='segmentation', scale=_scale_tzyx(pix_res), opacity=0.7)
    return v


def tracks_to_matrix(tracks, pix_res, min_track_len=1):
    """coastal tracks ``{track_id: {t: (z, y, x) µm}}`` → napari ``[track_id, t, z, y, x]`` matrix.

    Vertices are converted µm → pixels (``pos_um / [z, y, x]``) so the matrix pairs with a napari
    ``scale`` of ``(1, z, y, x)`` — matching the image/labels layers.
    """
    zyx = np.array([pix_res['z'], pix_res['y'], pix_res['x']], dtype=np.float64)
    rows = []
    for tid, tpoints in tracks.items():
        if len(tpoints) < min_track_len:
            continue
        for t, pos_um in sorted(tpoints.items()):
            z, y, x = np.asarray(pos_um, dtype=np.float64) / zyx
            rows.append([int(tid), int(t), z, y, x])
    return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)


def show_tracks(tracks, pix_res, instances=None, image=None, ch_indices=None,
                min_track_len=1, viewer=None):
    """Show tracks (optionally over the raw image and/or its segmentation).

    Args:
        tracks:        {track_id: {t: (z, y, x) µm}} from track_sequence.
        pix_res:       {'z','y','x'} µm/pixel.
        instances:     optional [T, Z, Y, X] labels to underlay.
        image:         optional [T,C,Z,Y,X] / [T,Z,Y,X] raw movie to underlay.
        ch_indices:    channels for the image underlay.
        min_track_len: drop tracks shorter than this many timepoints.
        viewer:        reuse an existing napari.Viewer.

    Returns the napari.Viewer.
    """
    napari = _require_napari()
    v = viewer
    if image is not None:
        v = show_images(image, pix_res, ch_indices=ch_indices, viewer=v)
    if instances is not None:
        v = v if v is not None else napari.Viewer()
        v.add_labels(np.asarray(instances), name='segmentation', scale=_scale_tzyx(pix_res), opacity=0.7)
    if v is None:
        v = napari.Viewer()

    data = tracks_to_matrix(tracks, pix_res, min_track_len=min_track_len)
    # scale spans (t, z, y, x); vertices are in pixels so scale supplies the µm conversion.
    v.add_tracks(data, name='tracks', scale=_scale_tzyx(pix_res),
                 color_by='track_id', colormap='turbo', tail_width=4, tail_length=30,
                 blending='additive')
    return v
