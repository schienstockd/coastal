"""napari viewers for the pipeline stages: raw images, segmentation, tracks.

The napari layer conventions live ONCE in cecelia (`cecelia.utils.napari_utils`) — coastal imports and
delegates to them, so coastal renders identically to cecelia's project viewer without duplicating the
convention logic (per-channel colormaps + additive blending, labels at ``opacity=0.7``, track
colour/tail params, contrast-from-sample). coastal already depends on cecelia's IO helpers and installs
cecelia editable in its env (see pixi.toml), so this is the same coupling, extended to display.

coastal never imports **napari** directly — it goes through cecelia too: ``napari_utils.new_viewer()``
creates the viewer and ``require_napari()`` (inside the helpers) raises a clear message if napari is
absent. cecelia + napari are imported LAZILY inside the functions, so ``import coastal`` stays
dependency-light (numpy only) — they're pulled only when you actually call ``show_*``.

What stays here is coastal-specific *orchestration*: reusing/creating the viewer, unpacking coastal's
array shapes (`_prep_image`), and converting coastal's µm track dict to napari's pixel matrix
(`tracks_to_matrix`). The generic ``add_image`` / ``add_labels`` / ``add_tracks`` calls are delegated.

Each ``show_*`` helper returns the napari ``Viewer`` (creating one if none is passed) so the stages can
be layered into a single viewer.
"""

import numpy as np

# Default per-channel colormaps (extend if a movie has > 4 channels). Mirrors
# cecelia.utils.napari_utils.CHANNEL_COLORMAPS — kept here as coastal's own choice of channel order.
CHANNEL_COLORMAPS = ['red', 'green', 'blue', 'yellow']


def _scale_tzyx(pix_res):
    """napari scale for a [T, Z, Y, X] array (time unscaled, Z/Y/X in µm)."""
    return (1.0, float(pix_res['z']), float(pix_res['y']), float(pix_res['x']))


def _prep_image(image, ch_indices):
    """Return (data, channel_axis) for napari from a [T,C,Z,Y,X] or [T,Z,Y,X] array."""
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
    from cecelia.utils import napari_utils
    v = viewer if viewer is not None else napari_utils.new_viewer()
    data, channel_axis = _prep_image(image, ch_indices)
    n_ch = data.shape[channel_axis] if channel_axis is not None else 1
    napari_utils.add_image(
        v, data, scale=_scale_tzyx(pix_res), channel_axis=channel_axis,
        channel_names=(channel_names if channel_axis is not None else (channel_names or 'image')),
        colormaps=(CHANNEL_COLORMAPS[:n_ch] or None) if channel_axis is not None else None,
    )
    return v


def show_segmentation(image, instances, pix_res, ch_indices=None, channel_names=None, viewer=None):
    """Overlay instance labels on the raw movie (image channels + a Labels layer at opacity 0.7).

    ``instances`` is a [T, Z, Y, X] integer label array (0 = background).
    """
    from cecelia.utils import napari_utils
    v = show_images(image, pix_res, ch_indices=ch_indices, channel_names=channel_names, viewer=viewer)
    napari_utils.add_labels(v, np.asarray(instances), scale=_scale_tzyx(pix_res), name='segmentation')
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
    from cecelia.utils import napari_utils
    v = viewer
    if image is not None:
        v = show_images(image, pix_res, ch_indices=ch_indices, viewer=v)
    if instances is not None:
        v = v if v is not None else napari_utils.new_viewer()
        napari_utils.add_labels(v, np.asarray(instances), scale=_scale_tzyx(pix_res), name='segmentation')
    if v is None:
        v = napari_utils.new_viewer()

    data = tracks_to_matrix(tracks, pix_res, min_track_len=min_track_len)
    # scale spans (t, z, y, x); vertices are in pixels so scale supplies the µm conversion.
    napari_utils.add_tracks(v, data, scale=_scale_tzyx(pix_res), name='tracks',
                            color_by='track_id', colormap='turbo', tail_width=4, tail_length=30)
    return v
