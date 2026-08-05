"""coastal.smooth — model-free smoothing for photon-limited fluorescence movies.

The counterpart to :mod:`coastal.denoise`, and deliberately a separate module: *that* one holds
**learned** restoration (the Cellpose-3 CPnet port) plus the ratio-preserving gain; this one holds
**smoothing** — a Gaussian in space and a running statistic in time, no model, no weights.

Naming matters here because three different things were converging on the word "denoise": a trained
net, a per-pixel gain, and a pair of local averages. Nothing in this module estimates or models
noise; it averages neighbours. So: smooth.

Why it exists
-------------
Resonance-scanner intravital data is **photon-limited**, not bit-depth-limited. Measured on
``zolIMa/fXgbTl`` (16-bit, 31x4x31x420x441, 15 s/frame): 86-95% of voxels per channel are ZERO, and
the observed maximum is **522 of 65535** — the data occupies 0.8% of its range. A 16-bit re-import
changed nothing, because there were never more than ~500 photons.

That breaks any statistic that assumes a channel has a *background population* to find. cecelia's
autofluorescence correction derives its background with a triangle threshold, and on this data the
threshold lands **inside the signal** — the reference channel kept **8.6%** of its signal past
background subtraction, so ~90% of it was discarded before the correction ran at all.

Smoothing repairs that by giving each voxel a local estimate instead of a photon count:

    channel        raw bg / signal kept        after smooth_channels
    nuc-GFP        40 / 12.6%                  7 / 85%
    mem-Tom        47 / 46.3%                  14 / 100%
    CD169-Kat      44 /  8.6%                  6 / 80%

Full measurement record, including four rejected alternatives:
``cecelia-feijoa/docs/todo/SMOOTHING_PLAN.md``.

The invariant
-------------
**One shared kernel, applied identically to every channel.** Not a convenience — the consumer is a
cross-channel ratio (the AF weight is ``b_t^p / sum_i b_i^p``; confetti identity is the channel
vector), and any *per-channel* transform corrupts it. This is what disqualified the learned net for
this job: its ``normalize99`` runs per plane per channel, so mem-Tom was divided by ~81 while
CD169-Kat was divided by ~35, and the difference was never undone.

Ordering is not optional either
-------------------------------
**Spatial first, then temporal.** A temporal statistic alone keeps **8.5%** of the reference
channel's signal — *worse than doing nothing* (15.4%) — because at single-digit photon counts a
median over 3 samples of mostly-zeros is zero. The Gaussian has to fill the sparse counts before a
temporal statistic has anything to work with.

Median or mean in time
----------------------
**Median, by default.** Both suppress noise; the median keeps masks tight. Measured at matched
foreground area (mem-Tom, one z-plane, objects = connected components >20 px):

    spatial only              area 139   objects 23
    + temporal mean 3         area 165   objects 21
    + temporal MEDIAN 3       area 140   objects 24

The mean averages a moving cell *in* at partial occupancy, inflating it ~34%; the median rejects it
as transient. Neither merges cells — verified by overlap test (1 true merge in 19, 0 objects lost),
not by object count, which cannot tell fusion from speck-cleanup.

**There is no spatial median**, on purpose. A median is robust to sparse outliers, and here the
signal *is* sparse positive counts, so it deletes it: a ``ball(1)`` spatial median left the
reference channel at **4.1%** signal kept, worse than raw, and ``ball(2)`` drove background
variance to exactly zero. (The R predecessor carried a ``medianFilter`` ball option; every
production run left it at 0.)

Lineage
-------
This supersedes the R implementation ``cleanupImages/slidingWindowCorrect.R`` +
``py/sliding_window_correct.py``, which took ``np.median`` over a T window per channel. Do not port
its window arithmetic: ``slice(i - sw, i + sw)`` is half-open, so the window was ``2*sw`` frames and
off-centre, and at its default ``sw=1`` the "median" of 2 samples degenerates to their mean.
``frames`` here is a full, centred, odd width.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter1d

__all__ = [
    "spatial_smooth",
    "temporal_smooth",
    "smooth_channels",
    "gaussian_restorer",
    "temporal_mean_restorer",
    "temporal_median_restorer",
]

#: Default xy Gaussian sigma, in pixels. Conservative against the ~15-20 px cells this was measured
#: on; it is the value the measurements above used, and it has NOT been swept.
DEFAULT_SIGMA = 1.0

#: Default temporal window (full width, centred, odd). 3 at 15 s/frame; 5 and 9 suppress marginally
#: more noise without improving masks.
DEFAULT_FRAMES = 3


def _axis_sigma(ndim, sigma, spatial_axes):
    """A per-axis sigma tuple that is `sigma` on the spatial axes and 0 everywhere else."""
    sig = [0.0] * ndim
    for ax in spatial_axes:
        sig[ax % ndim] = float(sigma)
    return tuple(sig)


def spatial_smooth(arr, sigma=DEFAULT_SIGMA, spatial_axes=(-2, -1)):
    """Gaussian blur over the spatial axes only, leaving every other axis untouched.

    ``spatial_axes`` defaults to the trailing two (Y, X). Returns float32. ``sigma <= 0`` is a
    no-op passthrough (as float32), so callers can disable the term without branching.
    """
    a = np.asarray(arr, dtype=np.float32)
    if sigma is None or sigma <= 0:
        return a
    return gaussian_filter(a, sigma=_axis_sigma(a.ndim, sigma, spatial_axes))


def temporal_smooth(arr, frames=DEFAULT_FRAMES, stat="median", time_axis=0):
    """Running statistic over ``time_axis``, with a centred window of ``frames``.

    ``frames`` is a FULL width and is forced odd (4 -> 5) so the window is centred on the frame it
    replaces — the R predecessor's off-by-one made this the one easy mistake here. ``frames <= 1``
    is a no-op passthrough, as is an axis of length 1 (a single timepoint, or a static image).

    ``stat='median'`` (default) rejects a cell that moved through the window; ``'mean'`` averages it
    in, which suppresses marginally more noise at the cost of ~34% mask inflation.
    """
    a = np.asarray(arr, dtype=np.float32)
    if frames is None or int(frames) <= 1:
        return a
    if a.shape[time_axis] <= 1:
        return a                            # nothing to average over
    n = int(frames)
    if n % 2 == 0:
        n += 1                              # centred windows are odd
    n = min(n, a.shape[time_axis] | 1)      # never wider than the axis (kept odd)
    if stat == "mean":
        return uniform_filter1d(a, size=n, axis=time_axis, mode="nearest")
    if stat != "median":
        raise ValueError(f"stat must be 'median' or 'mean', got {stat!r}")
    size = [1] * a.ndim
    size[time_axis] = n
    return median_filter(a, size=tuple(size), mode="nearest")


def smooth_channels(arr, sigma=DEFAULT_SIGMA, frames=DEFAULT_FRAMES, stat="median",
                    channel_axis=0, time_axis=None, channels=None, spatial_axes=(-2, -1)):
    """Smooth every channel with the SAME kernel: spatial Gaussian, then a temporal statistic.

    This is the entry point; the two halves are exposed separately for callers that need only one.

    Args:
        arr:          any shape with a channel axis, e.g. ``(C, Y, X)`` or ``(T, C, Z, Y, X)``.
        sigma:        xy Gaussian sigma in px; ``0`` disables the spatial term.
        frames:       centred odd temporal width; ``0``/``1`` disables the temporal term.
        stat:         ``'median'`` (default) or ``'mean'`` for the temporal statistic.
        channel_axis: which axis holds channels.
        time_axis:    which axis is time. ``None`` disables the temporal term — pass it explicitly
                      rather than letting this guess, because guessing wrong smooths across Z and
                      silently blurs a stack into a slab.
        channels:     which channels to smooth (default all); the rest pass through untouched.
        spatial_axes: the axes to blur, default trailing two.

    Returns float32 of the same shape. Absolute intensities are NOT preserved — this is a local
    average — so absolute-brightness measurements must read the unsmoothed store.

    ORDER IS LOAD-BEARING: spatial before temporal. Reversed, a temporal median on photon-limited
    input has only zeros to work with and keeps less signal than no smoothing at all.
    """
    a = np.asarray(arr, dtype=np.float32)
    ca = channel_axis % a.ndim
    sel = range(a.shape[ca]) if channels is None else list(channels)

    out = a.copy()
    for c in sel:
        idx = [slice(None)] * a.ndim
        idx[ca] = c
        idx = tuple(idx)
        # the per-channel view drops the channel axis, so the caller's spatial/time axes must be
        # re-resolved against the reduced shape
        sub = a[idx]
        sub_spatial = tuple(ax if ax < 0 else (ax - 1 if ax > ca else ax) for ax in spatial_axes)
        sub = spatial_smooth(sub, sigma, spatial_axes=sub_spatial)
        if time_axis is not None and frames and int(frames) > 1:
            ta = time_axis % a.ndim
            if ta == ca:
                raise ValueError("time_axis and channel_axis cannot be the same axis")
            sub = temporal_smooth(sub, frames, stat, time_axis=ta - 1 if ta > ca else ta)
        out[idx] = sub
    return out


# ── projection restorers, for `denoise.denoise_preserving_ratio` ──────────────────────────────
# Model-free, so they live here rather than beside the net. `denoise` re-exports them, so existing
# imports keep working.

def gaussian_restorer(sigma=1.0):
    """Spatial restorer: a plane-wise Gaussian. No weights, no download, no net. The default,
    because it is the only one valid for *any* input shape.

    Chosen on measurement. Ablating restorers inside the gain wrapper (4 movies x 2 z-planes,
    recall against raw-grey seeds): raw 84.5% recall / 596 blobs, ``sigma=1`` 87.5% / 158,
    ``denoise_cyto3`` 88.2% / 138. The net wins by 0.7pp — not enough to make a weights download
    the default path.
    """
    def restore(proj):
        return spatial_smooth(proj, sigma)
    return restore


def temporal_mean_restorer(window=3):
    """Temporal restorer: a running mean along axis 0. Requires the leading axis to be TIME.

    At matched foreground area (4 movies x 2 z-planes x 3 frames) it beats every spatial option:
    raw 83.9% recall at 3% area, ``gaussian_restorer(1)`` 89.3%, ``temporal_mean_restorer(3)``
    **92.3%**. Wrapped in the gain, identity stays at 99.5% against 97.6% for averaging the
    channels directly.

    The window is small on purpose: motion-compensating the frames first scores *worse* (87.4%),
    because the segmenter is flow-supervised and alignment deletes the motion it is conditioned on.
    See ``docs/todo/DENOISE_PLAN.md`` -> B2.
    """
    def restore(proj):
        if proj.ndim < 3:
            raise ValueError("temporal_mean_restorer needs a time axis; got a single plane")
        return temporal_smooth(proj, window, stat="mean", time_axis=0)
    return restore


def temporal_median_restorer(window=3):
    """Temporal restorer: a running MEDIAN along axis 0. Requires the leading axis to be TIME.

    The counterpart to :func:`temporal_mean_restorer`, and the better choice when mask geometry
    matters: on photon-limited input the mean averages a moving cell in at partial occupancy and
    inflates objects ~34% (area 165 vs 139 for spatial-only), while the median rejects it and holds
    area at 140 with a HIGHER object count (24 vs 21). Not yet measured inside the gain wrapper on
    confetti data — ``temporal_mean_restorer`` remains what is measured there.
    """
    def restore(proj):
        if proj.ndim < 3:
            raise ValueError("temporal_median_restorer needs a time axis; got a single plane")
        return temporal_smooth(proj, window, stat="median", time_axis=0)
    return restore
