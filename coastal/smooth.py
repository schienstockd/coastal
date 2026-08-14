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
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter, uniform_filter1d

__all__ = [
    "spatial_smooth",
    "temporal_smooth",
    "smooth_channels",
    "gaussian_restorer",
    "temporal_mean_restorer",
    "temporal_median_restorer",
    "temporal_gated",
    "gated_frame",
    "noise_sigma",
]

#: Default xy Gaussian sigma, in pixels. Conservative against the ~15-20 px cells this was measured
#: on; it is the value the measurements above used, and it has NOT been swept.
DEFAULT_SIGMA = 1.0

#: Default temporal window (full width, centred, odd). 3 at 15 s/frame; 5 and 9 suppress marginally
#: more noise without improving masks.
#:
#: That ceiling is a property of `stat='median'`, not of temporal smoothing: widening the median buys
#: noise reduction by destroying signal. Measured on a 30 s intravital movie (noise removed / punctum
#: amplitude kept / motion-sharpness ratio):
#:
#:     median(3) 32% / 0.92 / 0.96      gated(3) 25% / 1.00 / 1.00
#:     median(5) 44% / 0.85 / 0.91      gated(5) 34% / 1.00 / 1.00
#:     median(7) 49% / 0.77 / 0.84      gated(7) 38% / 1.00 / 1.01
#:     median(9) 53% / 0.69 / 0.77      gated(9) 41% / 1.00 / 1.01
#:
#: `stat='gated'` holds 1.00/1.00 at every width, so with it the window is worth raising.
DEFAULT_FRAMES = 3

#: Half-width of the block-match search, in px, for `stat='gated'`.
#:
#: 1 px is both the CHEAPEST and the BEST setting measured: the median inter-frame displacement over
#: signal is ~1 px at 30 s (Farneback, harness validated against known shifts) and ~half that at 15 s,
#: so +/-1 already covers the bulk of real motion, while a wider search costs quadratically and finds
#: slightly worse matches. Measured at frames=5: +/-1 36% noise removed at 0.12 s/plane, +/-2 35% at
#: 0.30, +/-3 34% at 0.55, +/-4 33% at 0.88. One default therefore serves both cadences.
DEFAULT_SEARCH = 1

#: Side of the square patch whose agreement decides a neighbour's weight.
#:
#: The weight MUST come from a patch, not a single pixel: a per-pixel difference is dominated by the
#: very noise being removed, so pixel-wise weights would be noise-driven. The patch mean averages the
#: noise out of the DECISION while leaving the ESTIMATE untouched.
DEFAULT_PATCH = 5


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
    in, which suppresses marginally more noise at the cost of ~34% mask inflation. ``'gated'``
    delegates to `temporal_gated`, which averages only what it can match and so preserves per-frame
    sharpness — note it gates on THIS array alone, so multi-channel callers must go through
    `smooth_channels` to get the shared gate the AF ratio requires.
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
    if stat == "gated":
        return temporal_gated(a, frames=n, time_axis=time_axis)
    if stat != "median":
        raise ValueError(f"stat must be 'median', 'mean' or 'gated', got {stat!r}")
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
        stat:         ``'median'`` (default), ``'mean'``, or ``'gated'`` for the temporal statistic.
                      ``'gated'`` preserves per-frame sharpness (see `temporal_gated`) and is gated
                      ONCE for all channels, from their sum — see the note below.
        channel_axis: which axis holds channels.
        time_axis:    which axis is time. ``None`` disables the temporal term — pass it explicitly
                      rather than letting this guess, because guessing wrong smooths across Z and
                      silently blurs a stack into a slab.
        channels:     which channels to smooth (default all); the rest pass through untouched.
        spatial_axes: the axes to blur, default trailing two.

    Returns float32 of the same shape. Absolute intensities are NOT preserved — this is a local
    average — so absolute-brightness measurements must read the unsmoothed store.

    ORDER IS LOAD-BEARING: spatial before temporal. Reversed, a temporal median on photon-limited
    input has only zeros to work with and keeps less signal than no smoothing at all. ``'gated'``
    keeps that order too: it matches on the spatially smoothed guide, which is also what makes the
    match robust at low photon counts.

    ONE SHARED KERNEL, INCLUDING THE ADAPTIVE ONE. Every channel is smoothed identically because the
    consumer is a cross-channel ratio (the AF weight is ``b_t^p / sum b_i^p``) and a per-channel
    transform corrupts it. ``'gated'`` is adaptive, so "identical" has to mean identical WEIGHTS: the
    match and the gate are derived once from the summed channels and applied to all of them. Gating
    each channel on its own content would decide differently at one voxel and break the ratio — and it
    also performs worse, because a dim channel then gates on its own noise instead of inheriting the
    match found in the total signal (measured: 36% vs 43% noise removed at equal sharpness).

    Measured ratio drift on a 4-channel 30 s movie — the shared gate disturbs the AF ratio LESS than
    the median it replaces, at matched noise reduction: median(5) 0.0290 at 44%, shared gate 0.0122 at
    43%. (A temporal median is nonlinear and per-channel, so each channel is pulled to a different
    timepoint's value; the "one shared kernel" guarantee only ever covered the linear terms.)
    """
    a = np.asarray(arr, dtype=np.float32)
    ca = channel_axis % a.ndim
    sel = range(a.shape[ca]) if channels is None else list(channels)

    if stat == "gated" and time_axis is not None and frames and int(frames) > 1:
        return _smooth_channels_gated(a, sigma, int(frames), ca, time_axis % a.ndim, list(sel),
                                      spatial_axes)

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


def noise_sigma(stack, time_axis=0):
    """Robust noise sigma from the temporal difference, for `stat='gated'`.

    MAD, not std, so the moving minority does not set the scale; `d = I_{t+1} - I_t` has variance
    2*sigma^2 wherever motion is absent.

    Public because a STREAMING caller must estimate this once over a representative slab and pass the
    same value to every `gated_frame` call — see that function on why a per-window estimate makes the
    gate drift across a movie.
    """
    d = np.diff(np.asarray(stack, dtype=np.float32), axis=time_axis)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def _gate_one(target, neighbours, g_target, g_neighbours, scale, search, patch):
    """Gated average of ONE frame against its neighbours. The core both forms share.

    `target`/`g_target` are (Y, X); `neighbours`/`g_neighbours` are sequences of (Y, X). The match and
    the weight are computed on the `g_*` guide and applied to the data — that separation is what lets
    several channels be gated identically (see `smooth_channels`).
    """
    acc = target.astype(np.float32).copy()             # the current frame always carries weight 1
    wsum = np.ones(target.shape, dtype=np.float32)
    for nb, g_nb in zip(neighbours, g_neighbours):
        best_d = np.full(target.shape, np.inf, dtype=np.float32)
        best_v = np.empty_like(best_d)
        for dy in range(-search, search + 1):
            for dx in range(-search, search + 1):
                d2 = uniform_filter((np.roll(np.roll(g_nb, dy, 0), dx, 1) - g_target) ** 2, patch)
                better = d2 < best_d
                if not better.any():
                    continue
                best_d = np.where(better, d2, best_d)
                best_v = np.where(better, np.roll(np.roll(nb, dy, 0), dx, 1), best_v)
        w = np.exp(-best_d / scale)
        acc += w * best_v
        wsum += w
    return acc / wsum


def _scale_from(guide, sigma, k):
    """Agreement scale: a patch mean of pure noise differences has expectation 2*sigma^2."""
    if sigma is None:
        sigma = noise_sigma(guide)
    return max(2.0 * (float(k) * float(sigma)) ** 2, 1e-12)


def gated_frame(window, guide=None, search=DEFAULT_SEARCH, patch=DEFAULT_PATCH, sigma=None, k=1.0):
    """Gated average of the CENTRE frame of a (W, Y, X) window — the STREAMING form.

    `temporal_gated` is the whole-series form. This exists because a caller that already holds a
    rolling window (cecelia's smoothing task streams one z-plane at a time) would otherwise have to
    call the series form and keep one frame of its output, doing W times the necessary work: each
    output frame needs its own matches, so computing W of them to use one is a factor of W wasted.

    A window with an even length has no centre; `W // 2` is used, matching the odd-width convention
    the rest of this module enforces.

    **Pass `sigma` when streaming.** Left as None it is estimated from what it was given, and a short
    window is a small sample — so each window would set its own gate strictness and the filter would
    behave slightly differently across a movie for no physical reason. The noise level is a property of
    the acquisition, so estimate it ONCE (`_noise_sigma` on a representative slab) and hand the same
    value to every call. With a shared sigma this and `temporal_gated` agree exactly, which is pinned
    by a test.
    """
    w = np.asarray(window, dtype=np.float32)
    if w.shape[0] <= 1:
        return w[0] if w.shape[0] else w
    g = w if guide is None else np.asarray(guide, dtype=np.float32)
    if g.shape != w.shape:
        raise ValueError(f"guide shape {g.shape} does not match window shape {w.shape}")
    c = w.shape[0] // 2
    idx = [i for i in range(w.shape[0]) if i != c]
    return _gate_one(w[c], [w[i] for i in idx], g[c], [g[i] for i in idx],
                     _scale_from(g, sigma, k), int(search), int(patch))


def _gated_plane_series(stack, guide, frames, search, patch, scale):
    """Agreement-gated average of one (T, Y, X) series, matched and weighted on `guide`."""
    T = stack.shape[0]
    half = frames // 2
    out = np.empty_like(stack)
    for t in range(T):
        nb = [min(max(t + dt, 0), T - 1) for dt in range(-half, half + 1) if dt != 0]
        nb = [n for n in nb if n != t]                # clamped at an edge: never average a frame with itself
        out[t] = _gate_one(stack[t], [stack[n] for n in nb], guide[t], [guide[n] for n in nb],
                           scale, search, patch)
    return out


def _smooth_channels_gated(a, sigma, frames, ca, ta, sel, spatial_axes,
                           search=DEFAULT_SEARCH, patch=DEFAULT_PATCH):
    """`smooth_channels` for ``stat='gated'``: spatial first, then ONE gate shared by every channel.

    Split out because the per-channel loop above cannot express a shared gate — the whole point is
    that the weights are computed once, across channels, and reused.
    """
    if ta == ca:
        raise ValueError("time_axis and channel_axis cannot be the same axis")

    def _sub_axes(axes):
        return tuple(ax if ax < 0 else (ax - 1 if ax > ca else ax) for ax in axes)

    def _idx(c):
        i = [slice(None)] * a.ndim
        i[ca] = c
        return tuple(i)

    sub_ta = ta - 1 if ta > ca else ta
    smoothed = {c: spatial_smooth(a[_idx(c)], sigma, spatial_axes=_sub_axes(spatial_axes))
                for c in sel}
    # the guide is the TOTAL signal of the smoothed channels: channel-agnostic by construction, and
    # the brightest thing available to match on, which is what a dim channel benefits from
    guide = None
    for v in smoothed.values():
        guide = v.copy() if guide is None else guide + v

    out = a.copy()
    for c, sub in smoothed.items():
        out[_idx(c)] = temporal_gated(sub, frames, time_axis=sub_ta, search=search, patch=patch,
                                      guide=guide)
    return out


def temporal_gated(arr, frames=DEFAULT_FRAMES, time_axis=0, search=DEFAULT_SEARCH,
                   patch=DEFAULT_PATCH, sigma=None, k=1.0, guide=None):
    """Motion-compensated, agreement-gated temporal averaging — the sharpness-preserving `stat`.

    For each neighbour frame, block-match a small window to find where this patch WENT, then weight
    that neighbour by how well the matched patch agrees. Static content matches, so it averages fully
    and noise falls ~1/sqrt(N). Content that arrived, left, or could not be tracked matches nowhere,
    the weights collapse, and the output IS the current frame.

    **Worst case is therefore the identity, never a blur** — the property a median cannot offer at any
    width, because it mixes whatever sits at the same pixel regardless of what it is. That is the whole
    point: temporal redundancy at intravital cadences is real but NOT co-located (~1 px median
    displacement at 30 s, tail to ~6 px), and a fixed window assumes it is.

    An established shape — non-local means / VBM3D with block matching — transplanted to intravital,
    not a new method.

    `guide` is the image the match and the weight are derived from; it defaults to `arr` itself. Pass a
    shared one to gate several channels identically — see `smooth_channels`, where that is required.

    `arr` may carry axes between time and the trailing (Y, X) — a Z stack is filtered plane by plane,
    never across Z.
    """
    a = np.asarray(arr, dtype=np.float32)
    if frames is None or int(frames) <= 1 or a.shape[time_axis] <= 1:
        return a
    n = int(frames)
    if n % 2 == 0:
        n += 1
    n = min(n, a.shape[time_axis] | 1)

    g = a if guide is None else np.asarray(guide, dtype=np.float32)
    if g.shape != a.shape:
        raise ValueError(f"guide shape {g.shape} does not match array shape {a.shape}")

    a = np.moveaxis(a, time_axis, 0)
    g = np.moveaxis(g, time_axis, 0)
    lead, mid, spatial = a.shape[0], a.shape[1:-2], a.shape[-2:]
    a2 = a.reshape((lead, -1) + spatial)
    g2 = g.reshape((lead, -1) + spatial)

    scale = _scale_from(g2, sigma, k)

    out = np.empty_like(a2)
    for i in range(a2.shape[1]):
        out[:, i] = _gated_plane_series(a2[:, i], g2[:, i], n, int(search), int(patch), scale)
    return np.moveaxis(out.reshape(a.shape), 0, time_axis)


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
