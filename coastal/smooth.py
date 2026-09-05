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

import cv2
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
    "gated_frames",
    "noise_sigma",
    "temporal_flow_warped",
    "flow_warped_frame",
    "flow_warped_frames",
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

#: Farneback averaging window for the flow-warped fusion engine. Larger = smoother flow, softer
#: fine detail. cecelia's flowRegister task exposes this via an `aggressiveness` select (gentle/
#: balanced/strong → 11/17/25) — the same three-tier ladder here would let a task expose the same
#: knob.
DEFAULT_FLOW_WINSIZE = 17

#: Farneback pyramid levels for the flow-warped fusion engine — more levels handle larger
#: displacements. 5 covers typical intravital drift.
DEFAULT_FLOW_PYR_LEVELS = 5

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
    sharpness. ``'farneback'`` delegates to `temporal_flow_warped`, which warps each neighbour onto
    the centre via dense optical flow and averages — same intent as gated (motion-compensated
    averaging so wider windows help), different mechanism (continuous flow vs. block match). Both
    gate on THIS array alone here, so multi-channel callers must go through `smooth_channels` to
    get the shared kernel the AF ratio requires.
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
    if stat == "farneback":
        return temporal_flow_warped(a, frames=n, time_axis=time_axis)
    if stat != "median":
        raise ValueError(
            f"stat must be 'median', 'mean', 'gated' or 'farneback', got {stat!r}")
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

    if stat == "farneback" and time_axis is not None and frames and int(frames) > 1:
        return _smooth_channels_flow_warped(a, sigma, int(frames), ca, time_axis % a.ndim,
                                            list(sel), spatial_axes)

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


def _offsets(search):
    return [(dy, dx) for dy in range(-search, search + 1) for dx in range(-search, search + 1)]


def _match(g_target, g_neighbours, scale, search, patch):
    """The gate itself: per neighbour, where each patch WENT and how much to trust it.

    Computed from the guide ALONE, and returned rather than applied, because it does not depend on the
    channel — several channels share one gate, so the match must be computed once and reused. Doing it
    per channel is the same answer C times over, and the block match is the expensive part (a
    `uniform_filter` per candidate offset, against a cheap `np.roll` to apply one).

    Returns a list of `(weight, take)` pairs, one per neighbour. `take` is a FLAT index array: the
    matched value for a neighbour is `nb.ravel()[take]`, i.e. one gather. Resolving the winning offset
    to an index here rather than in the apply step matters — the alternative is re-rolling the whole
    plane once per candidate offset for every channel, which is the same work again per channel.
    """
    H, W = g_target.shape
    yy, xx = np.indices((H, W), dtype=np.int32)   # int32: 4 GB of pixels per plane before it overflows,
                                                 # and halves the index array a big plane carries
    out = []
    for g_nb in g_neighbours:
        best_d = np.full(g_target.shape, np.inf, dtype=np.float32)
        take = np.empty((H, W), dtype=np.int32)
        for dy, dx in _offsets(search):
            d2 = uniform_filter((np.roll(np.roll(g_nb, dy, 0), dx, 1) - g_target) ** 2, patch)
            better = d2 < best_d
            if not better.any():
                continue
            best_d = np.where(better, d2, best_d)
            # np.roll(nb, dy, 0)[y, x] == nb[(y - dy) % H, (x - dx) % W]
            idx = ((yy - dy) % H) * W + ((xx - dx) % W)
            take = np.where(better, idx, take)
        out.append((np.exp(-best_d / scale), take))
    return out


def _apply_gate(target, neighbours, gate):
    """Average `target` with its neighbours under a precomputed gate (see `_match`).

    One gather per neighbour — the gate already resolved WHERE each pixel's match is.
    """
    acc = target.astype(np.float32).copy()              # the current frame always carries weight 1
    wsum = np.ones(target.shape, dtype=np.float32)
    for nb, (w, take) in zip(neighbours, gate):
        acc += w * nb.reshape(-1)[take].reshape(acc.shape)
        wsum += w
    return acc / wsum


def _gate_one(target, neighbours, g_target, g_neighbours, scale, search, patch):
    """One frame, one channel — match then apply. Kept as the single-channel spelling."""
    return _apply_gate(target, neighbours,
                       _match(g_target, g_neighbours, scale, search, patch))


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
    return gated_frames([w], guide=g, search=search, patch=patch, sigma=sigma, k=k)[0]


def gated_frames(windows, guide, search=DEFAULT_SEARCH, patch=DEFAULT_PATCH, sigma=None, k=1.0):
    """`gated_frame` for SEVERAL channels that share one gate — the form multi-channel callers want.

    The match and the weights come from `guide` alone, so they are identical for every channel. Calling
    `gated_frame` per channel recomputes them C times: the block match is a `uniform_filter` per
    candidate offset, while applying a known match is an `np.roll`, so the redundant work dominates.
    Measured on a 4-channel plane this is the difference between 4x and 1x the matching cost.

    `windows` is a sequence of (W, Y, X) windows, all co-registered with `guide`. Returns the filtered
    CENTRE frame of each, in order.
    """
    g = np.asarray(guide, dtype=np.float32)
    ws = [np.asarray(w, dtype=np.float32) for w in windows]
    for w in ws:
        if w.shape != g.shape:
            raise ValueError(f"window shape {w.shape} does not match guide shape {g.shape}")
    if g.shape[0] <= 1:
        return [w[0] for w in ws]
    c = g.shape[0] // 2
    idx = [i for i in range(g.shape[0]) if i != c]
    gate = _match(g[c], [g[i] for i in idx], _scale_from(g, sigma, k), int(search), int(patch))
    return [_apply_gate(w[c], [w[i] for i in idx], gate) for w in ws]


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


def _smooth_channels_flow_warped(a, sigma, frames, ca, ta, sel, spatial_axes,
                                 winsize=DEFAULT_FLOW_WINSIZE,
                                 pyr_levels=DEFAULT_FLOW_PYR_LEVELS,
                                 max_shift_px=None):
    """`smooth_channels` for ``stat='farneback'``: spatial first, then ONE flow shared by every channel.

    Mirrors `_smooth_channels_gated`. The flow is derived from the summed smoothed channels, so it is
    channel-agnostic by construction and — critically — a dim channel inherits the warp found in the
    bright signal instead of tracking its own noise. Same invariant `_smooth_channels_gated` enforces
    for the block-match kernel, extended to the continuous one.
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
    guide = None
    for v in smoothed.values():
        guide = v.copy() if guide is None else guide + v

    out = a.copy()
    for c, sub in smoothed.items():
        out[_idx(c)] = temporal_flow_warped(sub, frames, time_axis=sub_ta,
                                            winsize=winsize, pyr_levels=pyr_levels,
                                            max_shift_px=max_shift_px, guide=guide)
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


# ── flow-warped temporal fusion, `stat='farneback'` ────────────────────────────────────────────
# The counterpart to `gated`: instead of block-matching where a patch went and gating its weight,
# compute dense optical flow from the centre frame to each neighbour, warp the neighbour BACK onto
# the centre, then average. Same intent — motion-compensated averaging so widening the window buys
# noise reduction without blurring moving cells — different mechanism.
#
# When to pick which: gated has a clean identity floor (unmatched patches collapse the weight and
# the output IS the input), so it never hurts. Flow-warped can hurt where the flow is unreliable
# — very sparse or very noisy planes — because a bad warp is still applied. The `max_shift_px`
# clamp is the safety net (per-pixel fall-back to source above the clamp), NOT a substitute for
# gating. Choose flow-warped when smooth continuous deformation dominates over discrete cell
# motion — the same regime `warp_to_reference` handles inside a whole movie.


def _farneback_flow(ref, mov, winsize, pyr_levels, poly_n=5, poly_sigma=1.2):
    """Dense Farneback flow from `ref` to `mov`. Returns (H, W, 2), (…, 0)=x, (…, 1)=y.

    Semantics of the returned flow (OpenCV convention): mov(x + flow.x, y + flow.y) ≈ ref(x, y),
    so `cv2.remap` with `map = pixel_grid + flow` sends `mov` back onto `ref`'s coordinates. That is
    exactly what the fusion engine wants — a per-frame warp that puts each neighbour on the centre.
    """
    return cv2.calcOpticalFlowFarneback(
        np.asarray(ref, dtype=np.float32),
        np.asarray(mov, dtype=np.float32),
        None, 0.5, int(pyr_levels), int(winsize),
        3, int(poly_n), float(poly_sigma), cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )


def _warp_by_flow(frame, flow, max_shift_px=None):
    """Warp `frame` (H, W) by `flow` (H, W, 2). Pixels above `max_shift_px` fall back to source —
    the same guard cecelia's flow_register uses, so a wild flow degrades to the identity per-pixel
    rather than smearing the plane."""
    H, W = frame.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = xx + flow[..., 0]
    map_y = yy + flow[..., 1]
    src = np.asarray(frame, dtype=np.float32)
    warped = cv2.remap(src, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    if max_shift_px is not None:
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        over = mag > float(max_shift_px)
        if over.any():
            warped = np.where(over, src, warped)
    return warped


def flow_warped_frame(window, guide=None, winsize=DEFAULT_FLOW_WINSIZE,
                      pyr_levels=DEFAULT_FLOW_PYR_LEVELS, poly_n=5, poly_sigma=1.2,
                      max_shift_px=None):
    """Flow-warped average of the CENTRE frame of a (W, Y, X) window — the STREAMING form.

    For each non-centre frame, compute Farneback flow from the centre and warp the neighbour back
    onto the centre's coordinates, then average all warped neighbours with the centre (which
    carries weight 1). `guide` defaults to `window` and drives the flow when several channels
    share one warp — see `flow_warped_frames`.

    `temporal_flow_warped` is the whole-series form. This exists for the same reason `gated_frame`
    does: a caller that already holds a rolling window would otherwise call the series form and keep
    one frame of its output, doing W times the necessary work.
    """
    w = np.asarray(window, dtype=np.float32)
    if w.shape[0] <= 1:
        return w[0] if w.shape[0] else w
    g = w if guide is None else np.asarray(guide, dtype=np.float32)
    if g.shape != w.shape:
        raise ValueError(f"guide shape {g.shape} does not match window shape {w.shape}")
    return flow_warped_frames([w], guide=g, winsize=winsize, pyr_levels=pyr_levels,
                              poly_n=poly_n, poly_sigma=poly_sigma,
                              max_shift_px=max_shift_px)[0]


def flow_warped_frames(windows, guide, winsize=DEFAULT_FLOW_WINSIZE,
                       pyr_levels=DEFAULT_FLOW_PYR_LEVELS, poly_n=5, poly_sigma=1.2,
                       max_shift_px=None):
    """`flow_warped_frame` for SEVERAL channels that share one flow — the multi-channel form.

    The flow depends only on `guide`, so gating each channel with its own flow would recompute the
    identical Farneback field C times. Farneback is the expensive half here (a `calcOpticalFlowFarneback`
    per neighbour, against a `cv2.remap` to apply one), so a shared-flow form matters — same reasoning
    that made `gated_frames` share its match.

    `windows` is a sequence of (W, Y, X) windows, all co-registered with `guide`. Returns the fused
    CENTRE frame of each, in order. Duplicate-neighbour handling matches `gated_frames`: an even width
    uses `W // 2` as the centre; a window with fewer than 2 unique neighbours after clamping falls back
    to the centre unchanged (the coastal convention against averaging a frame with itself).
    """
    g = np.asarray(guide, dtype=np.float32)
    ws = [np.asarray(w, dtype=np.float32) for w in windows]
    for w in ws:
        if w.shape != g.shape:
            raise ValueError(f"window shape {w.shape} does not match guide shape {g.shape}")
    if g.shape[0] <= 1:
        return [w[0] for w in ws]
    c = g.shape[0] // 2
    # Flow computed from the guide's centre to each other guide frame, ONCE, then applied to each
    # channel. Deduplicated indices: on an edge the caller may have clamped several neighbours to
    # the same source frame (see `_gated_plane_series` for the same convention) — pairing centre
    # with itself gives a zero flow and inflates the centre's weight, so we skip those.
    idx_nb = [i for i in range(g.shape[0]) if i != c]
    flows = [_farneback_flow(g[c], g[i], winsize, pyr_levels, poly_n, poly_sigma) for i in idx_nb]
    out = []
    for w in ws:
        acc = w[c].astype(np.float32).copy()
        n = 1.0
        for i, flow in zip(idx_nb, flows):
            acc += _warp_by_flow(w[i], flow, max_shift_px)
            n += 1.0
        out.append(acc / n)
    return out


def temporal_flow_warped(arr, frames=DEFAULT_FRAMES, time_axis=0,
                         winsize=DEFAULT_FLOW_WINSIZE, pyr_levels=DEFAULT_FLOW_PYR_LEVELS,
                         poly_n=5, poly_sigma=1.2, max_shift_px=None, guide=None):
    """Motion-compensated temporal averaging via dense Farneback flow — whole-series form.

    For each timepoint, warp its ±half neighbours onto its coordinates via dense flow, then average.
    Unlike `temporal_gated` (block-match agreement), the flow is continuous, so smooth deformation
    is handled without the discretisation of a search grid; unlike a plain temporal mean, motion is
    compensated so moving cells stay sharp at wider windows.

    `guide` is the image the flow is derived from; it defaults to `arr` itself. Pass a shared one
    to warp several channels identically — see `flow_warped_frames`.

    `arr` may carry axes between time and the trailing (Y, X) — a Z stack is filtered plane by
    plane, never across Z (a stack collapse is a different operation with different invariants).
    """
    a = np.asarray(arr, dtype=np.float32)
    if frames is None or int(frames) <= 1 or a.shape[time_axis] <= 1:
        return a
    n = int(frames)
    if n % 2 == 0:
        n += 1
    n = min(n, a.shape[time_axis] | 1)
    half = n // 2
    T = a.shape[time_axis]

    g = a if guide is None else np.asarray(guide, dtype=np.float32)
    if g.shape != a.shape:
        raise ValueError(f"guide shape {g.shape} does not match array shape {a.shape}")

    a = np.moveaxis(a, time_axis, 0)
    g = np.moveaxis(g, time_axis, 0)
    lead, mid, spatial = a.shape[0], a.shape[1:-2], a.shape[-2:]
    a2 = a.reshape((lead, -1) + spatial)
    g2 = g.reshape((lead, -1) + spatial)

    out = np.empty_like(a2)
    for i in range(a2.shape[1]):
        for t in range(T):
            nb = [min(max(t + dt, 0), T - 1) for dt in range(-half, half + 1)]
            win_a = np.stack([a2[k, i] for k in nb])
            win_g = np.stack([g2[k, i] for k in nb])
            out[t, i] = flow_warped_frame(win_a, guide=win_g, winsize=winsize,
                                          pyr_levels=pyr_levels, poly_n=poly_n,
                                          poly_sigma=poly_sigma, max_shift_px=max_shift_px)
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
