"""Loss functions for training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _blur_at_cell_scale(x, blur_sigma):
    """Separable Gaussian, so a per-pixel statistic becomes a blob-shaped one."""
    if blur_sigma <= 0:
        return x
    radius = max(1, int(3 * blur_sigma))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(coords ** 2) / (2 * blur_sigma ** 2))
    k = k / k.sum()
    x = F.conv2d(x, k.view(1, 1, 1, -1), padding=(0, radius))
    return F.conv2d(x, k.view(1, 1, -1, 1), padding=(radius, 0))


def _blob_target(scalar_map, blur_sigma, norm_percentile):
    """Turn a [B,1,H,W] per-pixel foreground score into a blob-shaped BCE target in [0,1].

    Shared by `ConfettiForegroundLoss` and `ForegroundLoss`, which differ ONLY in how they derive
    `scalar_map`. That is not a tidiness choice — it is the measured finding. On real data the two
    derivations agree to `pearson r >= 0.99` (foreground IoU 96.3% on confetti, 94.1% on
    co-expressed markers), because these two steps are what carry the signal and the colour term
    contributes almost nothing. Keeping one implementation makes that structural instead of a
    coincidence that could drift apart. Measurements: docs/SEGMENTATION.md -> *What confetti
    actually contributes*.

    Rescale per image by a high PERCENTILE, not the max: with the max, one unusually bright cell
    sets the scale and every typical cell lands far below the inference prob_threshold, so the
    model learns a near-empty foreground. That is the diagnosed cause of the first confetti retrain
    collapsing to 14 labels/frame (see `ConfettiForegroundLoss`). A percentile is insensitive to
    that single outlier; values above it clamp to 1, which is what a saturated cell should read as.

    p99 measured on real AF-corrected frames (kSUFux/jHMfOI, r0hufV + ag0pAo): the fraction of
    detected cells whose target clears the 0.4 inference threshold is 21-24% under `max`, 92-93% at
    p99.5 and 100% at p99, for ~2.5% frame coverage - which matches the actual cell density. Below
    p98 the target starts claiming background.

    KNOWN LIMITATION — the rescale is purely relative, so it has no way to say "there is nothing
    here". A frame whose only content is noise on a NONZERO pedestal has that noise stretched to
    fill [0,1] and is claimed as ~100% foreground: the mirror of the "any objective without a
    coverage term rewards finding nothing" warning in `ConfettiForegroundLoss`. Pinned by
    `tests/test_loss_foreground.py::test_a_frame_of_pure_noise_on_a_pedestal_is_claimed_entirely`.

    This does not fire on the real data measured so far, because the background is exactly 0
    (photon-limited and clipped at import): across fXgbTl (32 z-planes), r0hufV (14) and 3w4IY5 (14)
    every plane has `p1 == 0`, and a genuinely empty plane has `p99 == 0` too, which already yields
    an empty target. It was left unguarded deliberately — a relative range cannot separate noise
    spread from cells (Gaussian noise has a relative p99-p1 range of ~37% of its mean, comparable to
    a real frame's), so the honest fix is an absolute signal estimate, not a threshold on contrast.
    Revisit if data with a camera bias offset appears.
    """
    target = _blur_at_cell_scale(scalar_map, blur_sigma)
    # .float(): under AMP the target arrives as float16 and torch.quantile rejects it.
    flat = target.view(target.size(0), -1).float()
    hi = torch.quantile(flat, norm_percentile / 100.0, dim=1).view(-1, 1, 1, 1) + 1e-6
    return torch.clamp(target.float() / hi, 0, 1)


def bce_floor(target):
    """The loss a PERFECT model achieves against this soft target: `mean H(target)`.

    Every prob-head term here is `binary_cross_entropy_with_logits` against a target that is a
    deterministic function of the input — brightness blurred and rescaled, dominant-colour
    confidence, an intensity/contrast/edge blend. BCE against a soft target cannot reach 0: its
    minimum is the target's own binary entropy, reached when the prediction equals the target
    exactly. That minimum is a property of the DATA, not of the model.

    Why record it: without it a loss curve is unreadable. Measured on flow.cyto (6 images, 2880
    frames), `foreground` settles at 0.26508 and this floor is 0.26499 — the model's entire
    remaining error is 0.00009, so 99.97% of the number plotted is a constant no model can move,
    and "the loss plateaus and nothing is learned after epoch 5" is a description of convergence.
    The same curve minus this floor goes to zero and says so.

    NOT applicable to the contrastive terms (`TemporalMetricsLoss`, `VarianceMetricsLoss`,
    `WarpConsistencyLoss`, `ConfettiBoundaryLoss`): they are hinges and cosine distances whose
    minimum IS 0, so there is nothing to subtract and a floor of 0 is the honest answer.

    Clamped rather than epsilon-added: `H(0)` and `H(1)` are exactly 0, but computing them directly
    is `0 * log(0)` = `nan`, and one saturated pixel would take out the whole epoch's mean. At the
    clamp `H` reads ~1.7e-6 instead of 0 — four orders below anything a loss curve is read to, and
    pinned by `tests/test_loss_floor.py::test_a_hard_target_has_a_floor_of_zero`.
    """
    with torch.no_grad():
        t = target.float().clamp(1e-7, 1 - 1e-7)
        return -(t * t.log() + (1 - t) * (1 - t).log()).mean()


def flow_discontinuity(metrics_dict, device=None):
    """A genuine flow-BOUNDARY signal from the metrics already computed: where does the flow field
    tear?

    `flow.extract_temporal_metrics` emits a metric named `cell_boundary_likelihood`, but it is
    `0.30*mag + 0.25*cumulative_mag + 0.25*edge_strength + 0.20*tangential_flow` — a blend of motion
    MAGNITUDE and image edge strength. It is therefore highest in the *interior* of a fast-moving
    cell and carries no information about two cells moving differently. Despite the name it is not a
    boundary prior, and nothing ever trained against it.

    What marks a cell-cell contact is the spatial DISCONTINUITY of the velocity field, not its
    magnitude. `strain` is the magnitude of the symmetric part of the velocity-gradient tensor and
    `vorticity` the antisymmetric part, so together they span ||grad v|| — no new flow computation is
    needed. `divergence` (the trace) is added because two cells pulling apart separate without shear.

    Returns [1, H, W] float in [0, 1], or None when the needed metrics are absent (so a caller with
    a reduced metric set degrades to the plain brightness target rather than erroring).

    Rationale for connecting this to the prob head at all: measured on 465 real touching
    different-colour pairs, 75.7% have relative motion above the flow noise floor (median 2.10
    px/frame against 1.28 px/frame within a cell), so ~3/4 of contacts are in principle separable
    from flow. See docs/SEGMENTATION.md -> *A real validation set*.
    """
    needed = [k for k in ("strain", "vorticity", "divergence") if k in metrics_dict]
    if not needed:
        return None
    parts = []
    for k in needed:
        v = metrics_dict[k]
        t = v if isinstance(v, torch.Tensor) else torch.from_numpy(v)
        parts.append(t.float().abs())
    d = torch.stack(parts, 0).sum(0)
    hi = torch.quantile(d.flatten(), 0.99) + 1e-6
    d = torch.clamp(d / hi, 0, 1)
    return d.unsqueeze(0).to(device) if device is not None else d.unsqueeze(0)


class ForegroundLoss(nn.Module):
    """Probability guidance from brightness at cell scale. No colour, no markers, no labels.

    The colour-blind form of `ConfettiForegroundLoss`, and the one to use on data that is not
    confetti. Target: per-pixel brightness (max over the input channels), blurred to cell scale,
    normalised by a high percentile — i.e. `_blob_target` applied to brightness rather than to
    dominant-colour confidence.

    **Why this exists: the colour term in `ConfettiForegroundLoss` was measured to do essentially
    nothing, even on real confetti.** Its target is `max_c softmax_ch_c`, and
    `flow.compute_variance_metrics` builds `softmax_ch_c` as a cross-channel softmax at
    `temp=0.3` over Gaussian-pooled (`pool_radius=5`) intensities, then rescales **each channel
    independently** with `normalize_metric`. Two consequences, both measured on
    `ccidDriftCorrected` mid-z, 6 frames:

    * The dominance share is pinned near its `1/C` floor. In the foreground on real confetti
      (kSUFux/r0hufV, C=3, floor 0.333): p5 0.356, median 0.397, p95 0.528, and **0% of
      foreground pixels are unambiguously one colour** (dominance > 0.8). So the factor the target
      multiplies brightness by is near-constant.
    * The per-channel `normalize_metric` then undoes the cross-channel comparability the softmax
      established, so the max is over independently-stretched maps.

    Net effect: this loss and `ConfettiForegroundLoss` produce the same target to
    `r = 0.9993` on confetti (foreground IoU 96.3%), `r = 0.9986` on co-expressed markers
    (IoU 94.1%), `r = 0.9906` on two-cell-type data (IoU 83.0%). The reported win of
    `ConfettiForegroundLoss` over `IntensityLoss` (2834 blobs -> 87, median 36 -> 90 px) is
    therefore attributable to the cell-scale blur and the p99 rescale, **not** to confetti.

    So this is not a downgrade for non-confetti data — it is the same objective with the
    inoperative term removed, and it needs no `variance_metrics`, which means no extra input
    channels and no train/inference mismatch (see `train.train_with_metrics`).

    Multi-marker data is the case that forces the distinction. On `zolIMa/fXgbTl`
    (SHG / nuc-GFP / mem-Tom / CD169-Kat) the confetti premise "a cell is a region of ONE colour"
    is simply false: nuc-GFP and mem-Tom label the nucleus and the membrane of the *same* cell, so
    a correct segmentation spans two colours. Brightness has no such problem.

    Pass `channels=` to `normalize_and_project` to keep structural channels (SHG, THG) out of the
    projection this reads — they are bright and are not cells.
    """

    def __init__(self, blur_sigma: float = 1.0, norm_percentile: float = 99.0,
                 boundary_weight: float = 0.0):
        super().__init__()
        self.blur_sigma = blur_sigma
        self.norm_percentile = norm_percentile
        self.boundary_weight = boundary_weight

    def target(self, frame, boundary=None):
        """The BCE target this loss fits — ONE implementation, used by `forward` and by the caller
        that wants `bce_floor` of it.

        Public because the floor of a soft target is only meaningful for the target the loss
        actually used: a second copy of this arithmetic in the training loop is how a reported
        floor and the curve it is subtracted from drift apart while both look plausible.

        Args:
            frame:    [B, C, H, W] intensity image(s); C>1 is reduced by max over channels
            boundary: optional [B, 1, H, W] flow-discontinuity map (see `flow_discontinuity`).
                      Where it is high the target is suppressed, so the prob map pinches between
                      differently-moving cells. This is the ONLY path by which optical flow
                      reaches the labels — see the class docstring.
        """
        bright = frame.float()
        bright = bright.max(dim=1, keepdim=True).values if bright.size(1) > 1 else bright
        target = _blob_target(bright, self.blur_sigma, self.norm_percentile)
        if boundary is not None and self.boundary_weight > 0:
            # Subtract, not multiply: a multiplicative gate would scale down whole cells wherever
            # they move at all, which is what `cell_boundary_likelihood` already does wrong (it is
            # a motion-MAGNITUDE blend, high in the interior of a fast cell). Subtracting a
            # blob-scaled discontinuity map carves a trough only where the flow field actually
            # tears, which is what separates two cells sliding past each other.
            b = _blob_target(boundary.float(), self.blur_sigma, self.norm_percentile)
            target = torch.clamp(target - self.boundary_weight * b, 0.0, 1.0)
        return target

    def with_floor(self, pred_prob, frame, boundary=None):
        """`(loss, floor)` from ONE target build — what the training loop calls.

        Both from the same `target()` call, deliberately: a floor computed from a separately built
        target is a number that can drift from the curve it is subtracted from while both still
        look plausible. See `bce_floor`.
        """
        t = self.target(frame, boundary)
        return F.binary_cross_entropy_with_logits(pred_prob.float(), t), bce_floor(t)

    def forward(self, pred_prob, frame, boundary=None):
        """
        Args:
            pred_prob: [B, 1, H, W] raw model logits (before sigmoid)
            frame:     [B, C, H, W] intensity image(s); C>1 is reduced by max over channels
            boundary:  optional [B, 1, H, W] flow-discontinuity map — see `target`.
        Returns:
            BCE against the brightness-at-cell-scale target.
        """
        return self.with_floor(pred_prob, frame, boundary)[0]


class ConfettiForegroundLoss(nn.Module):
    """Probability guidance from confetti colour instead of grayscale texture.

    The premise (Dominik's): a cell is a contiguous region of ONE confetti colour, so a
    region spanning several colours is a segmentation error. That makes "one colour
    dominates here, brightly" a foreground signal — and unlike intensity/edge statistics
    it is smooth across a cell interior rather than peaking at boundaries and on noise.

    Why this exists: `IntensityLoss` builds its target as
    `0.5*bright + 0.3*local_contrast + 0.2*edge`, i.e. half a per-pixel intensity
    threshold and half two edge detectors, with no notion of a cell-sized object and no
    confetti input at all. Measured on a real frame, the resulting prob map thresholds
    into **2535 connected components, 98% of them under 100 px, median 3 px** — the model
    is being asked to reproduce speckle, which is the origin of the ~86% fragment rate
    downstream. See docs/SEGMENTATION.md.

    The target is built from the variance metrics already computed for training
    (`flow.compute_variance_metrics` → `softmax_ch_*`), which are a per-channel local
    softmax scaled by brightness. Their per-pixel max is therefore high only where one
    channel clearly dominates and the pixel is bright: cell interiors, not background and
    not colour-ambiguous overlaps. A blur at roughly cell scale turns it from a per-pixel
    statistic into a blob-shaped objective.

    Note this guards **under**-segmentation only. A fragment of a single colour looks
    perfectly good to this signal, so it cannot penalise over-segmentation on its own —
    the blur is what supplies a size prior.

    **The first attempt collapsed detection, and the cause was target scaling, not the
    idea.** The blurred colour-confidence map was normalised by its per-image **max**, so
    only the brightest cell approached 1.0 while typical cells landed well below the prob
    threshold used at inference — on real frames just 22% of cells cleared it. Normalising
    by `norm_percentile` (default p99) instead puts 100% of them above it. Retrained on the
    16-bit AF+drift movies that fix turns 2834 blobs / 161 labels / median 36 px into
    87 blobs / 32 labels / median 90 px, with full cell coverage.

    `blur_sigma` is a merge↔split dial, not a separation knob: measured against synthetic
    crowded ground truth (2 movies x 2 densities x 3 frames, each at its own best prob
    threshold), all three settings sit on one frontier —

        blur 2.0   thr 0.50   F1@.35 65.2%   merged 4.5%   split  6.2%   area/GT 0.94
        blur 1.0   thr 0.40   F1@.35 66.6%   merged 2.4%   split  9.2%   area/GT 0.71
        no blur    thr 0.15   F1@.35 50.3%   merged 0.1%   split 29.2%   area/GT 0.52

    Sharpening the target buys fewer merges by shattering cells and shrinking masks. 1.0 is
    the default because it wins on F1 outright while roughly halving merges; do not read it
    as a fix for touching cells of different colour. Most of what remains is downstream —
    56% of merges come from `segment._merge_split_instances`, 33% from region growing.

    Beware the metrics that made the collapsed version look like a success: fragment-% and
    multi-colour-% both improved dramatically (86%→90% frag but 38.7%→4.3% under-
    segmentation, purity 1.000) because neither has a coverage term. Any objective without
    one rewards finding nothing.
    """

    def __init__(self, blur_sigma: float = 1.0, norm_percentile: float = 99.0):
        super().__init__()
        self.blur_sigma = blur_sigma
        self.norm_percentile = norm_percentile

    def _blur(self, x):
        """Separable Gaussian at cell scale, so the target is blobs not pixels."""
        return _blur_at_cell_scale(x, self.blur_sigma)

    def target(self, pred_prob, variance_metrics):
        """The BCE target this loss fits, or `None` when no `softmax_ch_*` metrics were supplied.

        ONE implementation, shared with `bce_floor` — see `ForegroundLoss.target`. `pred_prob` is
        taken only for its device and dtype; nothing about the prediction enters the target.
        """
        targets = []
        for m in variance_metrics:
            keys = sorted(k for k in m.keys() if k.startswith('softmax_ch_'))
            if not keys:
                return None
            stack = torch.stack([
                (v if isinstance(v, torch.Tensor) else torch.from_numpy(v))
                .to(pred_prob.device).float()
                for v in (m[k] for k in keys)
            ], dim=0)                                   # [C, H, W]
            targets.append(stack.max(dim=0).values)     # dominant-colour confidence
        scalar_map = torch.stack(targets, dim=0).unsqueeze(1)   # [B, 1, H, W]
        # The blur and the percentile rescale are shared with `ForegroundLoss` — see `_blob_target`,
        # which also records the measurement that the two agree to r >= 0.99, because the
        # dominant-colour term above is pinned near its 1/C floor on real data.
        return _blob_target(scalar_map, self.blur_sigma, self.norm_percentile)

    def with_floor(self, pred_prob, variance_metrics):
        """`(loss, floor)` from ONE target build — see `ForegroundLoss.with_floor`.

        Both are 0 when no confetti metrics were supplied: with no target there is no objective, so
        a floor of 0 is the honest report rather than a `nan` that would poison the curve.
        """
        t = self.target(pred_prob, variance_metrics)
        if t is None:
            zero = torch.zeros((), device=pred_prob.device, dtype=torch.float32)
            return zero, zero
        return F.binary_cross_entropy_with_logits(pred_prob.float(), t), bce_floor(t)

    def forward(self, pred_prob, variance_metrics):
        """
        Args:
            pred_prob:         [B, 1, H, W] raw model logits (before sigmoid)
            variance_metrics:  list of B dicts with `softmax_ch_*` [H, W] arrays
        Returns:
            BCE against the confetti foreground target, or 0.0 if no metrics supplied.
        """
        return self.with_floor(pred_prob, variance_metrics)[0]


class ConfettiBoundaryLoss(nn.Module):
    """Push embeddings apart across a confetti-colour boundary; pull them together within a colour.

    Measured motivation. Segmentation merges **86.7% of real touching different-colour cell
    pairs** (465 pairs mined from all 9 confetti movies, colour as ground truth). The
    embeddings are the reason: cosine across a different-colour contact is 0.920 against
    0.945 within a single cell — Cohen's d = 0.22, i.e. a cell boundary is nearly
    indistinguishable from cell interior. Nothing downstream can recover that; see
    docs/SEGMENTATION.md.

    Why the existing terms do not supply it:

    - ``VarianceMetricsLoss`` is negatives-only and mines the k pixels *farthest* in metric
      space within a window — cells that are already far apart, the easy case. Two pixels
      straddling a contact are never presented as a negative.
    - ``TemporalMetricsLoss`` *pulls together* pixels with similar motion, which is exactly
      what two touching cells drifting as a pair have. It actively fuses them.

    So this term mines the contacts **explicitly** rather than sampling and hoping. That
    matters because contacts are rare: the training movies are sparse, and a random window
    almost never contains one. Explicit mining is what makes the signal usable at this
    density — but it cannot manufacture events that are not there, so genuinely crowded
    confetti data remains the real fix.

    A pixel pair at offset ``offsets`` is a **negative** when the two pixels' dominant
    confetti channels differ and both are confidently coloured, and a **positive** when they
    agree. Negatives get a hinge on cosine similarity; positives a symmetric pull, so the
    term cannot be minimised by simply scattering every embedding.

    Confetti is used to build the *pairing*, never as model input — identical in kind to
    ``ConfettiForegroundLoss``. At inference the variance channels are zero-filled and the
    network must predict boundaries from greyscale + flow alone. That caps what this can
    achieve: ~24% of real contacts are co-moving, and for those nothing in the input
    distinguishes the contact from cell interior. The other ~76% have resolvable relative
    motion (median 2.10 px/frame against a 1.28 px/frame within-cell noise floor).
    """

    def __init__(self, margin: float = 0.5, min_confidence: float = 0.5,
                 offsets: tuple = (2, 4), pos_weight: float = 0.3,
                 max_pairs: int = 4096):
        super().__init__()
        self.margin = margin
        self.min_confidence = min_confidence
        self.offsets = offsets
        self.pos_weight = pos_weight
        self.max_pairs = max_pairs

    @staticmethod
    def _colour(m, device):
        """(dominant channel, confidence) from the softmax_ch_* metrics, or None."""
        keys = sorted(k for k in m.keys() if k.startswith('softmax_ch_'))
        if len(keys) < 2:
            return None, None
        stack = torch.stack([
            (v if isinstance(v, torch.Tensor) else torch.from_numpy(v)).to(device).float()
            for v in (m[k] for k in keys)
        ], dim=0)                                        # [C, H, W]
        total = stack.sum(dim=0) + 1e-6
        conf, dom = (stack / total).max(dim=0)           # share held by the winner
        # Scale by brightness so background — where one channel "wins" on noise alone — is
        # never confidently coloured. Percentile, not max, for the same reason as
        # ConfettiForegroundLoss: one bright cell must not set the scale.
        bright = stack.sum(dim=0)
        hi = torch.quantile(bright.flatten().float(), 0.99) + 1e-6
        return dom, conf * torch.clamp(bright / hi, 0, 1)

    def forward(self, metric_emb, variance_metrics):
        """
        Args:
            metric_emb:        [B, D, H, W] embeddings
            variance_metrics:  list of B dicts with `softmax_ch_*` [H, W] arrays
        Returns:
            Scalar loss, or 0.0 when no confetti metrics are supplied (so it is a no-op at
            inference and on grayscale-only training data).
        """
        if not variance_metrics:
            return torch.zeros((), device=metric_emb.device, dtype=torch.float32)

        device = metric_emb.device
        emb = F.normalize(metric_emb.float(), dim=1, p=2)
        B = emb.shape[0]
        losses = []

        for b in range(B):
            m = variance_metrics[b] if isinstance(variance_metrics, list) else variance_metrics
            dom, conf = self._colour(m, device)
            if dom is None:
                continue
            valid = conf > self.min_confidence
            e = emb[b]                                   # [D, H, W]

            for off in self.offsets:
                # Compare each pixel with the one `off` away, along each axis in turn. A colour
                # boundary shows up as a pair that is close in space but different in colour.
                for axis in (0, 1):
                    lo = (slice(off, None), slice(None)) if axis == 0 else \
                         (slice(None), slice(off, None))
                    hi = (slice(None, -off), slice(None)) if axis == 0 else \
                         (slice(None), slice(None, -off))
                    a, b_ = e[(slice(None),) + lo], e[(slice(None),) + hi]
                    if a.numel() == 0:
                        continue
                    d1, d2 = dom[lo], dom[hi]
                    both = valid[lo] & valid[hi]
                    if not both.any():
                        continue
                    sim = (a * b_).sum(dim=0)            # cosine, embeddings are unit norm
                    diff = both & (d1 != d2)             # across a colour boundary
                    same = both & (d1 == d2)             # within one colour

                    if diff.any():
                        s = sim[diff]
                        if s.numel() > self.max_pairs:
                            s = s[torch.randperm(s.numel(), device=device)[:self.max_pairs]]
                        losses.append(torch.clamp(self.margin + s, min=0.0).mean())
                    if same.any() and self.pos_weight > 0:
                        s = sim[same]
                        if s.numel() > self.max_pairs:
                            s = s[torch.randperm(s.numel(), device=device)[:self.max_pairs]]
                        losses.append(self.pos_weight * (1.0 - s).mean())

        if not losses:
            return torch.zeros((), device=device, dtype=torch.float32)
        return torch.stack(losses).mean()


class IntensityLoss(nn.Module):
    """Probability guidance using intensity + contrast + edges.

    NOTE: the target is `0.5*bright + 0.3*local_contrast + 0.2*edge` — half a per-pixel
    intensity threshold, half two edge detectors — so it has no cell-scale structure and
    ignores confetti entirely. It trains the prob head to reproduce speckle (measured:
    2535 components per frame, median 3 px). Prefer `ConfettiForegroundLoss`; see
    docs/SEGMENTATION.md.
    """

    def target(self, frame):
        """The BCE target this loss fits — ONE implementation, shared with `bce_floor`.

        See `ForegroundLoss.target` for why this is public rather than inlined in `forward`.
        """
        # Build target in float32 regardless of AMP dtype
        frame = frame.float()

        frame_mean = frame.view(frame.size(0), -1).mean(dim=1, keepdim=True).view(-1, 1, 1, 1)
        frame_std = frame.view(frame.size(0), -1).std(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-5
        bright = (frame > frame_mean + frame_std).float()

        frame_unfold = F.unfold(frame, kernel_size=5, padding=2)
        local_std = frame_unfold.std(dim=1, keepdim=True) + 1e-5
        contrast = local_std.view_as(frame)

        contrast_mean = contrast.view(contrast.size(0), -1).mean(dim=1, keepdim=True).view(-1, 1, 1, 1)
        contrast_std = contrast.view(contrast.size(0), -1).std(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-5
        contrast_norm = torch.clamp((contrast - contrast_mean) / contrast_std, 0, 1)

        edge_y = torch.abs(frame[:, :, 1:, :] - frame[:, :, :-1, :])
        edge_x = torch.abs(frame[:, :, :, 1:] - frame[:, :, :, :-1])
        edge_y = F.pad(edge_y, (0, 0, 0, 1))
        edge_x = F.pad(edge_x, (0, 1, 0, 0))
        edge = (edge_x + edge_y) / 2

        edge_mean = edge.view(edge.size(0), -1).mean(dim=1, keepdim=True).view(-1, 1, 1, 1)
        edge_std = edge.view(edge.size(0), -1).std(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-5
        edge_norm = torch.clamp((edge - edge_mean) / edge_std, 0, 1)

        cell_target = 0.5 * bright + 0.3 * contrast_norm + 0.2 * edge_norm
        return cell_target ** 0.5  # push cell regions toward 1.0

    def with_floor(self, pred_prob, frame):
        """`(loss, floor)` from ONE target build — see `ForegroundLoss.with_floor`."""
        t = self.target(frame)
        # binary_cross_entropy_with_logits is AMP-safe (fuses sigmoid internally)
        return F.binary_cross_entropy_with_logits(pred_prob.float(), t), bce_floor(t)

    def forward(self, pred_prob, frame):
        """
        Args:
            pred_prob: [B, 1, H, W] raw model logits (before sigmoid)
            frame: [B, 1, H, W] intensity images
        """
        return self.with_floor(pred_prob, frame)[0]


def _contrastive_metric_loss(metric_emb, metrics_dict, k_neighbors, margin, max_pixels=5000):
    B, D, H, W = metric_emb.shape

    # A TENSOR, not 0.0: with an empty metrics dict every batch item hits the `continue` below, and a
    # float initialiser then makes `total_loss / B` a plain float — which the training loop calls
    # `.item()` on, raising AttributeError. That only happens when there are no metrics at all, i.e.
    # exactly the flow-ablation case, so it went unnoticed.
    #
    # NO `requires_grad=True` here: that makes it a leaf, and the `total_loss += ...` below is then an
    # in-place write to a leaf requiring grad, which torch refuses. Gradient still flows because the
    # tensors being added carry it; this initialiser only has to be the right *type*.
    total_loss = torch.zeros((), device=metric_emb.device)

    for b in range(B):
        # Extract metrics for batch b
        if isinstance(metrics_dict, list):
            metrics_dict_b = metrics_dict[b]
        else:
            metrics_dict_b = metrics_dict

        metric_list = []
        for name in sorted(metrics_dict_b.keys()):
            arr = metrics_dict_b[name]
            if isinstance(arr, torch.Tensor):
                tensor = arr.float().to(metric_emb.device)
            else:
                tensor = torch.from_numpy(arr).float().to(metric_emb.device)
            metric_list.append(tensor)

        if not metric_list:
            continue

        metrics_stacked = torch.stack(metric_list, dim=0)  # [M, H, W]
        num_metrics = len(metric_list)

        emb_flat = metric_emb[b].view(D, -1).T  # [H*W, D]
        metrics_flat = metrics_stacked.view(num_metrics, -1).T  # [H*W, M]

        n_pixels = len(metrics_flat)
        if n_pixels > max_pixels:
            indices = torch.randperm(n_pixels, device=metrics_flat.device)[:max_pixels]
            metrics_flat = metrics_flat[indices]
            emb_flat = emb_flat[indices]
            n_pixels = max_pixels

        metrics_norm = F.normalize(metrics_flat, dim=1, p=2)
        emb_norm = F.normalize(emb_flat, dim=1, p=2)

        metric_dist = torch.cdist(metrics_norm, metrics_norm, p=2)
        emb_sim = torch.mm(emb_norm, emb_norm.T)

        k = min(k_neighbors, n_pixels - 1)
        if k < 1:
            continue

        _, sorted_indices = torch.sort(metric_dist, dim=1)
        pos_indices = sorted_indices[:, 1:k + 1]
        neg_indices = sorted_indices[:, -k:]

        pos_sims = emb_sim.gather(1, pos_indices)
        neg_sims = emb_sim.gather(1, neg_indices)

        loss_pos = torch.clamp(1.0 - pos_sims, min=0.0).mean()
        loss_neg = torch.clamp(margin + neg_sims, min=0.0).mean()
        total_loss += loss_pos + loss_neg

    return total_loss / B if B > 0 else torch.tensor(0.0, device=metric_emb.device, requires_grad=True)
  

class TemporalMetricsLoss(nn.Module):
    """Embeddings preserve temporal (optical flow) metric structure via contrastive learning."""

    def __init__(self, k_neighbors=10, margin=0.5, max_pixels=2000):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.margin = margin
        self.max_pixels = max_pixels

    def forward(self, metric_emb, metrics_dict):
        return _contrastive_metric_loss(
            metric_emb, metrics_dict, self.k_neighbors, self.margin, self.max_pixels
        )


class WarpConsistencyLoss(nn.Module):
    """Self-supervised temporal embedding consistency via optical flow warping.

    Pulls emb[t, y, x] toward emb[t+1, y+v, x+u] using bilinear warping.
    Operates on foreground pixels only (prob_t > prob_threshold).
    Uses cosine distance to avoid norm collapse.
    """

    def __init__(self, prob_threshold: float = 0.3):
        super().__init__()
        self.prob_threshold = prob_threshold

    def forward(
        self,
        emb_t:   "torch.Tensor",   # [B, D, H, W]
        emb_t1:  "torch.Tensor",   # [B, D, H, W]
        flow_uv: "torch.Tensor",   # [B, 2, H, W]  [0]=u(x-dir), [1]=v(y-dir) in pixels
        prob_t:  "torch.Tensor",   # [B, 1, H, W]  logits
    ) -> "torch.Tensor":
        B, D, H, W = emb_t.shape
        device = emb_t.device

        # Build identity grid in normalised [-1, 1] coordinates
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')   # [H, W]
        base = torch.stack([grid_x, grid_y], dim=-1)              # [H, W, 2]
        base = base.unsqueeze(0).expand(B, -1, -1, -1)            # [B, H, W, 2]

        # Convert pixel-space displacement to normalised coordinates
        u_norm = flow_uv[:, 0] / (W / 2.0)   # [B, H, W]
        v_norm = flow_uv[:, 1] / (H / 2.0)   # [B, H, W]
        # grid_sample grid[..., 0] = x (column), grid[..., 1] = y (row)
        disp = torch.stack([u_norm, v_norm], dim=-1)  # [B, H, W, 2]

        # Backward-warp: sample emb_t1 at locations shifted by the flow
        grid = (base + disp).clamp(-1, 1)
        emb_t1_warped = F.grid_sample(
            emb_t1.float(), grid, mode='bilinear', align_corners=True, padding_mode='border'
        )  # [B, D, H, W]

        # Foreground mask
        mask = (prob_t.sigmoid() > self.prob_threshold).squeeze(1)  # [B, H, W]
        if not mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        e_t  = emb_t.permute(0, 2, 3, 1)[mask]          # [P, D]
        e_w  = emb_t1_warped.permute(0, 2, 3, 1)[mask]  # [P, D]
        cos  = F.cosine_similarity(e_t.float(), e_w.float(), dim=-1)  # [P]
        return (1.0 - cos).mean()


class VarianceMetricsLoss(nn.Module):
    """Windowed repulsion loss on cross-channel variance metrics.

    Within random spatial windows, pushes apart the embeddings of the pixels that are
    *farthest* in metric space (the k hardest negatives per pixel): it is a
    **negatives-only** term — there is no positive/attraction pull here. The attractive
    signal comes from the other losses (``IntensityLoss``, ``TemporalMetricsLoss``);
    this one only adds cross-cell separation on the variance channels.

    NOTE: an earlier docstring claimed this also pulled same-cell positives together; it
    never did. If a symmetric push/pull is wanted, add a ``loss_pos`` on ``sorted_idx[:, 1:k+1]``
    mirroring ``TemporalMetricsLoss`` — that is a training-behaviour change, deliberately
    left out here.
    """

    def __init__(self, k_neighbors=10, margin=0.5, window_size=32, max_tiles=8):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.margin = margin
        self.window_size = window_size
        self.max_tiles = max_tiles

    def forward(self, metric_emb, metrics_dict, frame_indices=None, max_pixels_per_patch=512):
        B, D, H, W = metric_emb.shape

        pw = min(self.window_size, H, W)
        all_tiles = [(y, x) for y in range(0, max(1, H - pw + 1), pw)
                             for x in range(0, max(1, W - pw + 1), pw)]

        total_loss = torch.tensor(0.0, device=metric_emb.device)

        for b in range(B):
            seed = frame_indices[b] if frame_indices is not None else 0
            if len(all_tiles) > self.max_tiles:
                g = torch.Generator()
                g.manual_seed(seed)
                perm = torch.randperm(len(all_tiles), generator=g)[:self.max_tiles].tolist()
                tiles = [all_tiles[i] for i in perm]
            else:
                tiles = all_tiles
            metrics_dict_b = metrics_dict[b] if isinstance(metrics_dict, list) else metrics_dict
            metric_list = []
            for name in sorted(metrics_dict_b.keys()):
                arr = metrics_dict_b[name]
                tensor = arr.float().to(metric_emb.device) if isinstance(arr, torch.Tensor) \
                    else torch.from_numpy(arr).float().to(metric_emb.device)
                metric_list.append(tensor)

            if not metric_list:
                continue

            met_b = torch.stack(metric_list, dim=0)  # [M, H, W]
            M = len(metric_list)
            patch_losses = []

            for y0, x0 in tiles:
                emb_patch = metric_emb[b, :, y0:y0+pw, x0:x0+pw].reshape(D, -1).T  # [pw², D]
                met_patch = met_b[:, y0:y0+pw, x0:x0+pw].reshape(M, -1).T          # [pw², M]

                n_px = met_patch.shape[0]
                if n_px > max_pixels_per_patch:
                    step = max(1, n_px // max_pixels_per_patch)
                    idx = torch.arange(0, n_px, step, device=met_patch.device)[:max_pixels_per_patch]
                    met_patch = met_patch[idx]
                    emb_patch = emb_patch[idx]
                    n_px = len(idx)

                k = min(self.k_neighbors, n_px - 1)
                if k < 1:
                    continue

                met_norm = F.normalize(met_patch, dim=1, p=2)
                emb_norm = F.normalize(emb_patch, dim=1, p=2)

                met_dist = torch.cdist(met_norm, met_norm, p=2)
                emb_sim = torch.mm(emb_norm, emb_norm.T)

                _, sorted_idx = torch.sort(met_dist, dim=1)
                neg_sims = emb_sim.gather(1, sorted_idx[:, -k:])

                loss_neg = torch.clamp(self.margin + neg_sims, min=0.0).mean()
                patch_losses.append(loss_neg)

            if patch_losses:
                total_loss = total_loss + torch.stack(patch_losses).mean()

        return total_loss / B
