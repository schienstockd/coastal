"""coastal.denoise — denoising for fluorescence microscopy.

The network (``CPnet``) and the normalization / tiling / stitching harness are **adapted from
Cellpose 3** (BSD-3-Clause, © 2023 Howard Hughes Medical Institute; Carsen Stringer & Marius
Pachitariu) — see ``THIRD_PARTY.md``. Cellpose is a **reference, not a runtime dependency**:
this module reproduces the architecture so it can load Cellpose's *public pretrained weights*,
but it never imports the ``cellpose`` package. That is deliberate — it lets a downstream env run
Cellpose 4 for segmentation while still getting Cellpose-3-quality restoration (the two are the
same PyPI distribution and cannot coexist; see ``docs/todo/DENOISE_PLAN.md``).

Method lineage (preserve in attribution): Cellpose 3 restoration builds on CARE / CSBDeep
(Weigert et al., Nat Methods 2018) and Noise2Void / Noise2Self self-supervision.

Runtime deps are torch + numpy + opencv (all already coastal deps). Device selection goes through
``coastal.device.resolve_device`` (cuda → mps → cpu). Mixed precision is **CUDA-only** by design
(``autocast=True``); MPS is a known crash surface for torch (see DENOISE_PLAN Decision 5) and the
fp16 path is never enabled there.

Public API
----------
- ``DenoiseModel(model_type=..., device=...)`` — holds the loaded net; reuse across planes.
- ``denoise_image(arr, ...)`` — one-shot convenience wrapper.
- ``denoise_preserving_ratio(arr, channels=...)`` — multi-channel denoising that leaves the ratios
  between channels intact. Use this, not per-channel ``denoise_image``, on confetti data: identity
  is the ratio, and restoring channels independently destroys it. **Needs no Cellpose weights** —
  it defaults to a Gaussian restorer, which measurement puts within 0.7pp of the net.
- ``ratio_preserving_gain(arr, restored, ...)`` — the array algebra behind it, model-free.
- ``gaussian_restorer(sigma=...)`` / ``temporal_mean_restorer(window=...)`` /
  ``cellpose_restorer(...)`` — pluggable projection restorers. For movies the temporal wins.

Input is grayscale: a 2-D ``(Y, X)`` plane or a stack of planes ``(Z, Y, X)`` / ``(T, Y, X)``
(each plane normalized and restored independently — matching how cecelia's cleanup task feeds
planes). Output is float32 of the same spatial shape; the Cellpose restoration range is ~[-1, 10]
(the caller rescales to bit depth).
"""

import os
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, uniform_filter1d

from coastal.device import resolve_device

# --- pretrained weights -----------------------------------------------------------------------
# Cellpose 3 restoration models. Weights are downloaded from cellpose.org on first use and cached.
# We reuse the standard ~/.cellpose/models cache by default so a machine that already has Cellpose
# does not re-download; override with COASTAL_MODELS_PATH or CELLPOSE_LOCAL_MODELS_PATH.
_MODEL_URL = "https://www.cellpose.org/models"
# Denoise models only. Cellpose 3 also publishes deblur_*/upsample_* restoration weights, but
# cecelia's cleanup only needs denoising (and upsample changes the output size — incompatible with
# the same-size correction the cleanup task writes), so we deliberately ship denoise-only.
MODEL_NAMES = ["denoise_cyto3", "denoise_cyto2", "denoise_nuclei"]


def _models_dir():
    env = os.environ.get("COASTAL_MODELS_PATH") or os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
    return Path(env) if env else Path.home() / ".cellpose" / "models"


def _weight_path(model_type):
    """Return the local path to the weights for ``model_type``, downloading if absent."""
    d = _models_dir()
    d.mkdir(parents=True, exist_ok=True)
    cached = d / model_type
    if not cached.exists():
        url = f"{_MODEL_URL}/{model_type}"
        # torch.hub gives a progress bar and atomic download; no extra dependency.
        torch.hub.download_url_to_file(url, os.fspath(cached), progress=True)
    return os.fspath(cached)


# ================================ CPnet architecture ==========================================
# Adapted verbatim (module structure preserved so Cellpose state_dict keys align) from
# cellpose/resnet_torch.py — BSD-3-Clause, © 2023 HHMI (Stringer & Pachitariu). Restoration uses
# the 2-D, nout=1 configuration; the segmentation-only 3-D / mkldnn branches are dropped.

def _batchconv(in_c, out_c, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_c, eps=1e-5, momentum=0.05),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_c, out_c, sz, padding=sz // 2),
    )


def _batchconv0(in_c, out_c, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_c, eps=1e-5, momentum=0.05),
        nn.Conv2d(in_c, out_c, sz, padding=sz // 2),
    )


class _ResDown(nn.Module):
    def __init__(self, in_c, out_c, sz):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj = _batchconv0(in_c, out_c, 1)
        for t in range(4):
            self.conv.add_module("conv_%d" % t,
                                 _batchconv(in_c if t == 0 else out_c, out_c, sz))

    def forward(self, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x


class _Downsample(nn.Module):
    def __init__(self, nbase, sz, max_pool=True):
        super().__init__()
        self.down = nn.Sequential()
        self.maxpool = nn.MaxPool2d(2, stride=2) if max_pool else nn.AvgPool2d(2, stride=2)
        for n in range(len(nbase) - 1):
            self.down.add_module("res_down_%d" % n, _ResDown(nbase[n], nbase[n + 1], sz))

    def forward(self, x):
        xd = []
        for n in range(len(self.down)):
            y = self.maxpool(xd[n - 1]) if n > 0 else x
            xd.append(self.down[n](y))
        return xd


class _BatchConvStyle(nn.Module):
    def __init__(self, in_c, out_c, style_c, sz):
        super().__init__()
        self.conv = _batchconv(in_c, out_c, sz)
        self.full = nn.Linear(style_c, out_c)

    def forward(self, style, x, y=None):
        if y is not None:
            x = x + y
        feat = self.full(style)
        for _ in range(len(x.shape[2:])):
            feat = feat.unsqueeze(-1)
        return self.conv(x + feat)


class _ResUp(nn.Module):
    def __init__(self, in_c, out_c, style_c, sz):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module("conv_0", _batchconv(in_c, out_c, sz))
        self.conv.add_module("conv_1", _BatchConvStyle(out_c, out_c, style_c, sz))
        self.conv.add_module("conv_2", _BatchConvStyle(out_c, out_c, style_c, sz))
        self.conv.add_module("conv_3", _BatchConvStyle(out_c, out_c, style_c, sz))
        self.proj = _batchconv0(in_c, out_c, 1)

    def forward(self, x, y, style):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y=y)
        x = x + self.conv[3](style, self.conv[2](style, x))
        return x


class _MakeStyle(nn.Module):
    def forward(self, x0):
        style = F.avg_pool2d(x0, kernel_size=x0.shape[2:])
        style = nn.Flatten()(style)
        style = style / torch.sum(style ** 2, axis=1, keepdim=True) ** .5
        return style


class _Upsample(nn.Module):
    def __init__(self, nbase, sz):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode="nearest")
        self.up = nn.Sequential()
        for n in range(1, len(nbase)):
            self.up.add_module("res_up_%d" % (n - 1), _ResUp(nbase[n], nbase[n - 1], nbase[-1], sz))

    def forward(self, style, xd):
        x = self.up[-1](xd[-1], xd[-1], style)
        for n in range(len(self.up) - 2, -1, -1):
            x = self.upsampling(x)
            x = self.up[n](x, xd[n], style)
        return x


class CPnet(nn.Module):
    """Cellpose residual-UNet-with-style, 2-D restoration configuration (nout=1).

    Attribute names mirror cellpose/resnet_torch.py so the public Cellpose 3 weight files load via
    ``load_state_dict`` unchanged.
    """

    def __init__(self, nbase, nout, sz, max_pool=True, diam_mean=30.):
        super().__init__()
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        self.downsample = _Downsample(nbase, sz, max_pool=max_pool)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = _Upsample(nbaseup, sz)
        self.make_style = _MakeStyle()
        self.output = _batchconv(nbaseup[0], nout, 1)
        self.diam_mean = nn.Parameter(data=torch.ones(1) * diam_mean, requires_grad=False)
        self.diam_labels = nn.Parameter(data=torch.ones(1) * diam_mean, requires_grad=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data):
        T0 = self.downsample(data)
        style = self.make_style(T0[-1])
        T1 = self.upsample(style, T0)
        T1 = self.output(T1)
        return T1, style


# ============================ normalize / tile / stitch (ported) ==============================
# Ported from cellpose/transforms.py + core.py (BSD-3-Clause, © 2023 HHMI). Reproduced to match
# Cellpose output numerically without importing cellpose. Only the inference path is kept
# (no augmentation, no 3-D conv, no mkldnn).

def normalize99(Y, lower=1, upper=99):
    """Scale so 0.0 = ``lower`` percentile, 1.0 = ``upper`` percentile (cellpose ``normalize99``)."""
    X = Y.astype(np.float32, copy=True)
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    if x99 - x01 > 1e-3:
        X -= x01
        X /= (x99 - x01)
    else:
        X[:] = 0
    return X


def _taper_mask(ly=224, lx=224, sig=7.5):
    bsize = max(224, max(ly, lx))
    xm = np.arange(bsize)
    xm = np.abs(xm - xm.mean())
    mask = 1 / (1 + np.exp((xm - (bsize / 2 - 20)) / sig))
    mask = mask * mask[:, np.newaxis]
    mask = mask[bsize // 2 - ly // 2:bsize // 2 + ly // 2 + ly % 2,
                bsize // 2 - lx // 2:bsize // 2 + lx // 2 + lx % 2]
    return mask


def _get_pad_yx(Ly, Lx, div=16, extra=1):
    Lpad = int(div * np.ceil(Ly / div) - Ly)
    ypad1 = extra * div // 2 + Lpad // 2
    ypad2 = extra * div // 2 + Lpad - Lpad // 2
    Lpad = int(div * np.ceil(Lx / div) - Lx)
    xpad1 = extra * div // 2 + Lpad // 2
    xpad2 = extra * div // 2 + Lpad - Lpad // 2
    return ypad1, ypad2, xpad1, xpad2


def _make_tiles(imgi, bsize=224, tile_overlap=0.1):
    nchan, Ly, Lx = imgi.shape
    tile_overlap = min(0.5, max(0.05, tile_overlap))
    bsizeY, bsizeX = min(bsize, Ly), min(bsize, Lx)
    ny = 1 if Ly <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Ly / bsize))
    nx = 1 if Lx <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Lx / bsize))
    ystart = np.linspace(0, Ly - bsizeY, ny).astype(int)
    xstart = np.linspace(0, Lx - bsizeX, nx).astype(int)
    ysub, xsub = [], []
    IMG = np.zeros((len(ystart), len(xstart), nchan, bsizeY, bsizeX), np.float32)
    for j in range(len(ystart)):
        for i in range(len(xstart)):
            ysub.append([ystart[j], ystart[j] + bsizeY])
            xsub.append([xstart[i], xstart[i] + bsizeX])
            IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1], xsub[-1][0]:xsub[-1][1]]
    return IMG, ysub, xsub, Ly, Lx


def _average_tiles(y, ysub, xsub, Ly, Lx):
    Navg = np.zeros((Ly, Lx))
    yf = np.zeros((y.shape[1], Ly, Lx), np.float32)
    mask = _taper_mask(ly=y.shape[-2], lx=y.shape[-1])
    for j in range(len(ysub)):
        yf[:, ysub[j][0]:ysub[j][1], xsub[j][0]:xsub[j][1]] += y[j] * mask
        Navg[ysub[j][0]:ysub[j][1], xsub[j][0]:xsub[j][1]] += mask
    yf /= Navg
    return yf


def _resize(img, Ly, Lx):
    return cv2.resize(np.asarray(img, dtype=np.float32), (Lx, Ly), interpolation=cv2.INTER_LINEAR)


class DenoiseModel:
    """Cellpose-3 image restoration, reimplemented (no ``cellpose`` import).

    Args:
        model_type: one of ``MODEL_NAMES`` (default ``"denoise_cyto3"``).
        device: torch device or ``None``/``'auto'`` (→ cuda → mps → cpu).
        pretrained: an explicit weights path; overrides ``model_type`` download.
    """

    def __init__(self, model_type="denoise_cyto3", device=None, pretrained=None, compile=False):
        self.model_type = model_type
        self.device = torch.device(resolve_device(device))
        # nuclei models were trained at diam 17, cyto at 30 (cellpose DenoiseModel convention).
        self.diam_mean = 17. if "nuclei" in model_type else 30.
        self.net = CPnet(nbase=[1, 32, 64, 128, 256], nout=1, sz=3,
                         diam_mean=self.diam_mean).to(self.device)
        wpath = pretrained if pretrained else _weight_path(model_type)
        state = torch.load(wpath, map_location=self.device, weights_only=True)
        # strict=False mirrors cellpose load_model (tolerates the diam_* param bookkeeping).
        self.net.load_state_dict(state, strict=False)
        self.net.eval()
        # torch.compile: tiles are a fixed (bsize x bsize) shape → good compile candidate. Opt-in
        # (first call pays a one-off compile cost); pairs with autocast on CUDA. No-op on failure.
        if compile:
            try:
                self.net = torch.compile(self.net)
            except Exception:
                pass

    def _forward(self, imgs, batch_size, autocast):
        """Run the net over a stack of already-normalized, already-rescaled planes.

        ``imgs``: (Lz, Ly, Lx, 1) float32.  Returns (Lz, Ly, Lx, 1) float32.
        Mirrors cellpose ``core.run_net`` (pad → tile → forward → taper-average → crop).
        """
        Lz, Ly0, Lx0, nchan = imgs.shape
        ypad1, ypad2, xpad1, xpad2 = _get_pad_yx(Ly0, Lx0)
        pads = np.array([[0, 0], [ypad1, ypad2], [xpad1, xpad2]])
        Ly, Lx = Ly0 + ypad1 + ypad2, Lx0 + xpad1 + xpad2
        bsize = 224
        ny = 1 if Ly <= bsize else int(np.ceil((1. + 2 * 0.1) * Ly / bsize))
        nx = 1 if Lx <= bsize else int(np.ceil((1. + 2 * 0.1) * Lx / bsize))
        ntiles = ny * nx
        nimgs = max(1, batch_size // ntiles)  # planes per batch
        yf = np.zeros((Lz, self.net.nout, Ly, Lx), np.float32)

        use_amp = autocast and self.device.type == "cuda"  # Decision 5: CUDA-only mixed precision
        for k in range(int(np.ceil(Lz / nimgs))):
            inds = np.arange(k * nimgs, min(Lz, (k + 1) * nimgs))
            tiles = []
            per = []
            for b in inds:
                imgb = np.pad(imgs[b].transpose(2, 0, 1), pads, mode="constant")
                IMG, ysub, xsub, Lyt, Lxt = _make_tiles(imgb, bsize=bsize, tile_overlap=0.1)
                ly, lx = IMG.shape[-2:]
                tiles.append((np.reshape(IMG, (ny * nx, nchan, ly, lx)), ysub, xsub, Lyt, Lxt))
                per.append(ny * nx)
            IMGa = np.concatenate([t[0] for t in tiles], axis=0)
            ya = np.zeros((IMGa.shape[0], self.net.nout, IMGa.shape[-2], IMGa.shape[-1]), np.float32)
            for j in range(0, IMGa.shape[0], batch_size):
                sl = slice(j, min(j + batch_size, IMGa.shape[0]))
                X = torch.from_numpy(IMGa[sl]).to(self.device, dtype=torch.float32)
                with torch.no_grad():
                    if use_amp:
                        with torch.autocast(device_type="cuda"):
                            out = self.net(X)[0]
                    else:
                        out = self.net(X)[0]
                ya[sl] = out.float().detach().cpu().numpy()
            off = 0
            for i, b in enumerate(inds):
                yfi = _average_tiles(ya[off:off + per[i]], tiles[i][1], tiles[i][2],
                                     tiles[i][3], tiles[i][4])
                yf[b] = yfi[:, :Ly, :Lx]
                off += per[i]
        yf = yf[:, :, ypad1:Ly - ypad2, xpad1:Lx - xpad2]
        return yf.transpose(0, 2, 3, 1)

    def eval(self, x, diameter=None, normalize=True, batch_size=8, tile_overlap=0.1,
             bsize=224, autocast=False):
        """Restore a 2-D plane ``(Y, X)`` or a stack ``(Z, Y, X)`` of grayscale planes.

        Each plane is normalized (1/99 percentile) and restored independently — matching how
        cecelia's cleanup task feeds planes. Returns float32 of the same spatial shape
        (Cellpose restoration range ~[-1, 10]).
        """
        x = np.asarray(x)
        squeeze = x.ndim == 2
        if squeeze:
            x = x[np.newaxis]           # (1, Y, X)
        Lz, Ly0, Lx0 = x.shape

        rescale = 1.0
        if diameter is not None and diameter > 0:
            rescale = self.diam_mean / diameter

        planes = np.empty((Lz, Ly0, Lx0), np.float32)
        for z in range(Lz):
            p = x[z]
            if normalize and np.ptp(p) > 0:
                p = normalize99(p)
            else:
                p = p.astype(np.float32, copy=True)
            planes[z] = p
        if rescale != 1.0:
            Lyr, Lxr = int(Ly0 * rescale), int(Lx0 * rescale)
            planes = np.stack([_resize(planes[z], Lyr, Lxr) for z in range(Lz)])

        out = self._forward(planes[..., np.newaxis], batch_size, autocast)  # (Lz,Ly,Lx,1)
        out = out[..., 0]
        if rescale != 1.0:
            out = np.stack([_resize(out[z], Ly0, Lx0) for z in range(Lz)])
        return out[0] if squeeze else out


def denoise_image(arr, model="denoise_cyto3", diameter=None, device=None, normalize=True,
                  batch_size=8, autocast=False):
    """One-shot restoration. See :class:`DenoiseModel`. ``arr`` is ``(Y,X)`` or ``(Z,Y,X)``."""
    return DenoiseModel(model_type=model, device=device).eval(
        arr, diameter=diameter, normalize=normalize, batch_size=batch_size, autocast=autocast)


# ================================ ratio-preserving restoration =================================
#
# Confetti identity is the RATIO between channels, so restoring each channel independently is the
# one thing a confetti pipeline must not do: three channels get three different NONLINEAR
# corrections, and the ratio they encode is gone. Per-channel restoration agrees with the raw
# dominant channel on only ~78% of cell PIXELS, and no un-mapping recovers it (normalising all
# channels through one shared window instead of per-plane reaches ~81% — the wrapper was never the
# problem).
#
# Restoring the PROJECTION once and applying it back as a per-pixel SCALAR gain sidesteps this: a
# scalar cancels in a ratio, so identity is preserved by construction, and the segmentation path
# still receives the denoised projection it consumes anyway.
#
# Scope of "by construction": exact PER PIXEL. Colour integrated over a cell MASK is a gain-
# weighted mean, and channels are not distributed identically inside a mask, so it shifts a little
# — measured 99.8% dominant-channel agreement over 1916 segmented cells, against 95.1% for
# per-channel restoration. Absolute intensity is NOT preserved either way (the gain is what
# changes it); anything measuring absolute brightness should read the raw store.
#
# See docs/todo/DENOISE_PLAN.md -> "Why cellpose denoise is wrong for confetti".

# Ceiling on restored/raw. It binds on ~0.5% of pixels — those where the projection sits at the
# noise floor, so the ratio runs away (measured p99.9 = 1.8e4, max 2.6e4 over 6 conditions). Not a
# tuning knob: swept 1.0 -> 4.0 on 3 movies x 2 z-planes, seed recall is flat (~89% at prob 0.6)
# at every value, and everything >= 1.3 is numerically indistinguishable. What a LOW cap costs is
# speckle — blob count rises 142 (cap 1.3+) -> 298 (cap 1.0) at equal recall, because clamping the
# gain to 1 leaves the noise floor un-suppressed.
GAIN_CAP = 4.0
_GAIN_FLOOR = 1e-3  # divisor floor, for pixels that are exactly zero.


def ratio_preserving_gain(arr, restored, channels=None, channel_axis=0, cap=GAIN_CAP):
    """Apply a restored projection back to a multi-channel image as a per-pixel scalar gain.

    Pure array algebra — no model, no weights. Split out from :func:`denoise_preserving_ratio` so
    the property that matters (channel ratios survive) is testable on its own, and so ``restored``
    can come from anything.

    That second point is not hypothetical. Ablating the restorer on 4 movies x 2 z-planes (prob
    0.6, recall against raw-grey seeds) puts most of the win in "smooth the projection" rather
    than in the net: raw 84.5% recall / 596 blobs, ``gaussian_filter(proj, 1)`` 87.5% / 158,
    ``denoise_cyto3`` 88.2% / 138. The net earns +0.7pp and ~13% fewer blobs over a free gaussian
    — real, but small enough that a caller without weights should reach for a gaussian, not go
    without.

    Args:
        arr:          multi-channel image, e.g. ``(C, Y, X)`` or ``(T, C, Y, X)``.
        restored:     the restored projection, shaped like ``arr`` with the channel axis dropped.
        channels:     which channels carry the signal being restored (default: all). Channels
                      outside this list are passed through untouched — e.g. a second-harmonic
                      channel that is not part of the confetti ratio.
        channel_axis: axis of ``arr`` holding channels.
        cap:          ceiling on the gain. See :data:`GAIN_CAP`.

    Returns:
        float32 array shaped like ``arr``. Within ``channels``, every pixel is the input scaled by
        one shared factor, so ``out[i] / out[j] == arr[i] / arr[j]`` everywhere the gain is
        non-zero — including where ``cap`` binds, since clipping changes the shared factor, not the
        ratio. What ``cap`` does cost is projection fidelity: ``out.mean(channels)`` equals
        ``restored`` only where the cap is slack.
    """
    a = np.moveaxis(np.asarray(arr, dtype=np.float32), channel_axis, 0)
    sel = list(range(a.shape[0])) if channels is None else list(channels)
    if not sel:
        raise ValueError("channels must select at least one channel")
    proj = a[sel].mean(axis=0)
    restored = np.asarray(restored, dtype=np.float32)
    if restored.shape != proj.shape:
        raise ValueError(f"restored {restored.shape} does not match the projection {proj.shape}")
    gain = np.clip(restored / np.maximum(proj, _GAIN_FLOOR), 0.0, cap)
    out = a.copy()
    out[sel] = a[sel] * gain
    return np.moveaxis(out, 0, channel_axis)


def gaussian_restorer(sigma=1.0):
    """Spatial restorer: a plane-wise Gaussian. No weights, no download, no net. The default,
    because it is the only one valid for *any* input shape — see ``temporal_mean_restorer`` for
    what to use when the leading axis really is time.

    Chosen on measurement, not on principle. Ablating restorers inside the gain wrapper (4 movies
    x 2 z-planes, recall against raw-grey seeds): raw 84.5% recall / 596 blobs, ``sigma=1``
    87.5% / 158, ``denoise_cyto3`` 88.2% / 138. The Cellpose net wins by 0.7pp — not enough to
    make a weights download and a vendored CPnet the default path for confetti data, which is
    what this whole exercise was trying to get away from.
    """
    def restore(proj):
        sig = (0,) * (proj.ndim - 2) + (sigma, sigma)
        return gaussian_filter(proj, sigma=sig)
    return restore


def temporal_mean_restorer(window=3):
    """Temporal restorer: a running mean along axis 0. **The best measured option for movies.**

    Requires the leading axis of the projection to be TIME. Not the default only because
    :func:`denoise_preserving_ratio` also accepts single timepoints, where this is meaningless.

    At matched foreground area (4 movies x 2 z-planes x 3 frames) it beats every spatial option:
    raw 83.9% recall at 3% area, ``gaussian_restorer(1)`` 89.3%, ``temporal_mean_restorer(3)``
    **92.3%**. Wrapped in the gain, identity stays at 99.5% against 97.6% for averaging the
    channels directly — the same segmentation, 1.9pp more identity, for free.

    Note the window is small on purpose. Motion-compensating the frames first (warping them into
    alignment before averaging) scores *worse* here, not better — 87.4% at 3% area — because this
    segmenter is flow-supervised, and alignment deletes the inter-frame motion the UNet is
    conditioned on. See docs/todo/DENOISE_PLAN.md -> B2.
    """
    def restore(proj):
        if proj.ndim < 3:
            raise ValueError("temporal_mean_restorer needs a time axis; got a single plane")
        return uniform_filter1d(proj.astype(np.float32), size=window, axis=0, mode="nearest")
    return restore


def cellpose_restorer(model="denoise_cyto3", device=None, diameter=None, batch_size=8,
                      autocast=False, _model=None):
    """Opt-in Cellpose-3 restoration as the projection restorer.

    Worth the extra 0.7pp only when that margin matters; :func:`gaussian_restorer` is the default.
    ``denoise_cyto2`` scores within noise of ``cyto3``; ``denoise_nuclei`` fails outright on cell
    data, flooding 60% of the frame with foreground.
    """
    net = _model if _model is not None else DenoiseModel(model_type=model, device=device)

    def restore(proj):
        flat = proj.reshape(-1, *proj.shape[-2:])
        out = np.empty_like(flat)
        for i, plane in enumerate(flat):
            lo, hi = np.percentile(plane, 1), np.percentile(plane, 99)
            if hi <= lo:                            # flat plane (e.g. drift padding)
                out[i] = plane
                continue
            # normalize99 is (X - p1) / (p99 - p1) — a known affine, so undoing it on the output
            # is exact and lands the result back in the input's units, where the gain is ~1.
            r = net.eval(plane, diameter=diameter, normalize=True,
                         batch_size=batch_size, autocast=autocast)
            out[i] = r * (hi - lo) + lo
        return out.reshape(proj.shape)
    return restore


def denoise_preserving_ratio(arr, channels=None, channel_axis=0, restorer=None, cap=GAIN_CAP):
    """Denoise a multi-channel image without disturbing the ratios between its channels.

    The mean projection over ``channels`` is restored plane by plane, then applied back to every
    selected channel as a per-pixel scalar via :func:`ratio_preserving_gain` — so the ratio that
    carries confetti identity comes through untouched.

    Args:
        arr:          ``(C, Y, X)`` or ``(T, C, Y, X)`` — any leading axes before the channel axis.
        channels:     which channels to denoise (default all); others pass through untouched.
        channel_axis: axis of ``arr`` holding channels.
        restorer:     ``projection -> restored projection``, same shape. Gets the WHOLE projection
                      (not one plane at a time) so temporal restorers can see the time axis.
                      Defaults to :func:`gaussian_restorer`. **For movies, prefer
                      ``temporal_mean_restorer(3)``** — measurably better and the leading axis is
                      time. :func:`cellpose_restorer` is available but earns little.
        cap:          see :data:`GAIN_CAP`.

    Returns float32 in the input's intensity units.
    """
    a = np.moveaxis(np.asarray(arr, dtype=np.float32), channel_axis, 0)
    sel = list(range(a.shape[0])) if channels is None else list(channels)
    proj = a[sel].mean(axis=0)                      # (…, Y, X)

    restore = restorer if restorer is not None else gaussian_restorer()
    restored = np.asarray(restore(proj), dtype=np.float32)

    return ratio_preserving_gain(arr, restored, channels=channels,
                                 channel_axis=channel_axis, cap=cap)
