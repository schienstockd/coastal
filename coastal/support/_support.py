"""SUPPORT training + inference wrappers — the pure-array API `coastal.support` exposes.

`train_support` and `denoise_stack` are the two entry points a caller uses. They own the training
loop, the mirror-padded inference and the arch dict — the two failure modes callers repeatedly hit
when driving the raw SUPPORT modules (untended `CUDA OOM` mid-epoch, init-zero boundary frames)
are fixed here in ONE place so every consumer inherits the same correct behaviour.

The wrappers do NOT touch disk. Reading OME-ZARR volumes, writing `.pt` files or their sidecar
manifests, and progress reporting to a task rail all live in the caller. This is the same shape as
`coastal.segment.LearnedAffinityInference` — pure arrays in, pure arrays out — so a notebook, a
cecelia task and a standalone script use the same entry point.
"""
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ._model import SUPPORT
from ._dataset import DatasetSUPPORT, DatasetSUPPORT_test_stitch, random_transform


# The arch dict is the shipping contract with the caller's manifest — inference reads these keys
# back verbatim to reconstruct the network. Keep the key set exactly aligned with `build_model`.
DEFAULT_ARCH = dict(
    inputFrames=61,
    patchXY=128,
    midChannels=[64, 128, 256, 512],
    depth=4,
    blindConvChannels=64,
    oneByOneChannels=[32, 16],
    lastLayerChannels=[64, 32, 16],
    bsSize=[3, 3],
    bp=False,
)


def _noop_progress(done: int, total: int) -> None:  # pragma: no cover - trivial default
    pass


def _noop_log(msg: str) -> None:  # pragma: no cover - trivial default
    pass


def build_model(arch: dict, device: torch.device) -> SUPPORT:
    """Instantiate SUPPORT from an `arch` dict on the target device.

    The `arch` dict is the same one written to the training manifest — this function is the ONE
    place that maps its keys to `SUPPORT(...)` kwargs, so trainer + inference stay in sync when a
    new arch knob appears.
    """
    return SUPPORT(
        in_channels=int(arch.get('inputFrames', DEFAULT_ARCH['inputFrames'])),
        mid_channels=list(arch.get('midChannels', DEFAULT_ARCH['midChannels'])),
        depth=int(arch.get('depth', DEFAULT_ARCH['depth'])),
        blind_conv_channels=int(arch.get('blindConvChannels', DEFAULT_ARCH['blindConvChannels'])),
        one_by_one_channels=list(arch.get('oneByOneChannels', DEFAULT_ARCH['oneByOneChannels'])),
        last_layer_channels=list(arch.get('lastLayerChannels', DEFAULT_ARCH['lastLayerChannels'])),
        bs_size=list(arch.get('bsSize', DEFAULT_ARCH['bsSize'])),
        bp=bool(arch.get('bp', DEFAULT_ARCH['bp'])),
    ).to(device)


def train_support(
    volumes: list,
    arch: dict,
    epochs: int = 20,
    batch_size: int = 2,
    lr: float = 5e-4,
    device: Optional[torch.device] = None,
    on_progress: Callable[[int, int], None] = _noop_progress,
    on_log: Callable[[str], None] = _noop_log,
    seed: int = 0,
):
    """Train SUPPORT on a POOLED list of `[T, Y, X]` float tensors.

    `volumes` is a flat list — the caller decides which channels / Z planes to pool (per fXgbTl
    2026-09-05 measurement, a pooled model beats per-channel specialists on intravital imaging,
    so the "one model per set" UX is the default).

    Args:
        volumes: list of `torch.FloatTensor` of shape `[T, Y, X]`, T ≥ `arch['inputFrames']`.
        arch: architecture dict (see `DEFAULT_ARCH`).
        epochs: training passes over the pooled patches.
        batch_size: patches per gradient step (lower if OOM).
        lr: Adam learning rate.
        device: torch device to train on (defaults to CUDA if available, else CPU).
        on_progress: `(done, total)` per-batch progress callback (`total = epochs * batches_per_epoch`).
        on_log: `str` log-line callback.
        seed: RNG seed for the augmentation transform.

    Returns:
        (state_dict, epoch_losses): the trained network's state_dict and one final loss per epoch.
    """
    if not volumes:
        raise ValueError('train_support: no volumes to train on')

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_frames = int(arch['inputFrames'])
    patch_xy = int(arch['patchXY'])

    train_ds = DatasetSUPPORT(
        volumes,
        patch_size=[input_frames, patch_xy, patch_xy],
        patch_interval=[5, patch_xy // 2, patch_xy // 2],
        load_to_memory=True,
    )
    train_ds.precompute_indices()
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=(device.type == 'cuda'))
    on_log(f'>> {len(train_ds)} patches per epoch')

    model = build_model(arch, device)
    n_params = sum(p.numel() for p in model.parameters())
    on_log(f'>> model params: {n_params / 1e6:.2f}M')

    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    L1 = torch.nn.L1Loss()
    L2 = torch.nn.MSELoss()
    rng = np.random.default_rng(seed)

    # One tick per BATCH — the operational unit matches coastal.train and cecelia's smooth_run.
    total = epochs * max(1, len(train_dl))
    done = 0
    on_progress(done, total)

    epoch_losses = []
    for ep in range(epochs):
        model.train()
        train_ds.precompute_indices()
        losses = []
        for noisy_image, _, _ in train_dl:
            noisy_image = noisy_image.to(device)
            noisy_image, _ = random_transform(noisy_image, None, rng, True)
            T = noisy_image.size(1)
            target = noisy_image[:, T // 2, :, :].unsqueeze(1)
            optim.zero_grad()
            out = model(noisy_image)
            loss = 0.5 * L1(out, target) + 0.5 * L2(out, target)
            loss.backward()
            optim.step()
            losses.append(loss.item())
            done += 1
            on_progress(done, total)
        ep_loss = float(np.mean(losses))
        epoch_losses.append(ep_loss)
        on_log(f'   epoch {ep + 1}/{epochs}: loss {ep_loss:.4f}')

    on_progress(total, total)
    return model.state_dict(), epoch_losses


def _mirror_pad_time(arr: np.ndarray, pad_t: int) -> np.ndarray:
    """Mirror-pad the time axis by `pad_t` on both sides so every original frame becomes a centre.

    Skips index 0/-1 in the reflection so the boundary frame is not duplicated (`arr[1:pad+1]`
    reversed rather than `arr[:pad]` reversed). Without this, SUPPORT returns init-zero frames at
    the first and last `pad_t` timepoints — verified 2026-09-05 on the DENOISE_INTEGRATION eval.
    """
    if pad_t <= 0:
        return arr
    head = arr[1:pad_t + 1][::-1]
    tail = arr[-pad_t - 1:-1][::-1]
    return np.concatenate([head, arr, tail], axis=0)


def _denoise_one_plane(model, arr_tyx: np.ndarray, pad_t: int, patch_xy: int,
                       batch_size: int, device: torch.device) -> np.ndarray:
    """Denoise one `[T, Y, X]` sub-volume and return the same shape.

    Mirror-pads T so every real frame has an `input_frames`-wide centred window; strips the pad
    off the returned array. De-normalises using the stitching dataset's own mean/std.
    """
    padded = _mirror_pad_time(arr_tyx, pad_t)
    padded_t = torch.from_numpy(padded.astype(np.float32)).float()

    input_frames = pad_t * 2 + 1
    test_ds = DatasetSUPPORT_test_stitch(
        padded_t,
        patch_size=[input_frames, patch_xy, patch_xy],
        patch_interval=[1, patch_xy // 2, patch_xy // 2],
        load_to_memory=True,
    )
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=(device.type == 'cuda'))
    denoised = np.zeros(test_ds.noisy_image.shape, dtype=np.float32)

    with torch.no_grad():
        for noisy_image, _, coord in test_dl:
            noisy_image = noisy_image.to(device)
            out = model(noisy_image)
            T = noisy_image.size(1)
            for bi in range(noisy_image.size(0)):
                sw0 = int(coord['stack_start_w'][bi]); sw1 = int(coord['stack_end_w'][bi])
                pw0 = int(coord['patch_start_w'][bi]); pw1 = int(coord['patch_end_w'][bi])
                sh0 = int(coord['stack_start_h'][bi]); sh1 = int(coord['stack_end_h'][bi])
                ph0 = int(coord['patch_start_h'][bi]); ph1 = int(coord['patch_end_h'][bi])
                s0 = int(coord['init_s'][bi])
                denoised[s0 + (T // 2), sh0:sh1, sw0:sw1] = \
                    out[bi].squeeze()[ph0:ph1, pw0:pw1].cpu().numpy()

    denoised = denoised * test_ds.std_image.item() + test_ds.mean_image.item()
    return denoised[pad_t:pad_t + arr_tyx.shape[0]]


def denoise_stack(
    arr_tyx: np.ndarray,
    state_dict: dict,
    arch: dict,
    batch_size: int = 2,
    device: Optional[torch.device] = None,
    on_progress: Callable[[int, int], None] = _noop_progress,
    on_log: Callable[[str], None] = _noop_log,
) -> np.ndarray:
    """Denoise a `[T, Y, X]` stack using a trained SUPPORT model.

    Args:
        arr_tyx: input array of shape `[T, Y, X]` (any float-castable dtype).
        state_dict: state_dict returned by `train_support` or loaded from a `.pt`.
        arch: the `arch` dict the model was trained under (SAME keys as `DEFAULT_ARCH`).
        batch_size: patches per forward pass (lower if OOM).
        device: torch device (defaults to CUDA if available, else CPU).
        on_progress: `(done, total)` progress callback — one tick per stack completed
                     (single-stack call = 0/1 → 1/1).
        on_log: `str` log-line callback.

    Returns:
        denoised `[T, Y, X]` numpy float32 array.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(arch, device)
    model.load_state_dict(state_dict)
    model.eval()

    input_frames = int(arch['inputFrames'])
    patch_xy = int(arch['patchXY'])
    pad_t = input_frames // 2

    on_progress(0, 1)
    out = _denoise_one_plane(model, arr_tyx.astype(np.float32, copy=False),
                             pad_t=pad_t, patch_xy=patch_xy,
                             batch_size=batch_size, device=device)
    on_progress(1, 1)
    return out
