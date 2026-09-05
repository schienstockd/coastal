"""SUPPORT — self-supervised temporal blind-spot denoiser.

Verbatim third-party algorithm from NICALab (Eom et al., *Nature Methods* 2023,
`DOI 10.1038/s41592-023-02005-8`). The upstream repository is `NICALab/SUPPORT` and is not
`pip install`-ready; the model architecture, blind-spot conv primitive and patch dataset are
vendored under this subpackage — pinned commit + local edits in `VENDORED.md`. License GPL-3
(compatible with coastal's GPL-3-or-later).

**Why here rather than in a caller.** SUPPORT is a denoising algorithm; algorithms live in coastal
(same reasoning that put `CPnet` inference in `coastal.denoise` when the Cellpose-3 restoration
work migrated). This subpackage exposes two thin wrappers so callers stay array-only:

    train_support(volumes, arch, epochs, batch_size, lr, on_progress, on_log) -> state_dict, losses
    denoise_stack(arr_tzyx, state_dict, arch, batch_size, on_progress, on_log) -> denoised_tzyx

The wrappers own the training loop and mirror-padded inference — the two knobs the SUPPORT paper
gets wrong when you drive it naively (opaque `CUDA out of memory` on training, init-zero frames at
the boundaries on inference). Callers (cecelia's `cleanupImages.denoise` task and its trainer) just
supply pixels and a manifest, and take pixels back.

State-dict + `arch` dict are the shipping contract — the SAME `arch` is written to the manifest
beside the `.pt` at training time and read back to reconstruct the network at inference. Keep both
wrappers self-consistent about which keys they read.
"""
from ._model import SUPPORT
from ._dataset import (
    DatasetSUPPORT,
    DatasetSUPPORT_test_stitch,
    random_transform,
    normalize,
    get_coordinate,
)
from ._support import train_support, denoise_stack, build_model, DEFAULT_ARCH

__all__ = [
    "SUPPORT",
    "DatasetSUPPORT",
    "DatasetSUPPORT_test_stitch",
    "random_transform",
    "normalize",
    "get_coordinate",
    "train_support",
    "denoise_stack",
    "build_model",
    "DEFAULT_ARCH",
]
