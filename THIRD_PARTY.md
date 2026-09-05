# Third-party code and models

## Cellpose 3 — CPnet architecture + restoration inference harness

`coastal/denoise.py` reimplements the **CPnet** network and the normalization / tiling / stitching
inference harness from **Cellpose 3**, and loads Cellpose's **public pretrained restoration
weights** (the **denoise** models — `denoise_cyto3`, `denoise_cyto2`, `denoise_nuclei`), downloaded
from <https://www.cellpose.org/models> on first use. (Cellpose 3 also publishes deblur/upsample
restoration weights; coastal ships denoise only.)

- **Upstream:** Cellpose — <https://github.com/MouseLand/cellpose>
- **License:** BSD-3-Clause
- **Copyright:** © 2023 Howard Hughes Medical Institute. Authored by Carsen Stringer and Marius
  Pachitariu.
- **Weights version:** trained/distributed with **cellpose 3.1.1.2**.
- **Adaptation:** architecture attribute names are preserved so the upstream `state_dict` loads
  unchanged; only the inference path is reproduced (no augmentation / 3-D conv / mkldnn / training).
  The `cellpose` package is **not** a runtime dependency — it is imported only by
  `tests/test_denoise.py` (a golden-value parity test, skipped if absent).

### Method lineage (upstream attribution to preserve)

Cellpose 3 image restoration builds on prior self-supervised / content-aware restoration work:

- **CARE / CSBDeep** — Weigert et al., "Content-aware image restoration: pushing the limits of
  fluorescence microscopy", *Nature Methods* 2018. <https://doi.org/10.1038/s41592-018-0216-7>
- **Noise2Void** — Krull et al., CVPR 2019. <https://doi.org/10.1109/CVPR.2019.00223>
- **Noise2Self** — Batson & Royer, ICML 2019.

## SUPPORT — self-supervised temporal blind-spot denoiser (vendored)

`coastal/support/` vendors the **SUPPORT** self-supervised temporal blind-spot denoiser from
NICALab. The model predicts the centre frame of an `input_frames`-wide temporal window from the
surrounding frames with the centre itself masked out. Used for photon-limited intravital 2P
denoising; cecelia's `cleanupImages.denoise` task drives it through `coastal.support.denoise_stack`
and `coastal.support.train_support`.

- **Upstream:** SUPPORT — <https://github.com/NICALab/SUPPORT>
- **Paper:** Eom et al., "Statistically unbiased prediction enables accurate denoising of
  voltage imaging data", *Nature Methods* 20:1581-1588 (2023).
  <https://doi.org/10.1038/s41592-023-02005-8>
- **License:** GPL-3.0 (compatible with coastal's GPL-3.0-or-later)
- **Copyright:** © NICALab, KAIST.
- **Vendored — not `pip install`-ready.** Upstream ships no PyPI wheel, and upstream response
  times are slow ([issue #25](https://github.com/NICALab/SUPPORT/issues/25)), so we own the copy
  rather than track upstream at runtime.
- **What is vendored:** `_model.py` (SUPPORT network), `_convhole.py` (blind-spot conv primitive),
  `_dataset.py` (DatasetSUPPORT / DatasetSUPPORT_test_stitch / random_transform / normalize /
  get_coordinate). The upstream training CLI and pooled-dataloader factory are dropped because we
  drive training through the `train_support` wrapper. Commit pin + local edits documented in
  `coastal/support/VENDORED.md`.
- **Migrated to coastal from cecelia (2026-09-05).** Formerly at
  `cecelia/python/cecelia/vendor/support/`. Move brings SUPPORT in line with CPnet — algorithms
  live in coastal, cecelia is a thin orchestration layer.

### Cellpose BSD-3-Clause license

```
Copyright © 2023 Howard Hughes Medical Institute

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions
   and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of
   conditions and the following disclaimer in the documentation and/or other materials provided
   with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific prior written
   permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
