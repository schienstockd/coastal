"""Training functions and datasets."""

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from coastal.model import UNetWithEmbeddings
from coastal.loss import (ConfettiBoundaryLoss, ConfettiForegroundLoss, ForegroundLoss,
                          flow_discontinuity,
                          IntensityLoss,
                          TemporalMetricsLoss, VarianceMetricsLoss, WarpConsistencyLoss)
from coastal.device import resolve_device


class TemporalDatasetWithAugmentation(Dataset):
    """Dataset with frame + temporal metrics + variance metrics as input."""

    def __init__(self, frames, temporal_metrics_norm, variance_metrics_norm=None,
                 flow_pairs=None, variance_as_input=True):
        """
        Args:
            frames: [T, H, W] grayscale frames (max/mean projection of multi-channel data)
            temporal_metrics_norm: list of T dicts (optical flow metrics)
            variance_metrics_norm: list of T dicts (cross-channel variance metrics), or None
            flow_pairs: list of T Optional[np.ndarray [H,W,2]] forward flow from frame t to
                        t+1 ([u=x-disp, v=y-disp] in pixels), or None at the last frame.
                        From extract_dense_flow_pairs(). Used for WarpConsistencyLoss.
            variance_as_input: also concatenate the variance metrics onto the model INPUT
                        (default True, the historical behaviour). Set False to use them for
                        supervision only — see `train_with_metrics`, which explains why that is
                        usually what you want: the production inference path cannot supply them,
                        so as input channels they are zeros at inference.
        """
        self.frames = frames
        self.temporal_metrics = temporal_metrics_norm
        self.variance_metrics = variance_metrics_norm or [{} for _ in range(len(frames))]
        self.flow_pairs = flow_pairs  # None = warp loss disabled
        self.variance_as_input = variance_as_input

    def __len__(self):
        return len(self.frames)

    def _stack_metrics(self, metrics_dict, frame_shape):
        metric_list = []
        for name in sorted(metrics_dict.keys()):
            arr = metrics_dict[name]
            tensor = arr.float() if isinstance(arr, torch.Tensor) else torch.from_numpy(arr).float()
            metric_list.append(tensor)
        if metric_list:
            return torch.stack(metric_list, dim=0)
        return torch.zeros(0, *frame_shape)

    def __getitem__(self, idx):
        raw = torch.from_numpy(self.frames[idx]).float()
        if raw.ndim == 3:  # [C, H, W] multi-channel → mean projection
            frame = raw.mean(dim=0, keepdim=True)  # [1, H, W]
        else:              # [H, W]
            frame = raw.unsqueeze(0)               # [1, H, W]

        t_metrics = self.temporal_metrics[idx] if idx < len(self.temporal_metrics) else {}
        v_metrics = self.variance_metrics[idx] if idx < len(self.variance_metrics) else {}

        t_stacked = self._stack_metrics(t_metrics, frame.shape[1:])
        v_stacked = self._stack_metrics(v_metrics, frame.shape[1:]) \
            if self.variance_as_input else torch.zeros(0, *frame.shape[1:])
        frame_and_metrics = torch.cat([frame, t_stacked, v_stacked], dim=0)

        item = {
            'frame_and_metrics': frame_and_metrics,
            'channels': frame,
            'temporal_metrics': t_metrics,
            'variance_metrics': v_metrics,
            'frame_idx': idx,
            'flow_uv': None,
            'frame_and_metrics_next': None,
        }

        # Warp consistency: also return the next frame and the flow connecting them
        if self.flow_pairs is not None and idx < len(self.flow_pairs):
            uv = self.flow_pairs[idx]
            if uv is not None and idx + 1 < len(self.frames):
                item['flow_uv'] = torch.from_numpy(
                    np.asarray(uv, dtype=np.float32)
                ).permute(2, 0, 1)  # [H,W,2] → [2,H,W]

                # Build frame_and_metrics for the next frame
                raw_next = torch.from_numpy(self.frames[idx + 1]).float()
                frame_next = raw_next.unsqueeze(0) if raw_next.ndim == 2 \
                    else raw_next.mean(dim=0, keepdim=True)
                t_next = self.temporal_metrics[idx + 1] \
                    if idx + 1 < len(self.temporal_metrics) else {}
                v_next = self.variance_metrics[idx + 1] \
                    if idx + 1 < len(self.variance_metrics) else {}
                t_next_s = self._stack_metrics(t_next, frame_next.shape[1:])
                v_next_s = self._stack_metrics(v_next, frame_next.shape[1:]) \
                    if self.variance_as_input else torch.zeros(0, *frame_next.shape[1:])
                item['frame_and_metrics_next'] = torch.cat(
                    [frame_next, t_next_s, v_next_s], dim=0
                )

        return item


def train_test_split(frames_prep, temporal_metrics_norm, train_ratio=0.8, shuffle=False):
    """
    Split frames and metrics into train/test sets (single movie/sequence).

    Args:
        frames_prep: [T, H, W] array
        temporal_metrics_norm: list of T metric dicts
        train_ratio: fraction for training (default 0.8)
        shuffle: whether to shuffle before splitting (default False)

    Returns:
        train_frames, test_frames, train_metrics, test_metrics
    """
    n_frames = len(frames_prep)
    n_train = int(n_frames * train_ratio)

    if shuffle:
        indices = np.random.permutation(n_frames)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
    else:
        train_idx = np.arange(n_train)
        test_idx = np.arange(n_train, n_frames)

    train_frames = frames_prep[train_idx]
    test_frames = frames_prep[test_idx]
    train_metrics = [temporal_metrics_norm[i] for i in train_idx]
    test_metrics = [temporal_metrics_norm[i] for i in test_idx]

    print(f"\nTrain/Test Split:")
    print(f"  Train frames: {len(train_frames)} ({train_ratio*100:.0f}%)")
    print(f"  Test frames:  {len(test_frames)} ({(1-train_ratio)*100:.0f}%)\n")

    return train_frames, test_frames, train_metrics, test_metrics


def train_test_split_per_movie(all_frames, all_metrics, train_ratio=0.8, shuffle=False):
    """
    Split train/test within each movie independently, then concatenate.

    Args:
        all_frames: list of frame arrays (one per movie)
        all_metrics: list of metric lists (one per movie)
        train_ratio: fraction for training per movie (default 0.8)
        shuffle: whether to shuffle within each movie (default False)

    Returns:
        train_frames: concatenated training frames [total_train, H, W]
        test_frames: concatenated test frames [total_test, H, W]
        train_metrics: concatenated training metrics (list of dicts)
        test_metrics: concatenated test metrics (list of dicts)
    """

    all_train_frames = []
    all_test_frames = []
    all_train_metrics = []
    all_test_metrics = []

    print(f"\n{'='*80}")
    print(f"TRAIN/TEST SPLIT WITHIN EACH MOVIE (train_ratio={train_ratio})")
    print(f"{'='*80}\n")

    for movie_idx, (frames, metrics) in enumerate(zip(all_frames, all_metrics)):
        n_frames = len(frames)
        n_train = int(n_frames * train_ratio)

        if shuffle:
            indices = np.random.permutation(n_frames)
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
        else:
            train_idx = np.arange(n_train)
            test_idx = np.arange(n_train, n_frames)

        train_frames = frames[train_idx]
        test_frames = frames[test_idx]
        train_metrics = [metrics[i] for i in train_idx]
        test_metrics = [metrics[i] for i in test_idx]

        all_train_frames.append(train_frames)
        all_test_frames.append(test_frames)
        all_train_metrics.extend(train_metrics)
        all_test_metrics.extend(test_metrics)

        print(f"Movie {movie_idx + 1}: {len(train_frames)} train + {len(test_frames)} test = {n_frames} total")

    # Concatenate across movies — fall back to flat list if spatial sizes differ
    try:
        train_frames_concat = np.concatenate(all_train_frames, axis=0)
        test_frames_concat = np.concatenate(all_test_frames, axis=0)
    except ValueError:
        train_frames_concat = [f for arr in all_train_frames for f in arr]
        test_frames_concat = [f for arr in all_test_frames for f in arr]
        print("Note: mixed spatial sizes detected — returning flat lists (use batch_size=1 or set target_size)")

    print(f"\n{'='*80}")
    print(f"COMBINED DATASET (all movies)")
    print(f"{'='*80}")
    print(f"Train: {len(train_frames_concat)} frames")
    print(f"Test:  {len(test_frames_concat)} frames")
    print(f"Total: {len(train_frames_concat) + len(test_frames_concat)} frames\n")

    return train_frames_concat, test_frames_concat, all_train_metrics, all_test_metrics


def train_with_metrics(frames_prep, temporal_metrics_norm, variance_metrics_norm=None,
                       num_epochs=50, batch_size=1, seed=42, device=None, embedding_dim=16,
                       variance_weight=1.0, intensity_weight=1.0, temporal_weight=2.0,
                       warp_weight=0.0, confetti_weight=0.0, confetti_blur_sigma=1.0,
                       boundary_weight=0.0,
                       foreground_weight=0.0, foreground_blur_sigma=1.0,
                       foreground_boundary_weight=0.0,
                       flow_pairs=None,
                       val_frames=None, val_temporal_metrics_norm=None,
                       val_variance_metrics_norm=None, val_flow_pairs=None,
                       variance_as_input=True,
                       max_grad_norm=1.0, variance_window_size=32, variance_dropout_p=0.5,
                       num_workers=4, use_amp=True):
    """
    Train embeddings with 3-loss design.

    Args:
        frames_prep: [T, H, W] or [T, C, H, W] training frames
        temporal_metrics_norm: list of T dicts (optical flow metrics)
        variance_metrics_norm: list of T dicts (cross-channel variance metrics), or None
        val_frames / val_temporal_metrics_norm / val_variance_metrics_norm / val_flow_pairs:
                          a HELD-OUT split, in the same shapes as the training ones. When given,
                          every term is also evaluated on it once per epoch (no grad, no
                          augmentation) and recorded as `val_*` in the history. Split with
                          `train_test_split_per_movie`, which splits WITHIN each movie so both
                          sides see every movie. Without this a loss curve cannot distinguish
                          convergence from memorising — it only ever says the number went down.
                          `val_flow_pairs` matters whenever `warp_weight > 0`: leave it out and
                          the warp term is zero on the held-out set, which drags `val_total`
                          below `total` for a reason that is not generalisation.
        num_epochs: training epochs
        batch_size: batch size
        seed: random seed
        device: cuda or cpu
        embedding_dim: embedding dimension
        variance_weight: weight for VarianceMetricsLoss (contrastive on variance, default 1.0)
        intensity_weight: weight for IntensityLoss (default 1.0). Its target is half a
                          per-pixel intensity threshold and half two edge detectors, so it
                          trains the prob head toward speckle — measured 2535 components
                          per frame, median 3 px. Turn it down when using confetti_weight.
        variance_as_input: concatenate the variance (confetti) metrics onto the model INPUT as well
                          as handing them to the losses (default True = historical behaviour).
                          **Prefer False.** The production inference path cannot supply them —
                          `TwoPassSegmentationInference.predict_frame` takes no `variance_metrics`
                          argument at all, and `LearnedAffinityInference` zero-fills any channel the
                          model expects beyond what was passed — so as input channels they are zeros
                          at inference while training sees them present 50% of the time per channel
                          (`variance_dropout_p`, applied as a plain mask with no inverted-dropout
                          rescale). All three zero at once is a 12.5% corner of the training
                          distribution but 100% of inference. Setting this False keeps confetti as
                          *supervision only*, which is all `ConfettiForegroundLoss`,
                          `ConfettiBoundaryLoss` and `VarianceMetricsLoss` need — they read
                          `variance_metrics_norm` directly. Measured cost of the mismatch on
                          zolIMa/fXgbTl: see docs/SEGMENTATION.md -> *What confetti actually
                          contributes*.
        foreground_weight: weight for ForegroundLoss (default 0.0 = off). The no-confetti prob-head
                          supervisor: brightness blurred to cell scale and p99-normalised. Use this
                          instead of confetti_weight on any data that is not confetti, and prefer it
                          to intensity_weight in general — it supplies the cell-scale shape prior
                          that IntensityLoss lacks. It needs no variance_metrics_norm, so the model
                          is built with no confetti input channels and there is no train/inference
                          mismatch. Measured to produce the same target as ConfettiForegroundLoss to
                          r >= 0.99 even ON confetti; see loss.ForegroundLoss.
        foreground_blur_sigma: cell-scale blur for that target (default 1.0 px), same dial as
                          confetti_blur_sigma.
        confetti_weight:  weight for ConfettiForegroundLoss (default 0.0 = off). Supervises
                          the prob head with "one confetti colour dominates here, brightly",
                          blurred to cell scale, instead of grayscale texture. Needs
                          variance_metrics_norm. See docs/SEGMENTATION.md.
        confetti_blur_sigma: cell-scale blur for that target (default 1.0 px). A merge↔split
                          dial rather than a separation knob — 1.0 wins on F1 and roughly
                          halves merges vs 2.0, but sharpening further shatters cells. See
                          the table in loss.ConfettiForegroundLoss.
        boundary_weight:  weight for ConfettiBoundaryLoss (default 0.0 = off). Pushes
                          embeddings apart across a confetti-colour boundary, which is the
                          one thing no other term supplies: measured on real touching cells,
                          embedding cosine across a different-colour contact is 0.920 against
                          0.945 within a cell, and segmentation merges 86.7% of such pairs.
                          Needs variance_metrics_norm. See docs/SEGMENTATION.md.
        temporal_weight: weight for TemporalMetricsLoss (default 2.0)
        max_grad_norm: gradient clipping threshold (default 1.0)
        variance_window_size: spatial window size for windowed variance contrastive loss (default 32)
        num_workers: DataLoader worker processes for background data loading (default 4, 0 = main process)
        use_amp: enable automatic mixed precision (float16) on CUDA for faster training (default True)

    Returns:
        model, history — `history[term]` per epoch, plus `history['val_' + term]` when a
        validation split was supplied.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = resolve_device(device)
    use_variance = variance_metrics_norm is not None
    use_amp = use_amp and device != 'cpu' and torch.cuda.is_available()

    n_temporal = len(next(iter(temporal_metrics_norm), {}))
    # `n_variance` is how many variance channels reach the model INPUT; the losses read
    # `variance_metrics_norm` directly and are unaffected by variance_as_input.
    n_variance = (len(next(iter(variance_metrics_norm), {}))
                  if use_variance and variance_as_input else 0)
    input_channels = 1 + n_temporal + n_variance

    print(f"\n{'='*80}")
    # Report exactly which losses are active, so a log line cannot misrepresent what was
    # optimised (the header used to omit the confetti term entirely).
    active = [n for n, w in (('INTENSITY', intensity_weight), ('TEMPORAL', temporal_weight),
                             ('VARIANCE', variance_weight), ('WARP', warp_weight),
                             ('CONFETTI', confetti_weight), ('FOREGROUND', foreground_weight),
                             ('BOUNDARY', boundary_weight)) if w > 0]
    n_losses = len(active)
    print(f"TRAINING: {' + '.join(active)} ({n_losses}-LOSS)")
    print(f"Loss: Intensity ({intensity_weight}) + Temporal ({temporal_weight}) + "
          f"Variance ({variance_weight}, window={variance_window_size}px, "
          f"dropout_p={variance_dropout_p}) + Warp ({warp_weight}) + "
          f"Confetti ({confetti_weight}, blur={confetti_blur_sigma}px) + "
          f"Foreground ({foreground_weight}, blur={foreground_blur_sigma}px, "
          f"flow_boundary={foreground_boundary_weight}) + "
          f"Boundary ({boundary_weight})")
    print(f"Gradient clipping: {max_grad_norm} | AMP: {use_amp} | Workers: {num_workers}")
    print(f"{'='*80}\n")

    dataset = TemporalDatasetWithAugmentation(
        frames_prep, temporal_metrics_norm,
        variance_metrics_norm if use_variance else None,
        flow_pairs=flow_pairs, variance_as_input=variance_as_input,
    )
    def collate_fn(batch):
        out = {
            'frame_and_metrics': torch.stack([b['frame_and_metrics'] for b in batch]),
            'channels': torch.stack([b['channels'] for b in batch]),
            'temporal_metrics': [b['temporal_metrics'] for b in batch],
            'variance_metrics': [b['variance_metrics'] for b in batch],
            'frame_idx': [b['frame_idx'] for b in batch],
        }
        # Warp fields — filter to items that have both flow and next frame
        warp_mask = [b['flow_uv'] is not None and b['frame_and_metrics_next'] is not None
                     for b in batch]
        if any(warp_mask):
            out['flow_uv'] = torch.stack(
                [b['flow_uv'] for b, m in zip(batch, warp_mask) if m]
            )
            out['frame_and_metrics_next'] = torch.stack(
                [b['frame_and_metrics_next'] for b, m in zip(batch, warp_mask) if m]
            )
            out['warp_mask'] = warp_mask
        else:
            out['flow_uv'] = None
            out['frame_and_metrics_next'] = None
            out['warp_mask'] = warp_mask
        return out

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=(device != 'cpu'),
    )

    # Same dataset class, same collate, same batch size — shuffle off, because a validation pass has
    # no reason to reorder and a stable order makes two runs comparable.
    val_loader = None
    if val_frames is not None and val_temporal_metrics_norm is not None and len(val_frames):
        val_dataset = TemporalDatasetWithAugmentation(
            val_frames, val_temporal_metrics_norm,
            val_variance_metrics_norm if use_variance else None,
            flow_pairs=val_flow_pairs, variance_as_input=variance_as_input,
        )
        # `num_workers=0` deliberately, where the training loader uses `num_workers`. Inheriting it
        # would keep a SECOND persistent worker pool alive for the whole run — doubling the worker
        # processes, each holding its own copy of the frames and metrics, to serve a set that is
        # typically a fifth the size and read forward-only once per epoch. The main thread can
        # prepare that; a duplicated dataset in memory is the kind of cost that only shows up on the
        # box with the big movies on it.
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
            num_workers=0, pin_memory=(device != 'cpu'),
        )
        print(f"Validation frames: {len(val_dataset)}")
        # `val_total` is only comparable with `total` if it is made of the SAME terms. Without flow
        # pairs the warp term is structurally zero on the held-out set while it is nonzero on the
        # training set, so `val_total` sits below `total` by `warp_weight * warp` for a reason that
        # has nothing to do with generalising — which is precisely the reading the val curve exists
        # to support. Say so rather than let the gap be misread as headroom.
        if warp_weight > 0 and val_flow_pairs is None:
            print("  ! no val_flow_pairs: warp is 0 on the held-out set, so val_total is NOT "
                  "comparable with total — compare the per-term curves instead")

    model = UNetWithEmbeddings(
        num_metrics=n_temporal + n_variance,
        num_frames=1,
        embedding_dim=embedding_dim,
    )
    model = model.to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input channels: 1 (frame) + {n_temporal} (temporal) + {n_variance} (variance, p_drop={variance_dropout_p}) = {input_channels}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Dataset size: {len(dataset)} frames\n")

    loss_intensity = IntensityLoss().to(device)
    loss_confetti = ConfettiForegroundLoss(blur_sigma=confetti_blur_sigma).to(device) \
        if confetti_weight > 0.0 else None
    loss_boundary = ConfettiBoundaryLoss().to(device) if boundary_weight > 0.0 else None
    loss_foreground = ForegroundLoss(blur_sigma=foreground_blur_sigma,
                                     boundary_weight=foreground_boundary_weight).to(device) \
        if foreground_weight > 0.0 else None
    loss_temporal = TemporalMetricsLoss().to(device)
    loss_variance = VarianceMetricsLoss(window_size=variance_window_size).to(device)
    loss_warp = WarpConsistencyLoss().to(device) if warp_weight > 0.0 else None

    optimizer = Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    TERMS = ('total', 'variance', 'intensity', 'temporal', 'warp', 'confetti', 'foreground',
             'boundary')
    history = {k: [] for k in TERMS}
    if val_loader is not None:
        history.update({f'val_{k}': [] for k in TERMS})

    def batch_losses(batch, training):
        """Every loss term for one batch, as tensors keyed like `history`.

        ONE implementation, called by the training loop and the validation pass alike. A separate
        eval-time copy is the classic way for a validation curve to drift from the thing it claims
        to measure — a term reweighted here and not there, and the two curves stop being comparable
        while still both going down.

        `training` gates only what must NOT happen at eval: the variance-channel dropout, which is
        augmentation. Everything else — the terms, the weights, the sum — is identical by
        construction.
        """
        frame_and_metrics = batch['frame_and_metrics'].to(device, non_blocking=True)
        channels = batch['channels'].to(device, non_blocking=True)
        v_metrics = batch['variance_metrics']
        t_metrics = batch['temporal_metrics']
        frame_indices = batch['frame_idx']

        # Channel dropout on variance input channels so the model learns to
        # function without them (inference uses zeros in those positions).
        if training and use_variance and n_variance > 0:
            B_cur = frame_and_metrics.shape[0]
            keep = torch.rand(B_cur, n_variance, 1, 1, device=device) > variance_dropout_p
            frame_and_metrics = frame_and_metrics.clone()
            frame_and_metrics[:, 1 + n_temporal:] *= keep.float()

        flow_uv_batch = batch.get('flow_uv')
        fm_next_batch = batch.get('frame_and_metrics_next')
        warp_mask     = batch.get('warp_mask', [])

        with torch.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
            # One encoder/decoder pass shared by all three losses.
            decoded = model.encode_decode(frame_and_metrics)
            pred_prob = model.prob_head(decoded)
            metric_emb = model.emb_head(decoded)

            l_intensity = loss_intensity(pred_prob, channels)
            l_confetti = loss_confetti(pred_prob, v_metrics) if loss_confetti is not None \
                else torch.tensor(0.0, device=device)
            l_boundary = loss_boundary(metric_emb, v_metrics) \
                if loss_boundary is not None else torch.tensor(0.0, device=device)
            # `channels` is the projected frame, so this needs no variance metrics at all.
            # The boundary term is the ONE path by which optical flow reaches the labels:
            # without it the prob head is supervised by brightness alone and flow only ever
            # entered as unsupervised input channels. See loss.flow_discontinuity.
            fg_boundary = None
            if loss_foreground is not None and foreground_boundary_weight > 0:
                bs = [flow_discontinuity(m, device=device) for m in t_metrics]
                if all(b is not None for b in bs):
                    fg_boundary = torch.stack(bs, dim=0)
            l_foreground = loss_foreground(pred_prob, channels, fg_boundary) \
                if loss_foreground is not None else torch.tensor(0.0, device=device)
            l_temporal = loss_temporal(metric_emb, t_metrics)
            l_variance = loss_variance(metric_emb, v_metrics, frame_indices=frame_indices) if use_variance else \
                torch.tensor(0.0, device=device)

            # Warp consistency: run model on next frame for batch items that have flow
            l_warp = torch.tensor(0.0, device=device)
            if loss_warp is not None and flow_uv_batch is not None and any(warp_mask):
                fm_next = fm_next_batch.to(device, non_blocking=True)
                flow_uv = flow_uv_batch.to(device, non_blocking=True)
                # Select the matching rows from frame t embeddings and prob
                mask_idx = [i for i, m in enumerate(warp_mask) if m]
                emb_t_sel   = metric_emb[mask_idx]
                prob_t_sel  = pred_prob[mask_idx]
                decoded_n   = model.encode_decode(fm_next)
                emb_t1_sel  = model.emb_head(decoded_n)
                l_warp = loss_warp(emb_t_sel, emb_t1_sel, flow_uv, prob_t_sel)

            total_loss = (intensity_weight * l_intensity +
                         temporal_weight * l_temporal +
                         variance_weight * l_variance +
                         warp_weight * l_warp +
                         confetti_weight * l_confetti +
                         foreground_weight * l_foreground +
                         boundary_weight * l_boundary)

        return {'total': total_loss, 'intensity': l_intensity, 'temporal': l_temporal,
                'variance': l_variance, 'warp': l_warp, 'confetti': l_confetti,
                'foreground': l_foreground, 'boundary': l_boundary}

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {k: 0.0 for k in TERMS}

        for batch_idx, batch in enumerate(dataloader):
            losses = batch_losses(batch, training=True)

            optimizer.zero_grad()
            scaler.scale(losses['total']).backward()

            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            for k in TERMS:
                epoch_losses[k] += losses[k].item()

        n = len(dataloader)
        for key in TERMS:
            epoch_losses[key] /= n
            history[key].append(epoch_losses[key])

        # Held-out pass. No grad, no augmentation, same terms and the same weights — so `val_total`
        # is comparable with `total` and the gap between them is the only thing that can tell
        # convergence from memorising. Without it a loss curve only ever says "it went down".
        #
        # The RNG state is snapshotted and put back afterwards, which is not incidental: iterating a
        # DataLoader draws its base seed from the GLOBAL torch stream, so the validation pass — with
        # no gradients and no augmentation of its own — would still shift every later shuffle and
        # every later dropout mask. Turning validation on would then hand you a DIFFERENT model, and
        # the curve would no longer describe the run you would have had without it. Measuring a run
        # must not change it.
        if val_loader is not None:
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            model.eval()
            val_losses = {k: 0.0 for k in TERMS}
            with torch.no_grad():
                for batch in val_loader:
                    losses = batch_losses(batch, training=False)
                    for k in TERMS:
                        val_losses[k] += losses[k].item()
            nv = max(1, len(val_loader))
            for key in TERMS:
                history[f'val_{key}'].append(val_losses[key] / nv)
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            warp_str = f" | warp={epoch_losses['warp']:.4f}" if warp_weight > 0 else ""
            conf_str = f" | conf={epoch_losses['confetti']:.4f}" if confetti_weight > 0 else ""
            fg_str = f" | fg={epoch_losses['foreground']:.4f}" if foreground_weight > 0 else ""
            bnd_str = f" | bnd={epoch_losses['boundary']:.4f}" if boundary_weight > 0 else ""
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"total={epoch_losses['total']:.4f} | "
                  f"int={epoch_losses['intensity']:.4f} | "
                  f"tmp={epoch_losses['temporal']:.4f} | "
                  f"var={epoch_losses['variance']:.4f}"
                  + warp_str + conf_str + fg_str + bnd_str)

    # Report every ACTIVE term. `intensity` is printed unconditionally above and is computed even at
    # weight 0, so a run with intensity_weight=0 shows a nonzero `int=` that contributes nothing to
    # `total` — read the weights in the header, not these numbers, to know what was optimised.
    print(f"\nFinal losses:")
    print(f"  Total:     {history['total'][-1]:.4f}"
          + (f"   (val {history['val_total'][-1]:.4f})" if val_loader is not None else ''))
    print(f"  Intensity: {history['intensity'][-1]:.4f}"
          f"{'' if intensity_weight > 0 else '   (weight 0 — not optimised)'}")
    print(f"  Temporal:  {history['temporal'][-1]:.4f}")
    print(f"  Variance:  {history['variance'][-1]:.4f}")
    if foreground_weight > 0:
        print(f"  Foreground:{history['foreground'][-1]:.4f}")
    if confetti_weight > 0:
        print(f"  Confetti:  {history['confetti'][-1]:.4f}")
    if boundary_weight > 0:
        print(f"  Boundary:  {history['boundary'][-1]:.4f}")
    if warp_weight > 0:
        print(f"  Warp:      {history['warp'][-1]:.4f}")
    print()

    return model, history


def save_model(model, path, metadata=None):
    """Save model weights and config to a .pt file.

    Args:
        model:    trained UNetWithEmbeddings instance
        path:     file path to save to (e.g. 'coastal_model.pt')
        metadata: optional dict of extra info to store (e.g. training params)
    """
    # Fall back to reading config from layer shapes for models instantiated
    # before embedding_dim/init_features were stored as attributes.
    embedding_dim = getattr(model, 'embedding_dim', model.emb_head.out_channels)
    init_features = getattr(model, 'init_features', model.encoders[0].conv.conv[0].out_channels)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_metrics': model.num_metrics,
            'num_frames': model.num_frames,
            'embedding_dim': embedding_dim,
            'init_features': init_features,
            'depth': model.depth,
        },
    }
    if metadata:
        checkpoint['metadata'] = metadata
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(path, device=None):
    """Load a model saved with save_model().

    Args:
        path:   path to the .pt file
        device: torch device; None/'auto' → cuda→mps→cpu (see coastal.device.resolve_device)

    Returns:
        model: UNetWithEmbeddings in eval mode
    """
    device = resolve_device(device)
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint['model_config']
    model = UNetWithEmbeddings(
        num_metrics=cfg['num_metrics'],
        num_frames=cfg.get('num_frames', 1),
        embedding_dim=cfg['embedding_dim'],
        init_features=cfg.get('init_features', 32),
        depth=cfg.get('depth', 3),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    if 'metadata' in checkpoint:
        print(f"Metadata: {checkpoint['metadata']}")
    print(f"Model loaded from {path}")
    return model


def extract_sequences_from_volume(volume, n_sequences=3, seq_len=20, random_seed=None):
    """Extract 2D training sequences from a 4D volume [T, C, Z, Y, X].

    Picks n_sequences z-slices evenly spaced across Z. For each z-slice
    a single random start timepoint is chosen, giving one sequence of
    seq_len consecutive frames from that z-plane.

    Args:
        volume:      [T, C, Z, Y, X] numpy array (or array-like / zarr)
        n_sequences: number of sequences to extract (= number of z-slices sampled)
        seq_len:     frames per sequence
        random_seed: integer seed for reproducibility (None = random)

    Returns:
        sequences:     list of n_sequences arrays, each [seq_len, C, H, W]
        sequence_info: list of (z_index, t_start) tuples
    """
    T, C, Z, H, W = volume.shape  # works for numpy, dask, zarr — no full load
    assert T >= seq_len, (
        f"Volume has only {T} timepoints but seq_len={seq_len}. "
        "Reduce seq_len or use a longer movie."
    )

    rng = np.random.default_rng(random_seed)
    z_indices = np.linspace(0, Z - 1, n_sequences, dtype=int)

    sequences = []
    sequence_info = []
    for z in z_indices:
        t_start = int(rng.integers(0, T - seq_len + 1))
        # np.array() triggers .compute() only for this slice (dask-safe)
        seq = np.array(volume[t_start:t_start + seq_len, :, int(z), :, :])  # [seq_len, C, H, W]
        sequences.append(seq)
        sequence_info.append((int(z), t_start))

    return sequences, sequence_info


def prepare_data_for_unet_batch_4d(
    volumes,
    n_sequences=3,
    seq_len=20,
    ch_indices=None,
    temporal_scales=[1, 2, 4],
    cumulative_window=2,
    random_seed=None,
    variance_config=None,
    target_size=None,
    resolution_level=0,
    return_flows=False,
):
    """Prepare training data from multiple 4D volumes [T, C, Z, Y, X].

    Extracts n_sequences 2D time sequences per volume (evenly spaced z-slices,
    random start timepoints). Optical flow is computed on the mean-projected
    single-channel version; variance metrics use the full multi-channel data.

    The returned lists have one entry per extracted sequence (n_volumes *
    n_sequences entries total) and can be fed directly into
    train_test_split_per_movie + train_with_metrics without any changes.

    Args:
        volumes:          list or dict of [T, C, Z, Y, X] arrays (one per image).
                          If dict, values may be lists of multi-resolution arrays;
                          resolution_level selects which to use (0 = full resolution).
        n_sequences:      sequences extracted per volume (= z-slices sampled)
        seq_len:          frames per sequence
        ch_indices:       channel indices to use (None = all)
        temporal_scales:  Farneback multi-scale parameters
        cumulative_window: cumulative displacement window
        random_seed:      integer seed for reproducibility
        variance_config:  VarianceMetricsConfig (None = softmax channels, pool_radius=1)
        target_size:      (H, W) to resize all frames to a common spatial size.
                          Required when volumes have different H×W and batch_size > 1.
                          None = keep original size (batch_size=1 required for mixed sizes).
        resolution_level: which resolution level to use when volumes is a dict of lists (default 0).

    Returns:
        all_frames:       list of [seq_len, H, W] float32 arrays (one per sequence)
        all_temporal:     list of metric-dict lists (one per sequence)
        all_variance:     list of variance metric-dict lists (one per sequence)
        all_frames_multi: list of [seq_len, C, H, W] uint8 arrays (for scoring / viz)
        all_flow_pairs:   (only when return_flows=True) list of flow-pair lists, one per
                          sequence; each inner list has seq_len entries ([H,W,2] or None)
    """
    import io
    import contextlib
    from coastal.flow import prepare_data_for_unet, compute_variance_metrics, VarianceMetricsConfig, normalize_and_project, extract_dense_flow_pairs

    # Accept dict {uid: [res0, res1, ...]} or plain list
    if isinstance(volumes, dict):
        volumes = [v[resolution_level] for v in volumes.values()]

    if variance_config is None:
        variance_config = VarianceMetricsConfig(pool_radius=1)

    n_volumes = len(volumes)
    n_total = n_volumes * n_sequences

    print(f"\n{'='*80}")
    print(f"4D BATCH PREPARATION")
    print(f"  {n_volumes} volumes × {n_sequences} sequences × {seq_len} frames = {n_total} sequences")
    print(f"{'='*80}\n")

    all_frames = []
    all_temporal = []
    all_variance = []
    all_frames_multi = []
    all_flow_pairs = []

    for vol_idx, volume in enumerate(volumes):
        T, C, Z, H, W = volume.shape  # dask-safe: no full load
        print(f"Volume {vol_idx + 1}/{n_volumes}: shape {volume.shape}")

        seed = None if random_seed is None else random_seed + vol_idx
        sequences, seq_info = extract_sequences_from_volume(
            volume, n_sequences=n_sequences, seq_len=seq_len, random_seed=seed
        )

        for seq_idx, (seq, (z_idx, t_start)) in enumerate(zip(sequences, seq_info)):
            frames_multi_uint8, frames_proj = normalize_and_project(seq, ch_indices, target_size=target_size)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                frames_prep, multi_scale_flows, _, temporal_metrics = prepare_data_for_unet(
                    frames_proj,
                    temporal_scales=temporal_scales,
                    cumulative_window=cumulative_window,
                )

            variance_metrics = compute_variance_metrics(frames_multi_uint8, variance_config)

            all_frames.append(frames_prep)
            # Materialise: prepare_data_for_unet hands back a lazy TemporalMetrics, but the
            # training Dataset indexes metrics per sample per epoch, so they must be computed
            # once here. Cheap at seq_len frames (vs. the full-T inference path).
            all_temporal.append(list(temporal_metrics))
            all_variance.append(variance_metrics)
            all_frames_multi.append(frames_multi_uint8)
            if return_flows:
                all_flow_pairs.append(extract_dense_flow_pairs(multi_scale_flows, scale=1))

            n_t = len(temporal_metrics[0])
            n_v = len(variance_metrics[0])
            print(f"  Seq {seq_idx + 1}/{n_sequences}: z={z_idx}, t={t_start}–{t_start + seq_len - 1} "
                  f"| {n_t} temporal + {n_v} variance metrics")

    print(f"\n✓ {n_total} sequences ready for train_test_split_per_movie\n")
    if return_flows:
        return all_frames, all_temporal, all_variance, all_frames_multi, all_flow_pairs
    return all_frames, all_temporal, all_variance, all_frames_multi


def prepare_data_for_unet_batch(movies, temporal_scales=[1, 2, 4, 8], cumulative_window=5):
    """
    Prepare data for all movies independently.
    Metrics computed per movie, NOT across movies.

    Args:
        movies: list of movies, each [Z, H, W] or [T, H, W]
        temporal_scales: scales for multi-scale optical flow
        cumulative_window: window size for cumulative displacement

    Returns:
        all_frames: list of normalized frame arrays (one per movie)
        all_metrics: list of metric lists (one per movie)
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING {len(movies)} MOVIES INDEPENDENTLY")
    print(f"Metrics computed per-movie (not across movies)")
    print(f"{'='*80}\n")

    all_frames = []
    all_metrics = []

    for movie_idx, movie in enumerate(movies):
        print(f"Movie {movie_idx + 1}/{len(movies)}: shape {np.array(movie).shape}")

        from coastal.flow import prepare_data_for_unet
        frames_prep, flows, cum_flows, metrics = prepare_data_for_unet(
            movie,
            temporal_scales=temporal_scales,
            cumulative_window=cumulative_window
        )

        all_frames.append(frames_prep)
        all_metrics.append(list(metrics))   # materialise for training (see batch_4d above)
        print(f"  ✓ {len(frames_prep)} frames with {len(metrics[0])} metrics each\n")

    return all_frames, all_metrics

