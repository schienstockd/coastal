"""Training functions and datasets."""

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from coastal.model import UNetWithEmbeddings
from coastal.loss import IntensityLoss, TemporalMetricsLoss, VarianceMetricsLoss, WarpConsistencyLoss


class TemporalDatasetWithAugmentation(Dataset):
    """Dataset with frame + temporal metrics + variance metrics as input."""

    def __init__(self, frames, temporal_metrics_norm, variance_metrics_norm=None,
                 flow_pairs=None):
        """
        Args:
            frames: [T, H, W] grayscale frames (max/mean projection of multi-channel data)
            temporal_metrics_norm: list of T dicts (optical flow metrics)
            variance_metrics_norm: list of T dicts (cross-channel variance metrics), or None
            flow_pairs: list of T Optional[np.ndarray [H,W,2]] forward flow from frame t to
                        t+1 ([u=x-disp, v=y-disp] in pixels), or None at the last frame.
                        From extract_dense_flow_pairs(). Used for WarpConsistencyLoss.
        """
        self.frames = frames
        self.temporal_metrics = temporal_metrics_norm
        self.variance_metrics = variance_metrics_norm or [{} for _ in range(len(frames))]
        self.flow_pairs = flow_pairs  # None = warp loss disabled

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
        v_stacked = self._stack_metrics(v_metrics, frame.shape[1:])
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
                v_next_s = self._stack_metrics(v_next, frame_next.shape[1:])
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
                       num_epochs=50, batch_size=1, seed=42, device='cuda', embedding_dim=16,
                       variance_weight=1.0, intensity_weight=1.0, temporal_weight=2.0,
                       warp_weight=0.0, flow_pairs=None,
                       max_grad_norm=1.0, variance_window_size=32, variance_dropout_p=0.5,
                       num_workers=4, use_amp=True):
    """
    Train embeddings with 3-loss design.

    Args:
        frames_prep: [T, H, W] or [T, C, H, W] training frames
        temporal_metrics_norm: list of T dicts (optical flow metrics)
        variance_metrics_norm: list of T dicts (cross-channel variance metrics), or None
        num_epochs: training epochs
        batch_size: batch size
        seed: random seed
        device: cuda or cpu
        embedding_dim: embedding dimension
        variance_weight: weight for VarianceMetricsLoss (contrastive on variance, default 1.0)
        intensity_weight: weight for IntensityLoss (default 1.0)
        temporal_weight: weight for TemporalMetricsLoss (default 2.0)
        max_grad_norm: gradient clipping threshold (default 1.0)
        variance_window_size: spatial window size for windowed variance contrastive loss (default 32)
        num_workers: DataLoader worker processes for background data loading (default 4, 0 = main process)
        use_amp: enable automatic mixed precision (float16) on CUDA for faster training (default True)

    Returns:
        model, history
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    use_variance = variance_metrics_norm is not None
    use_amp = use_amp and device != 'cpu' and torch.cuda.is_available()

    n_temporal = len(next(iter(temporal_metrics_norm), {}))
    n_variance = len(next(iter(variance_metrics_norm), {})) if use_variance else 0
    input_channels = 1 + n_temporal + n_variance

    print(f"\n{'='*80}")
    n_losses = 3 + (1 if warp_weight > 0 else 0)
    print(f"TRAINING: INTENSITY + TEMPORAL + VARIANCE{' + WARP' if warp_weight > 0 else ''} ({n_losses}-LOSS)")
    print(f"Loss: Intensity ({intensity_weight}) + Temporal ({temporal_weight}) + Variance ({variance_weight}, window={variance_window_size}px, dropout_p={variance_dropout_p}) + Warp ({warp_weight})")
    print(f"Gradient clipping: {max_grad_norm} | AMP: {use_amp} | Workers: {num_workers}")
    print(f"{'='*80}\n")

    dataset = TemporalDatasetWithAugmentation(
        frames_prep, temporal_metrics_norm,
        variance_metrics_norm if use_variance else None,
        flow_pairs=flow_pairs,
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
    loss_temporal = TemporalMetricsLoss().to(device)
    loss_variance = VarianceMetricsLoss(window_size=variance_window_size).to(device)
    loss_warp = WarpConsistencyLoss().to(device) if warp_weight > 0.0 else None

    optimizer = Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    history = {
        'total': [],
        'variance': [],
        'intensity': [],
        'temporal': [],
        'warp': [],
    }

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {
            'total': 0.0,
            'variance': 0.0,
            'intensity': 0.0,
            'temporal': 0.0,
            'warp': 0.0,
        }

        for batch_idx, batch in enumerate(dataloader):
            frame_and_metrics = batch['frame_and_metrics'].to(device, non_blocking=True)
            channels = batch['channels'].to(device, non_blocking=True)
            v_metrics = batch['variance_metrics']
            t_metrics = batch['temporal_metrics']
            frame_indices = batch['frame_idx']

            # Channel dropout on variance input channels so the model learns to
            # function without them (inference uses zeros in those positions).
            if use_variance and n_variance > 0:
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
                             warp_weight * l_warp)

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()

            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            epoch_losses['total'] += total_loss.item()
            epoch_losses['variance'] += l_variance.item()
            epoch_losses['intensity'] += l_intensity.item()
            epoch_losses['temporal'] += l_temporal.item()
            epoch_losses['warp'] += l_warp.item()

        n = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= n
            history[key].append(epoch_losses[key])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            warp_str = f" | warp={epoch_losses['warp']:.4f}" if warp_weight > 0 else ""
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"total={epoch_losses['total']:.4f} | "
                  f"int={epoch_losses['intensity']:.4f} | "
                  f"tmp={epoch_losses['temporal']:.4f} | "
                  f"var={epoch_losses['variance']:.4f}"
                  + warp_str)

    print(f"\nFinal losses:")
    print(f"  Total:     {history['total'][-1]:.4f}")
    print(f"  Intensity: {history['intensity'][-1]:.4f}")
    print(f"  Temporal:  {history['temporal'][-1]:.4f}")
    print(f"  Variance:  {history['variance'][-1]:.4f}")
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


def load_model(path, device='cuda'):
    """Load a model saved with save_model().

    Args:
        path:   path to the .pt file
        device: device to load the model onto

    Returns:
        model: UNetWithEmbeddings in eval mode
    """
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
            all_temporal.append(temporal_metrics)
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
        all_metrics.append(metrics)
        print(f"  ✓ {len(frames_prep)} frames with {len(metrics[0])} metrics each\n")

    return all_frames, all_metrics

