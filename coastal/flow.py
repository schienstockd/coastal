"""
TEMPORAL FLOW METRICS WITH FARNEBACK OPTICAL FLOW (STREAMLINED)
================================================================
Multi-scale windowing with Farneback + advanced flow deformation metrics.
Removes redundant metrics that don't vary (motion_at_corners, motion_at_edges, angle).
"""

import numpy as np
import cv2
from tqdm import tqdm
from joblib import Parallel, delayed


def calc_flow_farneback_between_frames(frame1, frame2):
    """Compute Farneback optical flow between two frames."""
    frame1 = np.array(frame1, dtype=np.float32)
    frame2 = np.array(frame2, dtype=np.float32)
    
    flow = cv2.calcOpticalFlowFarneback(
        frame1, frame2,
        None, 0.5, 3, 15, 3, 5, 1.2,
        cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )
    
    vx = flow[..., 0]
    vy = flow[..., 1]
    
    return vx, vy


def compute_flow_for_frame(i, frames_array, scale, N):
    """Compute flow between frame i and frame i+scale."""
    if i + scale >= N:
        return None
    
    frame1 = frames_array[i]
    frame2 = frames_array[i + scale]
    
    vx, vy = calc_flow_farneback_between_frames(frame1, frame2)
    
    return {
        'u': vx,
        'v': vy,
        'scale': scale,
        'frame_pair': (i, i + scale)
    }


def compute_multi_scale_optical_flow(frames, scales=[1, 2, 4, 8], n_jobs=-1, verbose=True):
    """Multi-scale Farneback optical flow with parallel processing."""

    frames_array = np.array(frames, dtype=np.uint8)
    N = frames_array.shape[0]
    multi_scale_flows = {}

    if verbose:
        print(f"\nComputing multi-scale optical flow (Farneback, parallel)...")
        print(f"Frame shape: {frames_array[0].shape}, scales: {scales}\n")

    for scale in scales:
        if verbose:
            print(f"  Processing scale={scale}...")

        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_flow_for_frame)(i, frames_array, scale, N)
            for i in range(N - scale)
        )

        flows_at_scale = [r for r in results if r is not None]
        multi_scale_flows[scale] = flows_at_scale
        if verbose:
            print(f"    ✓ {len(flows_at_scale)} flows")

    return multi_scale_flows


def compute_cumulative_displacement_frame(center_idx, frames_array, window_size):
    """Compute cumulative displacement for a single frame."""
    win_start = max(0, center_idx - window_size // 2)
    win_end = min(len(frames_array), center_idx + window_size // 2 + 1)
    
    vx_cum = np.zeros_like(frames_array[0], dtype=np.float32)
    vy_cum = np.zeros_like(frames_array[0], dtype=np.float32)
    
    frame_count = 0
    for idx in range(win_start, win_end - 1):
        frame1 = frames_array[idx]
        frame2 = frames_array[idx + 1]
        
        try:
            vx, vy = calc_flow_farneback_between_frames(frame1, frame2)
            vx_cum += vx
            vy_cum += vy
            frame_count += 1
        except Exception:
            continue
    
    if center_idx < 3 or center_idx % 50 == 0:
        mag_cum = np.sqrt(vx_cum**2 + vy_cum**2)
        print(f"  Center {center_idx}: {frame_count} frames, cumulative mag: min={mag_cum.min():.6f}, max={mag_cum.max():.6f}")
    
    return {
        'u': vx_cum,
        'v': vy_cum,
        'window_size': window_size,
        'center_frame': center_idx
    }


def compute_cumulative_displacement(frames, window_size=5, n_jobs=-1, verbose=True):
    """Cumulative displacement with parallel processing."""

    frames_array = np.array(frames, dtype=np.uint8)
    N = frames_array.shape[0]

    if verbose:
        print(f"Computing cumulative displacement (parallel, window={window_size})...\n")

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_cumulative_displacement_frame)(center_idx, frames_array, window_size)
        for center_idx in range(N)
    )

    cumulative_flows = [r for r in results if r is not None]

    if verbose:
        print(f"✓ {len(cumulative_flows)} cumulative flows\n")
    return cumulative_flows


def _flow_deformation(u, v):
    """Divergence, vorticity and strain-rate magnitude of a 2D flow field.

    ``u`` = x-displacement, ``v`` = y-displacement on an image array with axes
    ``[y=0, x=1]``, so ``∂/∂x = np.gradient(·, axis=1)`` and ``∂/∂y = np.gradient(·, axis=0)``.

      divergence = ∂u/∂x + ∂v/∂y   (expansion / compression)
      vorticity  = ∂v/∂x − ∂u/∂y   (rotation)
      strain     = ‖E‖ of the symmetric strain-rate tensor,
                   E_xx=∂u/∂x, E_yy=∂v/∂y, E_xy=½(∂u/∂y + ∂v/∂x)

    Returns raw (un-normalised) arrays; callers normalise for the learned features.
    """
    du_dx = np.gradient(u, axis=1)
    du_dy = np.gradient(u, axis=0)
    dv_dx = np.gradient(v, axis=1)
    dv_dy = np.gradient(v, axis=0)

    divergence = du_dx + dv_dy
    vorticity = dv_dx - du_dy
    E_xy = 0.5 * (du_dy + dv_dx)
    strain = np.sqrt(du_dx**2 + dv_dy**2 + 2 * E_xy**2)
    return divergence, vorticity, strain


def extract_temporal_metrics(frames, multi_scale_flows, cumulative_flows, frame_idx):
    """Extract rich temporal motion metrics for ONE frame."""
    
    frames_array = np.array(frames, dtype=np.float32)
    frames_array = (frames_array - frames_array.min()) / (frames_array.max() - frames_array.min() + 1e-5)
    frame = frames_array[frame_idx]
    
    metrics = {}
    
    # ==== MULTI-SCALE FLOW MAGNITUDES ====
    scales = sorted(multi_scale_flows.keys())
    scale_data = []
    
    for scale in scales:
        flows = multi_scale_flows[scale]
        
        if len(flows) == 0:
            continue
        
        idx = min(len(flows) - 1, max(0, frame_idx - 1))
        flow = flows[idx]
        
        u, v = flow['u'], flow['v']
        mag = np.sqrt(u**2 + v**2)
        mag = normalize_metric(mag)
        
        metrics[f'mag_{scale}'] = mag
        scale_data.append((scale, u, v, mag))
    
    if not scale_data:
        return metrics
    
    # ==== ACCELERATION & DIRECTION STABILITY ====
    if len(scale_data) > 1:
        acc = normalize_metric(np.abs(scale_data[-1][3] - scale_data[0][3]))
        metrics['acceleration'] = acc
        
        _, u0, v0, _ = scale_data[0]
        _, un, vn, _ = scale_data[-1]
        # Cosine similarity between coarse- and fine-scale flow vectors:
        # (f0·fn) / (|f0|·|fn|), clipped to [0,1] (keep only aligned/stable directions).
        mag0 = np.sqrt(u0**2 + v0**2)
        magn = np.sqrt(un**2 + vn**2)
        dot = (u0*un + v0*vn) / (mag0 * magn + 1e-5)
        metrics['direction_stability'] = np.clip(dot, 0, 1).astype(np.float32)
    
    # ==== CUMULATIVE DISPLACEMENT ====
    if cumulative_flows:
        closest = min(range(len(cumulative_flows)),
                     key=lambda i: abs(cumulative_flows[i]['center_frame'] - frame_idx))
        cum = cumulative_flows[closest]
        
        cum_mag = np.sqrt(cum['u']**2 + cum['v']**2)
        metrics['cumulative_mag'] = normalize_metric(cum_mag)
    
    # ==== FLOW DEFORMATION (divergence, vorticity, strain) ====
    _, u, v, _ = scale_data[0]
    divergence, vorticity, strain = _flow_deformation(u, v)
    metrics['divergence'] = normalize_metric(divergence)
    metrics['vorticity'] = normalize_metric(vorticity)
    metrics['strain'] = normalize_metric(strain)
    
    # ==== STRUCTURE TENSOR (image) ====
    gx = np.gradient(frame, axis=0)
    gy = np.gradient(frame, axis=1)
    
    Ixx = cv2.GaussianBlur(gx*gx, (7, 7), 2.0)
    Iyy = cv2.GaussianBlur(gy*gy, (7, 7), 2.0)
    Ixy = cv2.GaussianBlur(gx*gy, (7, 7), 2.0)
    
    tr = Ixx + Iyy
    det = Ixx*Iyy - Ixy**2
    disc = np.sqrt(np.maximum(tr**2 - 4*det, 0))
    
    l1 = (tr + disc) / 2
    l2 = (tr - disc) / 2
    
    metrics['edge_strength'] = normalize_metric(l1 - l2)
    
    # ==== FLOW ALIGNMENT WITH IMAGE GRADIENT ====
    grad_mag = np.sqrt(gx**2 + gy**2 + 1e-5)
    grad_x_norm = gx / grad_mag
    grad_y_norm = gy / grad_mag
    
    flow_mag = np.sqrt(u**2 + v**2 + 1e-5)
    dot_product = (gx * u + gy * v) / (grad_mag * flow_mag)
    metrics['flow_structure_alignment'] = normalize_metric(np.abs(dot_product))
    
    normal_flow = np.abs(u * grad_x_norm + v * grad_y_norm)
    metrics['normal_flow'] = normalize_metric(normal_flow)
    
    tangential_flow = np.abs(-u * grad_y_norm + v * grad_x_norm)
    metrics['tangential_flow'] = normalize_metric(tangential_flow)
    
    # ==== CELL BOUNDARY LIKELIHOOD ====
    if cumulative_flows:
        mag = normalize_metric(np.sqrt(u**2 + v**2))
        boundary = (
            0.30 * mag +
            0.25 * metrics.get('cumulative_mag', np.zeros_like(mag)) +
            0.25 * metrics['edge_strength'] +
            0.20 * metrics['tangential_flow']
        )
        metrics['cell_boundary_likelihood'] = normalize_metric(boundary)
    
    return metrics


def compute_all_temporal_metrics(frames, multi_scale_flows, cumulative_flows, verbose=True):
    """Extract temporal metrics for all frames."""
    N = len(frames)
    metrics_all = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"EXTRACTING TEMPORAL METRICS FOR ALL {N} FRAMES")
        print(f"{'='*70}")

    for idx in tqdm(range(N), desc='Frame metrics', disable=not verbose):
        m = extract_temporal_metrics(frames, multi_scale_flows, cumulative_flows, idx)
        metrics_all.append(m)

    if verbose:
        print(f"✓ Extracted metrics for {len(metrics_all)} frames")
    return metrics_all


from dataclasses import dataclass
from typing import Optional

from scipy.ndimage import gaussian_filter


def normalize_metric(arr: np.ndarray, percentile: tuple = (0.02, 99.98)) -> np.ndarray:
    mn = np.percentile(arr, percentile[0])
    mx = np.percentile(arr, percentile[1])
    if mx - mn < 1e-8:
        mn, mx = arr.min(), arr.max()
    return np.clip((arr - mn) / (mx - mn + 1e-8), 0, 1).astype(np.float32)


@dataclass
class VarianceMetricsConfig:
    """
    Controls which variance metrics are computed and their parameters.

    Metrics:
        use_softmax_channels    Per-channel softmax values (softmax_ch_0, softmax_ch_1, ...).
                                Directly encode color identity: CMAC cell → high ch_0,
                                GFP cell → high ch_1, etc. Entropy and contact maps are
                                functions of these values and are therefore redundant.

    Parameters:
        softmax_temp            Softmax temperature. Lower = peakier (more discriminative).
        pool_radius             Gaussian sigma for local smoothing.
    """
    use_softmax_channels: bool = True

    softmax_temp: float = 0.3
    pool_radius: int = 5


def _softmax(arr: np.ndarray, temp: float) -> np.ndarray:
    """Softmax over channel axis (axis=0) with temperature scaling."""
    logits = arr / (temp + 1e-8)
    logits -= logits.max(axis=0, keepdims=True)
    exp_l = np.exp(logits)
    return exp_l / (exp_l.sum(axis=0, keepdims=True) + 1e-8)


def compute_variance_metrics(
    frames_multi: np.ndarray,
    config: Optional[VarianceMetricsConfig] = None,
) -> list[dict]:
    """
    Compute per-pixel cross-channel variance metrics for each frame.

    Args:
        frames_multi:   [T, C, H, W] multi-channel frames (uint8 or float)
        config:         VarianceMetricsConfig. Defaults to all metrics enabled.

    Returns:
        List of T dicts mapping metric name -> normalized [H, W] float32 array in [0, 1].

    Metric keys:
        softmax_ch_0..N-1   Per-channel local softmax values (one per input channel).
    """
    if config is None:
        config = VarianceMetricsConfig()

    arr = np.asarray(frames_multi, dtype=np.float32)
    T, C, H, W = arr.shape

    arr_norm = np.zeros_like(arr)
    for c in range(C):
        ch = arr[:, c, :, :]
        arr_norm[:, c, :, :] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

    brightness = arr_norm.max(axis=1)  # [T, H, W] — zero in background

    metrics_all = []

    for t in range(T):
        frame = arr_norm[t]  # [C, H, W]
        w = brightness[t]    # [H, W]
        m = {}

        frame_local = np.stack(
            [gaussian_filter(frame[c], sigma=config.pool_radius) for c in range(C)],
            axis=0,
        )
        softmax_local = _softmax(frame_local, config.softmax_temp)

        if config.use_softmax_channels:
            for c in range(C):
                m[f'softmax_ch_{c}'] = normalize_metric(softmax_local[c] * w)

        metrics_all.append(m)

    return metrics_all


def normalize_and_project(frames_seq, ch_indices=None, percentile_lo=0.01, percentile_hi=99.99,
                           target_size=None):
    """Normalize multi-channel frames per-channel and compute mean projection.

    Mirrors the manual normalization done in the notebook before computing optical
    flow, so training and inference preprocess data identically.

    Args:
        frames_seq:     [T, C, H, W] raw frames (uint8 or float)
        ch_indices:     channel indices to select (None = all)
        percentile_lo:  lower percentile for clipping (default 0.01)
        percentile_hi:  upper percentile for clipping (default 99.99)
        target_size:    (H, W) to resize output to, or None to keep original size.
                        Required when mixing volumes with different spatial dimensions.

    Returns:
        frames_multi:   [T, C', H, W] uint8, per-channel normalized to [0, 255]
        frames_proj:    [T, H, W] uint8, mean projection across channels
    """
    arr = np.asarray(frames_seq, dtype=np.float32)
    if ch_indices is not None:
        arr = arr[:, list(ch_indices)]

    T, C, H, W = arr.shape
    frames_norm = arr.copy()

    for c in range(C):
        ch = frames_norm[:, c]
        lo = np.percentile(ch, percentile_lo)
        hi = np.percentile(ch, percentile_hi)
        frames_norm[:, c] = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)

    frames_multi = (frames_norm * 255).astype(np.uint8)
    frames_proj = (frames_norm.max(axis=1) * 255).astype(np.uint8)

    if target_size is not None:
        tH, tW = target_size
        out_multi = np.zeros((T, C, tH, tW), dtype=np.uint8)
        out_proj = np.zeros((T, tH, tW), dtype=np.uint8)
        for t in range(T):
            for c in range(C):
                out_multi[t, c] = cv2.resize(frames_multi[t, c], (tW, tH))
            out_proj[t] = cv2.resize(frames_proj[t], (tW, tH))
        frames_multi, frames_proj = out_multi, out_proj

    return frames_multi, frames_proj


def prepare_data_for_unet(frames, temporal_scales=[1, 2, 4, 8], cumulative_window=5, verbose=True):
    """
    Complete pipeline: frames → temporal metrics → UNet-ready data
    Uses Farneback optical flow with advanced flow deformation metrics.
    Streamlined to remove redundant metrics.
    """

    if verbose:
        print(f"\n{'='*80}")
        print(f"TEMPORAL WINDOWING PIPELINE (FARNEBACK OPTICAL FLOW - STREAMLINED)")
        print(f"{'='*80}\n")

    frames_array = np.array(frames, dtype=np.uint8)

    if verbose:
        print(f"Input: {len(frames_array)} frames of shape {frames_array[0].shape}")
        print(f"Temporal scales: {temporal_scales}\n")

    # Compute flows
    multi_scale_flows = compute_multi_scale_optical_flow(
        frames_array, scales=temporal_scales, verbose=verbose
    )
    cum_flows = compute_cumulative_displacement(
        frames_array, window_size=cumulative_window, verbose=verbose
    )

    # Normalize frames for downstream processing
    frames_normalized = frames_array.astype(np.float32)
    frames_normalized = (frames_normalized - frames_normalized.min()) / (frames_normalized.max() - frames_normalized.min() + 1e-5)

    # Extract metrics
    metrics = compute_all_temporal_metrics(
        frames_normalized, multi_scale_flows, cum_flows, verbose=verbose
    )

    if verbose:
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"Output metrics per frame:")
        for key in sorted(metrics[0].keys()):
            print(f"  - {key}: {metrics[0][key].dtype} {metrics[0][key].shape}")

    return frames_normalized, multi_scale_flows, cum_flows, metrics


def extract_dense_flow_pairs(multi_scale_flows: dict, scale: int = 1) -> list:
    """Extract per-frame dense flow arrays from the multi_scale_flows structure.

    Returns a list of length T where entry t is a [H, W, 2] float32 array
    (forward flow from frame t to frame t+1, [u=x-disp, v=y-disp] in pixels),
    or None if no flow is available for that frame (e.g. the last frame).

    Args:
        multi_scale_flows: dict returned by compute_multi_scale_optical_flow or
                           prepare_data_for_unet (keyed by scale integer)
        scale:             which temporal scale to use (default 1 = consecutive frames)

    Returns:
        List[Optional[np.ndarray]] of shape [H, W, 2]
    """
    flows_at_scale = multi_scale_flows.get(scale, [])
    if not flows_at_scale:
        return []

    # flows_at_scale is a list of dicts {u, v, scale, frame_pair}
    # indexed by source frame (frame_pair[0])
    by_t = {f['frame_pair'][0]: f for f in flows_at_scale}
    T = max(by_t.keys()) + 2  # last source frame + 1 for t+1 frame + 1 for length

    result = []
    for t in range(T):
        f = by_t.get(t)
        if f is None:
            result.append(None)
        else:
            uv = np.stack([f['u'], f['v']], axis=-1).astype(np.float32)  # [H, W, 2]
            result.append(uv)
    return result
