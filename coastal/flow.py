"""
TEMPORAL FLOW METRICS WITH FARNEBACK OPTICAL FLOW (STREAMLINED)
================================================================
Multi-scale windowing with Farneback + advanced flow deformation metrics.
Removes redundant metrics that don't vary (motion_at_corners, motion_at_edges, angle).
"""

from collections.abc import Sequence

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

    frames_array = np.asarray(frames, dtype=np.float32)
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

    frames_array = np.asarray(frames, dtype=np.float32)
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


def extract_temporal_metrics(frames, multi_scale_flows, cumulative_flows, frame_idx,
                             frame_range=None):
    """Extract rich temporal motion metrics for ONE frame.

    Args:
        frames:            [T, H, W] stack (only frame_idx is normalised and used)
        multi_scale_flows: {scale: [flow dicts]} from compute_multi_scale_optical_flow
        cumulative_flows:  [cum dicts] from compute_cumulative_displacement
        frame_idx:         which frame to compute metrics for
        frame_range:       optional (min, max) over the whole `frames` stack. The
                           normalisation is global by design, so this must be the true
                           stack min/max — pass it to skip re-scanning the stack on
                           every frame (`TemporalMetrics` caches it once).

    Returns:
        dict of 14 [H, W] float32 metric planes.
    """
    # Normalise only the frame we need. Scaling stays global (the stack's min/max), but
    # materialising the whole stack as float32 here cost a full copy per frame — i.e.
    # O(T) copies for O(T) frames on the 4D inference path.
    frames_array = np.asarray(frames)
    if frame_range is None:
        lo, hi = frames_array.min(), frames_array.max()
    else:
        lo, hi = frame_range
    frame = (frames_array[frame_idx].astype(np.float32) - lo) / (hi - lo + 1e-5)

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


class TemporalMetrics(Sequence):
    """The T per-frame metric dicts, computed on demand.

    A drop-in for the eager `list` it replaces — `len()`, `m[t]`, slicing and iteration
    all behave identically, and `m[t]` is exactly `extract_temporal_metrics(..., t)`. The
    only difference is *when* the work happens.

    This exists because the 14 float32 metric planes per frame dominate memory on the 4D
    inference path (14 × T × H × W × 4 B = 3.1 GB at T=180, 531×586) while
    `predict_sequence` only ever looks at one frame at a time. The flow fields stay
    precomputed and shared — frame t reads neighbouring scales, so they cannot be
    streamed — and they are ~1.8 GB rather than 3.1 GB.

    **Materialise for training.** Metrics are meant to be computed once and reused across
    epochs, and a Dataset indexes them per sample per epoch, so training paths call
    `list(...)` (`prepare_data_for_unet_batch_4d` does this for you). Single-pass
    consumers — segmentation inference — should iterate lazily and leave it alone.
    """

    def __init__(self, frames, multi_scale_flows, cumulative_flows):
        self._frames = np.asarray(frames)
        self._multi_scale_flows = multi_scale_flows
        self._cumulative_flows = cumulative_flows
        # Global scaling, scanned once instead of once per frame.
        self._frame_range = (self._frames.min(), self._frames.max())

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        idx = int(idx)
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(f"frame index out of range: {idx}")
        return extract_temporal_metrics(
            self._frames, self._multi_scale_flows, self._cumulative_flows, idx,
            frame_range=self._frame_range,
        )


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
        frames_multi:   [T, C', H, W] float32, per-channel normalized to [0, 255]
        frames_proj:    [T, H, W] float32, mean projection across channels

    The 0-255 RANGE is kept (every downstream consumer and the trained model expect it) but the
    result is no longer quantised to uint8. OpenCV's Farneback accepts float32 directly -- verified
    against known shifts on 16-bit intravital data -- so the 8-bit step was a lossy round-trip that
    bought nothing, and it is where the wrap bug lived. Images are 16-bit now that compression made
    that cheap; nothing in this package should funnel them through 8 bits.
    """
    # Select channels *before* the float32 conversion: converting all C first and then
    # discarding channels cost a full-movie float32 copy per z-slice. One copy, and it is
    # already private, so the per-channel scaling below can run in place.
    arr = np.asarray(frames_seq)
    if ch_indices is not None:
        arr = arr[:, list(ch_indices)]
    frames_norm = arr.astype(np.float32)

    T, C, H, W = frames_norm.shape

    for c in range(C):
        ch = frames_norm[:, c]
        lo = np.percentile(ch, percentile_lo)
        hi = np.percentile(ch, percentile_hi)
        frames_norm[:, c] = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)

    frames_multi = (frames_norm * 255).astype(np.float32)
    frames_proj = (frames_norm.max(axis=1) * 255).astype(np.float32)

    if target_size is not None:
        tH, tW = target_size
        out_multi = np.zeros((T, C, tH, tW), dtype=np.float32)
        out_proj = np.zeros((T, tH, tW), dtype=np.float32)
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

    Returns:
        frames_normalized:  [T, H, W] float32 in [0, 1]
        multi_scale_flows:  {scale: [flow dicts]}
        cumulative_flows:   [cum dicts]
        metrics:            `TemporalMetrics` — a lazy Sequence of T metric dicts. Indexes
                            and iterates exactly like the list it replaces; call `list(...)`
                            to materialise (training does, inference must not — see the
                            class docstring and docs/SEGMENTATION.md).
    """

    if verbose:
        print(f"\n{'='*80}")
        print(f"TEMPORAL WINDOWING PIPELINE (FARNEBACK OPTICAL FLOW - STREAMLINED)")
        print(f"{'='*80}\n")

    frames_array = np.asarray(frames, dtype=np.float32)

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

    # Metrics stay lazy: one frame's 14 planes at a time instead of all T up front.
    metrics = TemporalMetrics(frames_normalized, multi_scale_flows, cum_flows)

    if verbose:
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"Output metrics per frame:")
        for key in sorted(metrics[0].keys()):
            print(f"  - {key}: {metrics[0][key].dtype} {metrics[0][key].shape}")

    return frames_normalized, multi_scale_flows, cum_flows, metrics


def flow_metrics_for_frame(window, center, temporal_scales=[1, 2, 4, 8], cumulative_window=5,
                           value_range=None):
    """The metric planes for ONE frame of a window, computing only the flows that frame reads.

    `prepare_data_for_unet` builds every flow in the stack because training consumes every frame.
    **Tiled inference consumes one.** Reading `extract_temporal_metrics` shows how few it touches:

      * per scale it takes a SINGLE flow — `flows[min(len(flows)-1, max(0, frame_idx-1))]`;
      * from the cumulative list it takes the one whose `center_frame` matches.

    So on a 17-frame window at scales [1,2,4,8] with `cumulative_window=5`, the frame reads 4
    multi-scale flows and one cumulative sum (5 consecutive flows) — 9 Farneback calls, where
    `prepare_data_for_unet` computes 53. That 6x is the dominant cost of a tiled segmentation run,
    because consecutive timepoints re-read almost the same window: at radius 8, frames t and t+1
    share 16 of 17.

    Returns `(frame_normalized, metrics)` — the same pair
    `prepare_data_for_unet(window, ...)` would give as `frames_normalized[center], metrics[center]`,
    which is asserted elementwise in `tests/test_flow_metrics_for_frame.py`. That equivalence is the
    whole contract: the metric KEYS and their order are a silent train/inference coupling (see
    `tests/test_flow_metric_count.py`), so this must not become a second, subtly different
    definition of the feature set.

    Args:
        window:            [W, H, W] frames in acquisition order — a window of the movie, NOT the
                           whole movie. Same photometric scaling as training (see
                           `normalize_and_project`).
        center:            index within `window` of the frame to compute metrics for. Need not be
                           the middle: a window truncated at the start/end of a movie is shorter on
                           one side, and truncation is deliberate — repeating or mirroring a frame
                           invents motion that was not imaged.
        temporal_scales:   frame lags to compute flow over. MUST match the trained model's.
        cumulative_window: centred window for the cumulative displacement. MUST match the model's.
        value_range:       `(lo, hi)` for the 0–1 intensity scaling, instead of the window's own
                           min/max. **Tiled inference should pass this.** Training scales by the
                           whole movie's min/max, so leaving it to a single tile-window would give
                           each tile its own photometric scale — the patchiness that
                           `normaliseToWhole` exists to prevent for cellpose, plus a train/inference
                           mismatch on the structure-tensor planes, which read the scaled frame
                           directly. Whole-movie callers can leave it None and get the previous
                           behaviour.

    A scale with `len(window) <= scale` yields no flow and its `mag_{scale}` plane is silently
    absent — which shifts every later channel. The caller is responsible for supplying a long
    enough window; `test_flow_metric_count.py` documents why that is not a warning but a bug.
    """
    frames_array = np.asarray(window, dtype=np.float32)
    N = frames_array.shape[0]
    if not 0 <= center < N:
        raise IndexError(f"center {center} outside window of {N} frames")

    # Mirror prepare_data_for_unet: the key is always present, empty when the window is too short,
    # because extract_temporal_metrics skips an empty list rather than raising.
    multi_scale_flows = {}
    for scale in temporal_scales:
        n_flows = max(0, N - scale)
        if n_flows == 0:
            multi_scale_flows[scale] = []
            continue
        # the one flow `extract_temporal_metrics` will pick for `center`
        i = min(n_flows - 1, max(0, center - 1))
        vx, vy = calc_flow_farneback_between_frames(frames_array[i], frames_array[i + scale])
        multi_scale_flows[scale] = [{'u': vx, 'v': vy, 'scale': scale,
                                     'frame_pair': (i, i + scale)}]

    # Cumulative displacement for `center` only. Same summation as
    # compute_cumulative_displacement_frame, minus its progress printing.
    win_start = max(0, center - cumulative_window // 2)
    win_end = min(N, center + cumulative_window // 2 + 1)
    vx_cum = np.zeros(frames_array.shape[1:], dtype=np.float32)
    vy_cum = np.zeros(frames_array.shape[1:], dtype=np.float32)
    for idx in range(win_start, win_end - 1):
        try:
            vx, vy = calc_flow_farneback_between_frames(frames_array[idx], frames_array[idx + 1])
        except Exception:
            continue
        vx_cum += vx
        vy_cum += vy
    cumulative_flows = [{'u': vx_cum, 'v': vy_cum, 'window_size': cumulative_window,
                         'center_frame': center}]

    # Same global scaling as prepare_data_for_unet -> TemporalMetrics: over the window, once.
    lo, hi = (frames_array.min(), frames_array.max()) if value_range is None else value_range
    frames_normalized = (frames_array - lo) / (hi - lo + 1e-5)

    # What `TemporalMetrics` caches: the scaled stack's own min/max, which for a whole-movie call is
    # (0, ~1) by construction. With an explicit `value_range` the window occupies only part of that
    # span, so re-deriving it here would stretch the tile back to full contrast — i.e. silently undo
    # the global scaling the caller asked for. Use the span the scaling defines instead.
    frame_range = ((frames_normalized.min(), frames_normalized.max()) if value_range is None
                   else (0.0, 1.0))

    metrics = extract_temporal_metrics(
        frames_normalized, multi_scale_flows, cumulative_flows, center, frame_range=frame_range)

    return frames_normalized[center], metrics


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
