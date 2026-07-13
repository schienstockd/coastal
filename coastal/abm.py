"""Kalman + LAP cell tracker: per-cell flows, Kalman/Hungarian assignment, scoring."""

from __future__ import annotations
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
import os

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from coastal.track import compute_3d_centroids


# --------------------------------------------------------------------------- #
# Per-cell optical flow                                                        #
# --------------------------------------------------------------------------- #

def compute_cell_flows(
    frames:       np.ndarray,
    instances_4d: np.ndarray,
    n_workers:    int = 0,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Mean Farneback (u, v) per cell per timepoint, averaged across Z-slices.

    cell_flows[t][cid] = mean [u, v] displacement (pixels) of cell cid
    from frame t to frame t+1.  Cell IDs are taken from instances_4d[t]
    (the source frame), so that:
      - TrackingDataset can look up flows_t.get(cid_a) for anchor cid_a ∈ instances_4d[t]
      - track_sequence can look up cell_flows.get(t-1, {}).get(last_cid) where
        last_cid ∈ instances_4d[t-1] to get the displacement from t-1 to t

    Flow is computed per Z-slice (not on the max projection), then averaged over
    the slices where each cell has pixels. Timepoints are processed in parallel
    (cv2 releases the GIL).

    Args:
        frames:       [T, Z, H, W] grayscale frames (any dtype; normalised internally)
        instances_4d: [T, Z, H, W] instance label array
        n_workers:    threads (0 = min(T-1, cpu_count))

    Returns:
        {t: {cell_id: np.ndarray([u_mean, v_mean])}}
    """
    if frames.ndim != 4:
        raise ValueError(
            f"compute_cell_flows expects frames [T, Z, H, W] (4-D), "
            f"got shape {frames.shape}. "
            f"If your volume is [T, C, Z, H, W], pass vol.sum(axis=1) "
            f"to collapse the channel dimension first."
        )
    T, Z = frames.shape[:2]

    def _to_uint8(frame: np.ndarray) -> np.ndarray:
        f = np.asarray(frame, dtype=np.float32)
        mn, mx = f.min(), f.max()
        if mx - mn < 1e-8:
            return np.zeros_like(f, dtype=np.uint8)
        return ((f - mn) / (mx - mn) * 255).astype(np.uint8)

    def _process_t(t):
        # Forward flow: from frame t to frame t+1, keyed by instances_4d[t] IDs.
        # Using the source frame's IDs means:
        #   TrackingDataset: flows_t.get(cid_a) where cid_a ∈ instances_4d[t] → correct
        #   track_sequence:  cell_flows.get(t-1, {}).get(last_cid) where last_cid ∈ instances_4d[t-1] → correct
        cell_ids = np.unique(instances_4d[t])
        cell_ids = cell_ids[cell_ids > 0]
        if len(cell_ids) == 0:
            return t, {}

        max_lab = int(cell_ids.max()) + 1
        sum_u  = np.zeros(max_lab, np.float64)
        sum_v  = np.zeros(max_lab, np.float64)
        counts = np.zeros(max_lab, np.float64)

        for z in range(Z):
            seg_z = instances_4d[t, z]   # source frame mask
            if seg_z.max() == 0:
                continue
            flow_z = cv2.calcOpticalFlowFarneback(
                _to_uint8(frames[t,     z]),   # source
                _to_uint8(frames[t + 1, z]),   # target
                None, 0.5, 4, 15, 3, 5, 1.2,
                cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
            )  # [H, W, 2]
            flat_seg = seg_z.ravel()
            fg = flat_seg > 0
            if not fg.any():
                continue
            labs = flat_seg[fg].astype(np.intp)
            sum_u  += np.bincount(labs, weights=flow_z[:, :, 0].ravel()[fg], minlength=max_lab)
            sum_v  += np.bincount(labs, weights=flow_z[:, :, 1].ravel()[fg], minlength=max_lab)
            counts += np.bincount(labs, minlength=max_lab)

        safe = np.where(counts > 0, counts, 1.0)
        mu = (sum_u / safe).astype(np.float32)
        mv = (sum_v / safe).astype(np.float32)
        return t, {
            int(cid): np.array([mu[cid], mv[cid]], np.float32)
            for cid in cell_ids if cid < max_lab
        }

    if T <= 1:
        return {}

    result: Dict[int, Dict[int, np.ndarray]] = {}
    nw = n_workers or min(T - 1, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=nw) as ex:
        for t, flow_dict in ex.map(_process_t, range(0, T - 1)):
            result[t] = flow_dict

    return result


def compute_cell_flow_features(
    frames:       np.ndarray,
    instances_4d: np.ndarray,
    n_workers:    int = 0,
) -> tuple:
    """Per-cell flow feature vector and dense flow field, derived from Farneback.

    Extends compute_cell_flows: for each cell, computes a 6-dim feature vector
    [u, v, magnitude, divergence, vorticity, strain] averaged over the cell mask
    across all Z-slices. Also returns the dense (u,v) flow field per frame.

    divergence < 0  → converging flow  (pre-crossing)
    vorticity       → cells rotating past each other
    strain          → deformation axis (crossing direction)

    These are in raw pixel/frame units. L2-normalisation happens downstream.

    Args:
        frames:       [T, Z, H, W] grayscale frames
        instances_4d: [T, Z, H, W] instance label array
        n_workers:    threads (0 = min(T-1, cpu_count))

    Returns:
        cell_features:  {t: {cell_id: np.ndarray([6])}}
        dense_flows:    {t: np.ndarray([H, W, 2])}  — (u,v) averaged over Z-slices
    """
    if frames.ndim != 4:
        raise ValueError(f"compute_cell_flow_features expects [T, Z, H, W], got {frames.shape}")
    T, Z, H, W = frames.shape

    def _to_uint8(frame: np.ndarray) -> np.ndarray:
        f = np.asarray(frame, dtype=np.float32)
        mn, mx = f.min(), f.max()
        if mx - mn < 1e-8:
            return np.zeros_like(f, dtype=np.uint8)
        return ((f - mn) / (mx - mn) * 255).astype(np.uint8)

    def _process_t(t):
        cell_ids = np.unique(instances_4d[t])
        cell_ids = cell_ids[cell_ids > 0]
        if len(cell_ids) == 0:
            return t, {}, np.zeros((H, W, 2), np.float32)

        max_lab = int(cell_ids.max()) + 1
        # accumulators: u, v, mag, div, vort, strain
        sums   = np.zeros((max_lab, 6), np.float64)
        counts = np.zeros(max_lab, np.float64)
        dense_sum   = np.zeros((H, W, 2), np.float64)
        dense_count = np.zeros((H, W), np.float64)

        for z in range(Z):
            seg_z = instances_4d[t, z]
            if seg_z.max() == 0:
                continue
            flow_z = cv2.calcOpticalFlowFarneback(
                _to_uint8(frames[t,     z]),
                _to_uint8(frames[t + 1, z]),
                None, 0.5, 4, 15, 3, 5, 1.2,
                cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
            )  # [H, W, 2]

            u, v = flow_z[:, :, 0], flow_z[:, :, 1]
            mag  = np.sqrt(u**2 + v**2)

            # Deformation metrics from spatial gradients of the flow field
            du_dy, du_dx = np.gradient(u)   # note: numpy gradient axis0=row(y), axis1=col(x)
            dv_dy, dv_dx = np.gradient(v)
            div   = du_dx + dv_dy
            vort  = dv_dx - du_dy
            Sxy   = 0.5 * (du_dy + dv_dx)
            strain = np.sqrt(du_dx**2 + dv_dy**2 + 2 * Sxy**2)

            dense_sum[:, :, 0] += u
            dense_sum[:, :, 1] += v
            dense_count += 1.0

            flat_seg = seg_z.ravel()
            fg = flat_seg > 0
            if not fg.any():
                continue
            labs = flat_seg[fg].astype(np.intp)
            for ci, arr in enumerate([u, v, mag, div, vort, strain]):
                sums[:, ci] += np.bincount(labs, weights=arr.ravel()[fg], minlength=max_lab)
            counts += np.bincount(labs, minlength=max_lab)

        safe = np.where(counts > 0, counts, 1.0)
        feats = (sums / safe[:, None]).astype(np.float32)   # [max_lab, 6]

        n_z = np.maximum(dense_count, 1.0)[:, :, None]
        dense_field = (dense_sum / n_z).astype(np.float32)  # [H, W, 2]

        return t, {int(cid): feats[cid] for cid in cell_ids if cid < max_lab}, dense_field

    if T <= 1:
        return {}, {}

    nw = n_workers or min(T - 1, os.cpu_count() or 4)
    cell_features: Dict[int, Dict[int, np.ndarray]] = {}
    dense_flows:   Dict[int, np.ndarray] = {}

    with ThreadPoolExecutor(max_workers=nw) as ex:
        for t, feats, df in ex.map(_process_t, range(0, T - 1)):
            cell_features[t] = feats
            dense_flows[t]   = df

    return cell_features, dense_flows


# --------------------------------------------------------------------------- #
# Tracking inference helpers                                                   #
# --------------------------------------------------------------------------- #

def _apply_kalman_update(
    kalman_x: np.ndarray,
    kalman_P: np.ndarray,
    pos_um:   np.ndarray,
    _H:       np.ndarray,
    _R:       np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standard Kalman position update. Returns (new_x, new_P); does not pin position."""
    innov = pos_um - _H @ kalman_x
    S     = _H @ kalman_P @ _H.T + _R
    K     = kalman_P @ _H.T @ np.linalg.inv(S)
    x_new = kalman_x + K @ innov
    P_new = (np.eye(6) - K @ _H) @ kalman_P
    return x_new, P_new


def _greedy_lookahead(
    kalman_x:        np.ndarray,
    kalman_P:        np.ndarray,
    color_ema:       Optional[np.ndarray],
    centroids:       Dict[int, Dict[int, np.ndarray]],
    cell_intensities: Optional[Dict[int, Dict[int, np.ndarray]]],
    t_start:         int,
    N:               int,
    scale:           np.ndarray,
    _F:              np.ndarray,
    _H:              np.ndarray,
    _Q:              np.ndarray,
    _R:              np.ndarray,
    chi2_gate:       float,
    max_cost:        float,
    w_color:         float,
    color_ema_alpha: float,
) -> float:
    """Greedy N-step Kalman lookahead starting at t_start. Returns cumulative cost.

    Simulates the track forward independently (no conflict resolution with other
    tracks). Used to score two competing assignment options: whichever leads to
    lower cumulative cost over N future frames is preferred.
    """
    x = kalman_x.copy()
    P = kalman_P.copy()
    c = color_ema.copy() if color_ema is not None else None
    total = 0.0

    for step in range(N):
        ts = t_start + step
        cents_t = centroids.get(ts)
        if not cents_t:
            total += max_cost
            continue

        # Kalman predict
        x = _F @ x
        P = _F @ P @ _F.T + _Q

        pred      = x[:3]
        det_ids_t = list(cents_t.keys())
        det_arr_t = np.stack([cents_t[cid] * scale for cid in det_ids_t])  # µm

        # Mahalanobis chi² per detection
        S     = _H @ P @ _H.T + _R
        S_inv = np.linalg.inv(S)
        innov = det_arr_t - pred[None, :]            # [N, 3]
        mahal = (innov @ S_inv * innov).sum(axis=1)  # [N] chi²

        cost_v = np.clip(mahal / chi2_gate, 0.0, 1.0)

        if w_color > 0.0 and c is not None and cell_intensities is not None:
            det_cols  = np.stack([
                np.asarray(cell_intensities.get(ts, {}).get(cid, np.zeros(3)), np.float32)
                for cid in det_ids_t
            ])
            det_norms = np.linalg.norm(det_cols, axis=1)
            c_norm    = np.linalg.norm(c)
            if c_norm > 1e-6:
                det_cn  = det_cols / (det_norms[:, None] + 1e-8)
                cos_sim = det_cn @ (c / c_norm)
                col_cost = (1.0 - cos_sim) / 2.0
                col_cost[det_norms < 1e-6] = 0.0
                cost_v += w_color * col_cost

        cost_v[mahal > chi2_gate] = 1e9

        best_i    = int(np.argmin(cost_v))
        best_cost = float(cost_v[best_i])

        if best_cost >= 1e9 or best_cost > max_cost:
            total += max_cost
            continue

        best_pos = det_arr_t[best_i]

        x, P = _apply_kalman_update(x, P, best_pos, _H, _R)
        # Color EMA is intentionally NOT updated during simulation.
        # Updating it would let the simulated track drift toward the wrong identity,
        # making wrong-colored future detections appear cheap — inverting the signal.
        total += best_cost

    return total


# --------------------------------------------------------------------------- #
# Tracking inference                                                           #
# --------------------------------------------------------------------------- #

def track_sequence(
    instances_4d:        np.ndarray,
    pix_res:             dict,
    cell_flows:          Optional[Dict[int, Dict[int, np.ndarray]]] = None,
    search_radius_um:    float = 50.0,
    max_cost:            float = 1.5,
    n_hist:              int   = 5,
    momentum_decay:      float = 0.8,
    max_gap:             int   = 1,
    min_cell_size_px:    int   = 0,
    process_noise:       float = 1.0,
    obs_noise:           float = 9.0,
    chi2_gate:           float = 9.21,
    cell_intensities:    Optional[Dict[int, Dict[int, np.ndarray]]] = None,
    dense_flow_fields:   Optional[Dict[int, np.ndarray]] = None,
    w_flow:              float = 0.0,
    w_color:             float = 0.0,
    color_ema:           float = 0.9,
    device:              str   = None,
    swap_lookahead:      int   = 0,
    return_margins:      bool  = False,
    return_swap_log:     bool  = False,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Kalman + Hungarian LAP cell tracker (the settled tracker).

    Cost matrix:
        cost = dist_cost
             [+ w_flow  * flow_cost ]   # dense Farneback flow-warp deviation
             [+ w_color * color_cost]   # confetti cosine distance (ceiling test only)

    where dist_cost is the Mahalanobis-gated distance from the *momentum-predicted*
    position to each detection. Momentum prediction (simplified Kalman) uses the
    exponentially-weighted mean of the last n_hist velocity vectors per track,
    anchoring the search at where each cell is expected to be rather than its
    last-known position. This dramatically reduces track fragmentation when cells
    move consistently across frames.

    Args:
        instances_4d:      [T, Z, H, W] segmentation output
        pix_res:           {'z', 'y', 'x'} in µm/pixel
        cell_flows:        output of compute_cell_flows() — {t: {cell_id: [u_px, v_px]}}.
                           When provided, XY position prediction uses the cell's measured
                           optical flow (u=X, v=Y in pixels) instead of velocity averaging.
                           Z position still uses momentum. Falls back to momentum when the
                           track was not seen at exactly t-1 or flow is missing.
        search_radius_um:  spatial search window for candidate cells (µm)
        max_cost:          maximum total cost to accept an assignment. Assignments with
                           cost > max_cost are rejected — the track enters gap mode rather
                           than locking onto a low-confidence detection.
        n_hist:            number of past velocity frames to keep per track (momentum buffer)
        momentum_decay:    exponential decay per older frame (0 = use only last velocity,
                           1 = uniform average). Default 0.8 weights recent frames heavily.
        max_gap:           frames a track can go unmatched before closing
        min_cell_size_px:  discard detections with fewer pixels in Z-max projection (0 = keep all)
        chi2_gate:         base χ² threshold for the 3-DOF Mahalanobis gate (default 9.21 ≈
                           χ²₃ 97.3% quantile; use 11.34 for a true 99% 3-DOF gate)
        dense_flow_fields: {t: [H, W, 2]} dense (u,v) flow from compute_cell_flow_features();
                           enables the w_flow flow-warp cost.
        w_flow:            weight for the dense flow-warp cost (0 = disabled).
        cell_intensities:  {t: {cell_id: [n_ch]}} confetti RGB for the w_color cost.
        w_color:           confetti cosine cost weight (0 = disabled, ceiling test only).
        color_ema:         EMA coefficient for track colour estimate.
        device:            torch device string.
        swap_lookahead:    N-step greedy lookahead to resolve conflicting assignments (0 = off).

    Returns:
        {track_id: {t: centroid_um [3]}}
    """
    T         = instances_4d.shape[0]
    scale     = np.array([pix_res['z'], pix_res['y'], pix_res['x']], dtype=np.float32)
    centroids = compute_3d_centroids(instances_4d)

    # Filter small detections
    if min_cell_size_px > 0:
        n_removed = 0
        for t in range(T):
            proj = instances_4d[t].max(axis=0)
            ids, counts = np.unique(proj[proj > 0], return_counts=True)
            small = {int(i) for i, c in zip(ids, counts) if c < min_cell_size_px}
            before = len(centroids.get(t, {}))
            centroids[t] = {cid: c for cid, c in centroids.get(t, {}).items()
                            if cid not in small}
            n_removed += before - len(centroids[t])
        print(f"Filtered {n_removed} small detections (<{min_cell_size_px} px)")
    # --- Active track state ---
    track_pos:       Dict[int, np.ndarray] = {}
    track_last_cell: Dict[int, int]        = {}  # cell_id matched at track_last_t
    track_last_t:    Dict[int, int]        = {}
    track_vels:      Dict[int, deque]      = {}  # velocity history (deque of [3] vecs)
    kalman_x:        Dict[int, np.ndarray] = {}  # [6] = [z, y, x, dz, dy, dx]
    kalman_P:        Dict[int, np.ndarray] = {}  # [6x6] covariance
    track_colors:    Dict[int, np.ndarray] = {}  # [3] L2-norm confetti EMA per track
    next_tid = 0
    output_tracks: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    _margins:  Dict[tuple, tuple] = {}   # (t, tid) -> (margin, assigned_cid)
    _swap_log: List[dict]         = []   # one entry per evaluated conflict pair

    # Kalman filter matrices (constant-velocity model, observe position only)
    _F  = np.eye(6);   _F[:3, 3:] = np.eye(3)           # state transition
    _H  = np.zeros((3, 6));  _H[:3, :3] = np.eye(3)     # observation
    _Q  = np.diag([process_noise]*3 + [process_noise*2]*3)
    _R  = np.diag([obs_noise]*3)
    _P0 = np.diag([obs_noise]*3 + [process_noise*4]*3)  # initial covariance

    def _kalman_init(pos_um: np.ndarray) -> tuple:
        x = np.zeros(6);  x[:3] = pos_um
        return x, _P0.copy()

    def _momentum_vel(tid: int) -> np.ndarray:
        """Exponentially-weighted mean velocity [3] for tid, or zeros."""
        vels = track_vels.get(tid)
        if not vels:
            return np.zeros(3, dtype=np.float32)
        n   = len(vels)
        ws  = np.array([momentum_decay ** (n - 1 - i) for i in range(n)], dtype=np.float32)
        ws /= ws.sum()
        vel = np.zeros(3, dtype=np.float32)
        for w, v in zip(ws, vels):
            vel += w * v
        return vel

    for t in range(T):
        cells_t = list(centroids[t].items())
        if not cells_t:
            continue

        cells_um = {cid: c * scale for cid, c in cells_t}
        det_ids  = list(cells_um.keys())

        if t == 0:
            for cid, pos_um in cells_um.items():
                track_pos[next_tid]        = pos_um
                track_last_cell[next_tid]  = cid
                track_last_t[next_tid]     = 0
                track_vels[next_tid]       = deque(maxlen=n_hist)
                kalman_x[next_tid], kalman_P[next_tid] = _kalman_init(pos_um)
                output_tracks[next_tid][0] = pos_um
                next_tid += 1
            continue

        # Prune stale tracks
        active_tids = [tid for tid, lt in track_last_t.items() if t - lt - 1 <= max_gap]
        for tid in [tid for tid in list(track_pos) if tid not in active_tids]:
            del track_pos[tid], track_last_t[tid]
            track_last_cell.pop(tid, None)
            track_vels.pop(tid, None)
            kalman_x.pop(tid, None)
            kalman_P.pop(tid, None)

        if not active_tids:
            for cid, pos_um in cells_um.items():
                track_pos[next_tid]        = pos_um
                track_last_cell[next_tid]  = cid
                track_last_t[next_tid]     = t
                track_vels[next_tid]       = deque(maxlen=n_hist)
                kalman_x[next_tid], kalman_P[next_tid] = _kalman_init(pos_um)
                output_tracks[next_tid][t] = pos_um
                next_tid += 1
            continue

        # --- Kalman predict step (updates kalman_x/P to t before cost matrix) ---
        for tid in active_tids:
            # Seed XY velocity from Farneback flow when available
            if cell_flows is not None and track_last_t[tid] == t - 1:
                flow_uv = cell_flows.get(t - 1, {}).get(track_last_cell.get(tid))
                if flow_uv is not None:
                    kalman_x[tid][4] = float(flow_uv[1]) * pix_res['y']
                    kalman_x[tid][5] = float(flow_uv[0]) * pix_res['x']
            kalman_x[tid] = _F @ kalman_x[tid]
            kalman_P[tid] = _F @ kalman_P[tid] @ _F.T + _Q

        # --- Cost matrix ---
        det_arr  = np.stack([cells_um[cid]         for cid in det_ids])     # [N, 3]
        pred_arr = np.stack([kalman_x[tid][:3]     for tid in active_tids]) # [M, 3]

        # Mahalanobis-distance validation gate over the 3-DOF position innovation, replacing a
        # fixed search_radius_um (cf. Wojke et al. 2017, "Simple Online and Realtime Tracking with
        # a Deep Association Metric" / DeepSORT, which gates on the χ² quantile of the Mahalanobis
        # distance). NOTE: the gate has 3 DOF (z,y,x), so the default chi2_gate=9.21 is the χ²₃
        # ~97.3% quantile (9.21 is the χ²₂ 99% value); a true 99% 3-DOF gate would be 11.34.
        _S_inv = np.stack([np.linalg.inv(_H @ kalman_P[tid] @ _H.T + _R)
                           for tid in active_tids])                          # [M, 3, 3]
        _innov = det_arr[None] - pred_arr[:, None]                          # [M, N, 3]
        mahal  = np.einsum('mni,mij,mnj->mn', _innov, _S_inv, _innov)      # [M, N]
        dist_cost = np.clip(mahal / chi2_gate, 0.0, 1.0)


        # --- Flow-warp cost (path-of-least-resistance) ---
        if dense_flow_fields is not None and w_flow > 0.0:
            _df_field = dense_flow_fields.get(t)
            if _df_field is not None:
                H_f, W_f = _df_field.shape[:2]
                # Sample flow at last observed position (t), not Kalman prediction (t+1).
                # Kalman predict step already advanced kalman_x by velocity, so
                # last_pos = pred - velocity recovers the position at time t.
                vel_arr  = np.stack([kalman_x[tid][3:6] for tid in active_tids])  # [M, 3]
                last_pos = pred_arr - vel_arr                                      # [M, 3] at t
                _py = np.clip(np.round(last_pos[:, 1] / scale[1]).astype(int), 0, H_f - 1)
                _px = np.clip(np.round(last_pos[:, 2] / scale[2]).astype(int), 0, W_f - 1)
                # flow at last observed position in µm
                _fu = _df_field[_py, _px, 0] * scale[2]   # u → x (µm)
                _fv = _df_field[_py, _px, 1] * scale[1]   # v → y (µm)
                flow_pred_yx = np.stack([
                    last_pos[:, 1] + _fv,
                    last_pos[:, 2] + _fu,
                ], axis=1)   # [M, 2] in µm — flow-warped prediction at t+1
                det_yx = det_arr[:, 1:3]   # [N, 2]
                flow_innov = det_yx[None] - flow_pred_yx[:, None]   # [M, N, 2]
                flow_dev   = np.sqrt((flow_innov**2).sum(-1))        # [M, N] µm
                gate_um    = chi2_gate ** 0.5 * np.mean(scale[1:3]) * 3.0
                flow_cost  = np.clip(flow_dev / (gate_um + 1e-8), 0.0, 1.0)
                dist_cost  = (1.0 - w_flow) * dist_cost + w_flow * flow_cost

        cost = dist_cost.copy()

        cost[mahal > chi2_gate] = 1e9   # Mahalanobis gate replaces fixed search_radius_um

        # Confetti color cost — cosine distance between track's EMA color and each detection.
        # Ceiling test only: cell_intensities carries confetti RGB, not available at inference.
        # Tracks/detections with no valid color contribute zero cost.
        if w_color > 0.0 and cell_intensities is not None:
            det_colors = np.stack([
                np.asarray(cell_intensities.get(t, {}).get(cid, np.zeros(3)), np.float32)
                for cid in det_ids
            ])                                                                        # [N, 3]
            det_norms = np.linalg.norm(det_colors, axis=1)                           # [N]
            track_colors_arr = np.stack([
                track_colors.get(tid, np.zeros(3, np.float32))
                for tid in active_tids
            ])                                                                        # [M, 3]
            tr_norms = np.linalg.norm(track_colors_arr, axis=1)                      # [M]
            det_cn  = det_colors   / (det_norms[:, None]   + 1e-8)                  # [N, 3]
            tr_cn   = track_colors_arr / (tr_norms[:, None] + 1e-8)                 # [M, 3]
            cos_sim    = tr_cn @ det_cn.T                                            # [M, N]
            color_cost = (1.0 - cos_sim) / 2.0                                      # [0, 1]
            color_cost[tr_norms  < 1e-6, :] = 0.0   # new tracks have no color yet
            color_cost[:, det_norms < 1e-6] = 0.0   # dim detections excluded
            cost += w_color * color_cost

        row_ind, col_ind = linear_sum_assignment(cost)

        # Margin = second-best valid cost − best cost (higher = more certain)
        if return_margins:
            for ri, ci in zip(row_ind, col_ind):
                if cost[ri, ci] >= 1e9 or cost[ri, ci] > max_cost:
                    continue
                row = cost[ri, :].copy()
                row[ci] = np.inf
                valid_alts = row[row < 1e9]
                margin = float(valid_alts.min() - cost[ri, ci]) if len(valid_alts) else np.inf
                _margins[(t, active_tids[ri])] = (margin, det_ids[ci])

        # Build final assignment (ri -> ci). swap_lookahead may override conflict pairs.
        final_assignment: Dict[int, int] = {
            ri: ci for ri, ci in zip(row_ind, col_ind)
            if cost[ri, ci] < 1e9 and cost[ri, ci] <= max_cost
        }

        if swap_lookahead > 0 and len(final_assignment) > 1:
            col_to_ri   = {ci: ri for ri, ci in final_assignment.items()}
            swaps_tried: set = set()

            for ri, ci in list(final_assignment.items()):
                # Find ri's preferred detection (cheapest alternative to ci)
                row_c = cost[ri, :].copy()
                row_c[ci] = np.inf
                alt_valid = np.where(row_c < 1e9)[0]
                if not len(alt_valid):
                    continue
                ci_want = int(alt_valid[np.argmin(row_c[alt_valid])])
                if row_c[ci_want] >= cost[ri, ci]:
                    continue  # positive margin — ri got its preferred detection

                # Negative margin: ri's preferred detection was taken by ri2
                ri2 = col_to_ri.get(ci_want)
                if ri2 is None:
                    continue
                pair = (min(ri, ri2), max(ri, ri2))
                if pair in swaps_tried:
                    continue
                swaps_tried.add(pair)

                ci2 = final_assignment[ri2]  # == ci_want
                # Both swap targets must be within gate for their new tracks
                if cost[ri, ci2] >= 1e9 or cost[ri2, ci] >= 1e9:
                    continue

                tid  = active_tids[ri]
                tid2 = active_tids[ri2]
                pos_ci  = cells_um[det_ids[ci]]
                pos_ci2 = cells_um[det_ids[ci2]]

                # Kalman states after each option (re-evaluated from current state)
                xA1, PA1 = _apply_kalman_update(kalman_x[tid].copy(),  kalman_P[tid].copy(),  pos_ci,  _H, _R)
                xA2, PA2 = _apply_kalman_update(kalman_x[tid2].copy(), kalman_P[tid2].copy(), pos_ci2, _H, _R)
                xB1, PB1 = _apply_kalman_update(kalman_x[tid].copy(),  kalman_P[tid].copy(),  pos_ci2, _H, _R)
                xB2, PB2 = _apply_kalman_update(kalman_x[tid2].copy(), kalman_P[tid2].copy(), pos_ci,  _H, _R)

                # Color EMA after each option
                def _col_upd(tid_k, cid_k):
                    c_p = track_colors.get(tid_k)
                    if not (w_color > 0.0 and cell_intensities is not None):
                        return c_p
                    raw = cell_intensities.get(t, {}).get(cid_k)
                    if raw is None:
                        return c_p
                    cv = np.asarray(raw, np.float32)
                    nv = np.linalg.norm(cv)
                    if nv < 1e-6:
                        return c_p
                    cv /= nv
                    return color_ema * (c_p if c_p is not None else cv) + (1.0 - color_ema) * cv

                _la = dict(
                    centroids=centroids, cell_intensities=cell_intensities,
                    t_start=t + 1, N=swap_lookahead, scale=scale,
                    _F=_F, _H=_H, _Q=_Q, _R=_R,
                    chi2_gate=chi2_gate, max_cost=max_cost,
                    w_color=w_color, color_ema_alpha=color_ema,
                )
                cost_A = (cost[ri,  ci]  + cost[ri2, ci2]
                          + _greedy_lookahead(xA1, PA1, _col_upd(tid,  det_ids[ci]),  **_la)
                          + _greedy_lookahead(xA2, PA2, _col_upd(tid2, det_ids[ci2]), **_la))
                cost_B = (cost[ri,  ci2] + cost[ri2, ci]
                          + _greedy_lookahead(xB1, PB1, _col_upd(tid,  det_ids[ci2]), **_la)
                          + _greedy_lookahead(xB2, PB2, _col_upd(tid2, det_ids[ci]),  **_la))

                swapped = cost_B < cost_A
                if swapped:
                    final_assignment[ri]  = ci2
                    final_assignment[ri2] = ci
                    col_to_ri[ci]  = ri2
                    col_to_ri[ci2] = ri

                if return_swap_log:
                    _swap_log.append({
                        't':        t,
                        'tid':      tid,
                        'tid2':     tid2,
                        'cost_A':   float(cost_A),
                        'cost_B':   float(cost_B),
                        'swapped':  swapped,
                        # final cell IDs for each track (after potential swap)
                        'cid_tid':  det_ids[ci2 if swapped else ci],
                        'cid_tid2': det_ids[ci  if swapped else ci2],
                        # previous cell IDs (before this frame's assignment)
                        'prev_cid_tid':  track_last_cell.get(tid),
                        'prev_cid_tid2': track_last_cell.get(tid2),
                    })

        assigned_det = set()
        for ri, ci in final_assignment.items():
            tid    = active_tids[ri]
            cid    = det_ids[ci]
            pos_um = cells_um[cid]

            # Kalman update
            kalman_x[tid], kalman_P[tid] = _apply_kalman_update(
                kalman_x[tid], kalman_P[tid], pos_um, _H, _R)

            # Confetti color EMA update
            if w_color > 0.0 and cell_intensities is not None:
                raw_col = cell_intensities.get(t, {}).get(cid)
                if raw_col is not None:
                    c_col = np.asarray(raw_col, np.float32)
                    n_col = np.linalg.norm(c_col)
                    if n_col > 1e-6:
                        c_col /= n_col
                        prev_c = track_colors.get(tid, c_col)
                        track_colors[tid] = color_ema * prev_c + (1.0 - color_ema) * c_col

            # Velocity history
            vel = pos_um - track_pos[tid]
            if tid not in track_vels:
                track_vels[tid] = deque(maxlen=n_hist)
            track_vels[tid].append(vel)

            track_pos[tid]        = pos_um
            track_last_cell[tid]  = cid
            track_last_t[tid]     = t
            output_tracks[tid][t] = pos_um
            assigned_det.add(ci)

        # Unassigned detections → new tracks
        for ci, cid in enumerate(det_ids):
            if ci not in assigned_det:
                pos_um = cells_um[cid]
                track_pos[next_tid]        = pos_um
                track_last_cell[next_tid]  = cid
                track_last_t[next_tid]     = t
                track_vels[next_tid]       = deque(maxlen=n_hist)
                kalman_x[next_tid], kalman_P[next_tid] = _kalman_init(pos_um)
                if w_color > 0.0 and cell_intensities is not None:
                    c0 = cell_intensities.get(t, {}).get(cid)
                    if c0 is not None:
                        cv = np.asarray(c0, np.float32)
                        nv = np.linalg.norm(cv)
                        if nv > 1e-6:
                            track_colors[next_tid] = cv / nv
                output_tracks[next_tid][t] = pos_um
                next_tid += 1

    print(f"track_sequence: {len(output_tracks)} tracks over {T} frames")
    if return_swap_log:
        n_eval    = len(_swap_log)
        n_swapped = sum(e['swapped'] for e in _swap_log)
        print(f"  swap_lookahead={swap_lookahead}: {n_eval} pairs evaluated, "
              f"{n_swapped} swapped ({100*n_swapped/max(n_eval,1):.1f}%)")
    if return_margins and return_swap_log:
        return dict(output_tracks), _margins, _swap_log
    if return_margins:
        return dict(output_tracks), _margins
    if return_swap_log:
        return dict(output_tracks), _swap_log
    return dict(output_tracks)


# --------------------------------------------------------------------------- #
# Post-processing                                                              #
# --------------------------------------------------------------------------- #

def stitch_tracklets(
    tracks:           Dict[int, Dict[int, np.ndarray]],
    stitch_gap:       int   = 4,
    stitch_max_cost:  float = 0.4,
    search_radius_um: float = 50.0,
    n_vel:            int   = 3,
    max_passes:       int   = 3,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Connect track fragments separated by missed detections.

    For each pair (tail of track A, head of track B) where B starts within
    stitch_gap frames of A ending:

        cost = 0.6 * spatial_continuity + 0.4 * direction_continuity

    Runs linear_sum_assignment per tail-end timepoint; iterates up to
    max_passes so A→B→C chains are resolved in successive passes.

    Args:
        tracks:           output of track_sequence()
        stitch_gap:       max frame gap to bridge (default 4)
        stitch_max_cost:  reject stitches above this threshold [0,1]
        search_radius_um: normalisation distance for spatial cost
        n_vel:            number of frames used to estimate tail/head velocity
        max_passes:       iteration limit for chain stitching

    Returns:
        stitched tracks dict (same format as input)
    """
    def _one_pass(trks: Dict[int, Dict[int, np.ndarray]]) -> Dict[int, Dict[int, np.ndarray]]:
        tids = [tid for tid, frames in trks.items() if len(frames) >= 2]
        if not tids:
            return trks

        info: Dict[int, dict] = {}
        for tid in tids:
            frames = trks[tid]
            ts     = sorted(frames.keys())
            n      = min(n_vel, len(ts) - 1)
            tv     = np.mean([frames[ts[-i]] - frames[ts[-i - 1]] for i in range(1, n + 1)], axis=0)
            hv     = np.mean([frames[ts[i + 1]] - frames[ts[i]] for i in range(n)], axis=0)
            info[tid] = dict(t_end=ts[-1], last_pos=frames[ts[-1]], tail_vel=tv,
                             t_start=ts[0], first_pos=frames[ts[0]], head_vel=hv)

        # Group tails by end-time, heads by start-time
        from collections import defaultdict as _dd
        tails_by_t: Dict[int, List[int]] = _dd(list)
        heads_by_t: Dict[int, List[int]] = _dd(list)
        for tid in tids:
            tails_by_t[info[tid]['t_end']].append(tid)
            heads_by_t[info[tid]['t_start']].append(tid)

        out          = {tid: dict(frames) for tid, frames in trks.items()}
        merged_into: Dict[int, int] = {}
        used_head    = set()
        used_tail    = set()

        for t_end in sorted(tails_by_t):
            src = [tid for tid in tails_by_t[t_end] if tid not in used_tail]
            dst = []
            for dt in range(1, stitch_gap + 1):
                dst.extend(tid for tid in heads_by_t.get(t_end + dt, [])
                           if tid not in used_head)
            if not src or not dst:
                continue

            cost_l = np.full((len(src), len(dst)), 1e9)
            for i, tid_i in enumerate(src):
                ti = info[tid_i]
                for j, tid_j in enumerate(dst):
                    tj  = info[tid_j]
                    gap = tj['t_start'] - ti['t_end']
                    pred = ti['last_pos'] + ti['tail_vel'] * gap
                    sc   = min(float(np.linalg.norm(tj['first_pos'] - pred))
                               / search_radius_um, 1.0)
                    tv_n = float(np.linalg.norm(ti['tail_vel']))
                    hv_n = float(np.linalg.norm(tj['head_vel']))
                    if tv_n > 1e-6 and hv_n > 1e-6:
                        dc = (1.0 - float(np.dot(ti['tail_vel'], tj['head_vel']))
                              / (tv_n * hv_n)) / 2.0
                    else:
                        dc = 0.5
                    cost_l[i, j] = 0.6 * sc + 0.4 * dc

            ri, ci = linear_sum_assignment(cost_l)
            for r, c in zip(ri, ci):
                if cost_l[r, c] > stitch_max_cost:
                    continue
                tail_tid = src[r]
                head_tid = dst[c]

                root = tail_tid
                while root in merged_into:
                    root = merged_into[root]
                hroot = head_tid
                while hroot in merged_into:
                    hroot = merged_into[hroot]

                if root == hroot or hroot not in out:
                    continue

                out[root].update(out.pop(hroot))
                merged_into[hroot] = root
                used_tail.add(tail_tid)
                used_head.add(head_tid)

        return out

    result   = tracks
    prev_len = len(tracks)
    for _ in range(max_passes):
        result = _one_pass(result)
        if len(result) == prev_len:
            break
        prev_len = len(result)

    n_merged = len(tracks) - len(result)
    print(f"stitch_tracklets: {len(tracks)} → {len(result)} tracks  ({n_merged} segments merged)")
    return result


# --------------------------------------------------------------------------- #
# Scoring                                                                     #
# --------------------------------------------------------------------------- #

def score_tracking(
    tracks: Dict[int, Dict[int, np.ndarray]],
    instances_4d: np.ndarray,
    volumes: np.ndarray,
    ch_indices: List[int],
    pix_res: dict,
    dim_quantile: float = 0.1,
    _color_ids=None,
    _centroids=None,
    verbose: bool = True,
) -> dict:
    """Measure tracking quality: color-switch rate + fragmentation.

    Color-switch rate: within each track segment, how often does the mapped
    confetti color change? A low rate means each individual segment follows
    the right cell — but tracks may still be fragmented (many short segments
    per physical cell).

    Fragmentation: for each confetti color (physical cell), how many distinct
    track IDs cover it across the full movie? A perfect tracker gives 1 per cell.

    Track length stats: distribution of segment lengths reveals fragmentation
    even without color information.

    Returns:
        dict with keys:
            color_switch_rate, n_tracks_evaluated, n_switches_total,
            n_transitions_total, per_track,
            track_lengths (dict: min/median/mean/max/n_total),
            fragmentation (dict: median/mean/max tracks_per_cell,
                           n_cells_evaluated, per_color)
    """
    from coastal.track import extract_cell_colors, compute_3d_centroids

    color_ids = _color_ids if _color_ids is not None else \
        extract_cell_colors(instances_4d, volumes, ch_indices, dim_quantile)
    centroids = _centroids if _centroids is not None else \
        compute_3d_centroids(instances_4d)
    scale      = np.array([pix_res['z'], pix_res['y'], pix_res['x']], dtype=np.float32)
    max_match  = 50.0

    # Pre-compute per-timepoint centroid arrays once
    cids_by_t:  Dict[int, list]       = {}
    cents_by_t: Dict[int, np.ndarray] = {}
    for t, cells in centroids.items():
        if cells:
            cids = list(cells.keys())
            cids_by_t[t]  = cids
            cents_by_t[t] = np.array([cells[c] * scale for c in cids], dtype=np.float32)

    # --- Track length distribution (no centroid lookup needed) ---
    all_lens = sorted(len(v) for v in tracks.values())
    n_total  = len(all_lens)
    length_stats = {
        'n_total': n_total,
        'min':     int(all_lens[0])             if all_lens else 0,
        'median':  int(np.median(all_lens))     if all_lens else 0,
        'mean':    float(np.mean(all_lens))     if all_lens else 0.0,
        'max':     int(all_lens[-1])            if all_lens else 0,
        'pct25':   int(np.percentile(all_lens, 25)) if all_lens else 0,
        'pct75':   int(np.percentile(all_lens, 75)) if all_lens else 0,
    }

    n_switches_total    = 0
    n_transitions_total = 0
    n_tracks_evaluated  = 0
    per_track = {}

    for tid, tpoints in tracks.items():
        tps = sorted(tpoints.keys())
        if len(tps) < 2:
            continue

        track_colors = []
        for t in tps:
            pos_um = tpoints[t]
            if t not in cents_by_t:
                track_colors.append(-1)
                continue
            cents_um   = cents_by_t[t]
            cell_ids_t = cids_by_t[t]
            dists      = np.sqrt(((cents_um - pos_um) ** 2).sum(axis=1))
            nearest    = cell_ids_t[int(dists.argmin())]
            col = color_ids.get(t, {}).get(nearest, -1) if dists.min() <= max_match else -1
            track_colors.append(col)

        valid = [(i, c) for i, c in enumerate(track_colors) if c >= 0]
        if len(valid) < 2:
            continue

        switches    = sum(1 for i in range(1, len(valid)) if valid[i][1] != valid[i - 1][1])
        transitions = len(valid) - 1

        # Collect run lengths between color switches
        runs, run = [], 1
        for i in range(1, len(valid)):
            if valid[i][1] == valid[i - 1][1]:
                run += 1
            else:
                runs.append(run)
                run = 1
        runs.append(run)

        dominant_color = max(set(c for _, c in valid), key=lambda c: sum(1 for _, cc in valid if cc == c))

        n_switches_total    += switches
        n_transitions_total += transitions
        n_tracks_evaluated  += 1
        per_track[tid] = {
            'switches':        switches,
            'transitions':     transitions,
            'switch_rate':     switches / transitions if transitions > 0 else 0.0,
            'length':          len(tps),
            'runs':            runs,
            'mean_run_len':    float(np.mean(runs)),
            'dominant_color':  dominant_color,
        }

    overall_rate = (n_switches_total / n_transitions_total
                    if n_transitions_total > 0 else float('nan'))

    all_run_lens = [v for pt in per_track.values() for v in [pt['mean_run_len']]]
    mean_seg_len = float(np.mean(all_run_lens)) if all_run_lens else float('nan')

    # --- Fragmentation: tracks per physical cell ---
    # Group track IDs by dominant confetti color; count how many segments cover each cell.
    color_to_tids: Dict[int, list] = defaultdict(list)
    for tid, pt in per_track.items():
        color_to_tids[pt['dominant_color']].append(tid)
    tracks_per_cell = sorted(len(v) for v in color_to_tids.values())
    frag_stats = {
        'n_cells_evaluated': len(tracks_per_cell),
        'median': float(np.median(tracks_per_cell)) if tracks_per_cell else float('nan'),
        'mean':   float(np.mean(tracks_per_cell))   if tracks_per_cell else float('nan'),
        'max':    int(tracks_per_cell[-1])           if tracks_per_cell else 0,
        'pct75':  float(np.percentile(tracks_per_cell, 75)) if tracks_per_cell else float('nan'),
        'frac_fragmented': float(np.mean([v > 1 for v in tracks_per_cell])) if tracks_per_cell else float('nan'),
    }

    # --- Within-run switches vs gap switches ---
    # A "gap switch" is a color switch at a frame immediately following a gap in the track
    # (track timepoints are not consecutive). These are likely stitching failures or new-cell
    # assignments, not within-run misassignments. Within-run switches are the harder problem.
    within_run_switches = 0
    gap_switches        = 0
    for tid, pt in per_track.items():
        tps = sorted(tracks[tid].keys())
        track_colors_seq = []
        for tp in tps:
            pos_um = tracks[tid][tp]
            if tp not in cents_by_t:
                track_colors_seq.append((-1, tp))
                continue
            cents_um   = cents_by_t[tp]
            cell_ids_t = cids_by_t[tp]
            dists      = np.sqrt(((cents_um - pos_um) ** 2).sum(axis=1))
            nearest    = cell_ids_t[int(dists.argmin())]
            col = color_ids.get(tp, {}).get(nearest, -1) if dists.min() <= max_match else -1
            track_colors_seq.append((col, tp))
        valid_seq = [(c, tp) for c, tp in track_colors_seq if c >= 0]
        for i in range(1, len(valid_seq)):
            c_prev, t_prev = valid_seq[i - 1]
            c_cur,  t_cur  = valid_seq[i]
            if c_cur != c_prev:
                if t_cur == t_prev + 1:
                    within_run_switches += 1
                else:
                    gap_switches += 1
    switch_breakdown = {
        'within_run': within_run_switches,
        'at_gap':     gap_switches,
        'total':      within_run_switches + gap_switches,
    }

    # --- Frame-to-frame continuity ---
    # For each consecutive frame pair, what fraction of active tracks survive to the next frame?
    # High continuity = tracks rarely break. This is a direct measure of fragmentation.
    active_at: Dict[int, set] = defaultdict(set)
    for tid, tpoints in tracks.items():
        for t_pt in tpoints:
            active_at[t_pt].add(tid)

    T_vals = sorted(active_at.keys())
    cont_rates = []
    for i in range(len(T_vals) - 1):
        t0, t1 = T_vals[i], T_vals[i + 1]
        if t1 == t0 + 1:
            at_t0  = active_at[t0]
            bridge = active_at[t0] & active_at[t1]
            if at_t0:
                cont_rates.append(len(bridge) / len(at_t0))

    mean_continuity = float(np.mean(cont_rates)) if cont_rates else float('nan')
    continuity_stats = {
        'mean':      mean_continuity,
        'min':       float(min(cont_rates)) if cont_rates else float('nan'),
        'max':       float(max(cont_rates)) if cont_rates else float('nan'),
        'per_frame': cont_rates,
    }

    if verbose:
        T = instances_4d.shape[0]
        print(f"--- Track length distribution ({n_total} tracks, {T} frames) ---")
        print(f"  min={length_stats['min']}  p25={length_stats['pct25']}  "
              f"median={length_stats['median']}  p75={length_stats['pct75']}  "
              f"max={length_stats['max']}  mean={length_stats['mean']:.1f}")
        n_full = sum(1 for l in all_lens if l == T)
        print(f"  full-length ({T} frames): {n_full} tracks  "
              f"({100*n_full/max(n_total,1):.0f}%)")

        print(f"\n--- Frame-to-frame continuity (track fragmentation) ---")
        print(f"  mean={mean_continuity:.3f}  "
              f"min={continuity_stats['min']:.3f}  max={continuity_stats['max']:.3f}")
        print(f"  Fraction of active tracks that survive each consecutive frame pair.")
        print(f"  1.0 = no breaks, ~0.8 = 20% of tracks re-initialise each frame.")

        print(f"\n--- Color-switch rate (within-segment identity) ---")
        print(f"  rate={overall_rate:.4f}  "
              f"({n_switches_total} switches / {n_transitions_total} transitions "
              f"across {n_tracks_evaluated} tracks)")
        print(f"  mean correct-segment length: {mean_seg_len:.1f} frames  "
              f"(longer = better; 1/switch_rate ≈ {1/overall_rate:.1f})" if overall_rate > 0
              else f"  mean correct-segment length: {mean_seg_len:.1f} frames")
        print(f"\n--- Switch breakdown (what causes identity errors?) ---")
        _sw_tot = switch_breakdown['total']
        _wr = switch_breakdown['within_run']
        _gp = switch_breakdown['at_gap']
        print(f"  within consecutive frames: {_wr}  ({100*_wr/max(_sw_tot,1):.0f}%)  ← wrong assignment")
        print(f"  at a gap (non-consecutive): {_gp}  ({100*_gp/max(_sw_tot,1):.0f}%)  ← stitching/re-init failure")
        print(f"  → if within-run dominates: improve per-frame cost matrix")
        print(f"  → if at-gap dominates: improve stitch_tracklets or gap-closing LAP")
        print(f"\n--- Fragmentation (tracks per physical cell) ---")
        print(f"  cells evaluated: {frag_stats['n_cells_evaluated']}  "
              f"median={frag_stats['median']:.1f}  mean={frag_stats['mean']:.1f}  "
              f"p75={frag_stats['pct75']:.1f}  max={frag_stats['max']}")
        print(f"  {100*frag_stats['frac_fragmented']:.0f}% of cells covered by >1 track segment"
              f"  (0% = perfect; high = re-initialisation dominates)")

    return {
        'color_switch_rate':    overall_rate,
        'mean_segment_length':  mean_seg_len,
        'n_tracks_evaluated':   n_tracks_evaluated,
        'n_switches_total':     n_switches_total,
        'n_transitions_total':  n_transitions_total,
        'per_track':            per_track,
        'track_lengths':        length_stats,
        'continuity':           continuity_stats,
        'fragmentation':        frag_stats,
        'switch_breakdown':     switch_breakdown,
    }
