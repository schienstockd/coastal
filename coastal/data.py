"""
Prepare temporal metrics and flows for UNet training with loss weight ablation.
Handles the conversion from prepare_data_for_unet output to training format.
"""

import numpy as np


def prepare_training_data(frames_prep, multi_scale_flows, temporal_metrics, use_scale=None):
    """
    Convert outputs from prepare_data_for_unet to training format.
    
    Args:
        frames_prep: [T, H, W] normalized frames
        multi_scale_flows: dict of {scale: [flow_dict, ...]}
        temporal_metrics: [T, {metric_name: [H, W]}]
    
    Returns:
        flows_dict: {frame_idx: {'u': [H,W], 'v': [H,W]}} (finest scale)
        metrics: {metric_name: [T, H, W]}
        temporal_metrics_norm: [T, {metric_name: [H, W]}] normalized
        valid_metric_names: list of metric names with ndim==2
    """
    
    T, H, W = frames_prep.shape
    
    # ====================================================================
    # 1. EXTRACT FLOWS AT FINEST SCALE
    # ====================================================================
    
    # Find finest scale (smallest scale value)
    if use_scale is None:
        use_scale = min(multi_scale_flows.keys())  # finest by default
    finest_flows = multi_scale_flows[use_scale]
    
    flows_dict = {}
    
    # Map flows to frame indices
    # Each flow corresponds to a frame pair (i, i+scale)
    # We'll assign the flow to frame i
    for flow_data in finest_flows:
        i_start, i_end = flow_data['frame_pair']
        flows_dict[i_start] = {
            'u': flow_data['u'].astype(np.float32),
            'v': flow_data['v'].astype(np.float32),
        }
    
    # Fill missing frames with zero flow (e.g., first frame)
    for frame_idx in range(T):
        if frame_idx not in flows_dict:
            flows_dict[frame_idx] = {
                'u': np.zeros((H, W), dtype=np.float32),
                'v': np.zeros((H, W), dtype=np.float32),
            }
    
    print(f"✓ Extracted flows for {len(flows_dict)} frames at scale {use_scale}")
    
    # ====================================================================
    # 2. CONVERT METRICS FROM LIST TO DICT FORMAT
    # ====================================================================
    
    # Input: [T, {metric_name: [H, W]}]
    # Output: {metric_name: [T, H, W]}
    
    metric_names = list(temporal_metrics[0].keys())
    metrics = {}
    
    for metric_name in metric_names:
        metric_stack = []
        for frame_idx in range(T):
            metric_data = temporal_metrics[frame_idx][metric_name]
            
            # Handle both numpy arrays and torch tensors
            if hasattr(metric_data, 'cpu'):  # torch tensor
                metric_data = metric_data.cpu().numpy()
            
            metric_stack.append(metric_data)
        
        metrics[metric_name] = np.array(metric_stack, dtype=np.float32)
    
    print(f"✓ Converted {len(metric_names)} metrics to stacked format")
    
    # ====================================================================
    # 3. NORMALIZE METRICS PER-FRAME
    # ====================================================================
    
    temporal_metrics_norm = []
    
    for frame_idx in range(T):
        frame_metrics_norm = {}
        
        for metric_name in metric_names:
            arr = temporal_metrics[frame_idx][metric_name].astype(np.float32)
            
            # Normalize to [0, 1] per metric, per frame
            arr_min = arr.min()
            arr_max = arr.max()
            
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min + 1e-8)
            else:
                arr = np.zeros_like(arr)
            
            arr = np.clip(arr, 0, 1).astype(np.float32)
            frame_metrics_norm[metric_name] = arr
        
        temporal_metrics_norm.append(frame_metrics_norm)
    
    print(f"✓ Normalized metrics for {T} frames")
    
    # ====================================================================
    # 4. FILTER TO 2D METRICS ONLY
    # ====================================================================
    
    valid_metric_names = [k for k in temporal_metrics_norm[0].keys() 
                          if temporal_metrics_norm[0][k].ndim == 2]
    
    print(f"✓ Selected {len(valid_metric_names)} 2D metrics:")
    for name in sorted(valid_metric_names):
        print(f"    - {name}")
    
    return flows_dict, metrics, temporal_metrics_norm, valid_metric_names


def validate_training_data(frames_prep, flows_dict, temporal_metrics_norm, valid_metric_names):
    """Validate shapes and types."""
    
    T, H, W = frames_prep.shape
    
    print(f"\n{'='*70}")
    print(f"DATA VALIDATION")
    print(f"{'='*70}\n")
    
    # Frames
    print(f"Frames: {frames_prep.shape} {frames_prep.dtype}")
    print(f"  Range: [{frames_prep.min():.4f}, {frames_prep.max():.4f}]")
    
    # Flows
    print(f"\nFlows: {len(flows_dict)} frame pairs")
    for frame_idx in [0, T//2, T-1]:
        if frame_idx in flows_dict:
            u = flows_dict[frame_idx]['u']
            v = flows_dict[frame_idx]['v']
            mag = np.sqrt(u**2 + v**2)
            print(f"  Frame {frame_idx}: u={u.shape} v={v.shape}, mag=[{mag.min():.3f}, {mag.max():.3f}]")
    
    # Metrics
    print(f"\nMetrics: {len(valid_metric_names)} selected")
    for metric_name in sorted(valid_metric_names)[:5]:
        metric_data = temporal_metrics_norm[0][metric_name]
        print(f"  {metric_name}: {metric_data.shape} {metric_data.dtype} [{metric_data.min():.3f}, {metric_data.max():.3f}]")
    
    print(f"\n✓ Data validation complete")


if __name__ == "__main__":
    print("""
    DATA PREPARATION FOR UNet TRAINING
    ===================================
    
    Usage:
    ------
    from prepare_data_for_training import prepare_training_data, validate_training_data
    
    flows_dict, metrics, temporal_metrics_norm, valid_metric_names = prepare_training_data(
        frames_prep=frames_prep,
        multi_scale_flows=flows,
        temporal_metrics=temporal_metrics
    )
    
    validate_training_data(frames_prep, flows_dict, temporal_metrics_norm, valid_metric_names)
    
    # Now use for training
    results = run_loss_weight_ablation(
        frames=frames_prep,
        flows=flows_dict,
        metrics=temporal_metrics_norm,
        metric_names=valid_metric_names,
        loss_weight_combos=loss_weight_combos,
        num_epochs=50,
        batch_size=2,
        seed=42,
        device='cuda'
    )
    """)
