"""Visualization and analysis functions."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
import itertools
import pandas as pd


def visualize_frame_segmentation(frame, prob_map, instances, props,
                               title='Frame Segmentation', figsize=(16, 5), prob_threshold=0.5):
    """Visualize frame with 4-panel segmentation overlay."""

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    ax = axes[0]
    ax.imshow(frame, cmap='gray')
    ax.set_title('Original Frame')
    ax.axis('off')

    ax = axes[1]
    im = ax.imshow(prob_map, cmap='hot')
    ax.set_title(f'UNet Probability (thresh={prob_threshold})')
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, label='P(Cell)')
    cbar.ax.axhline(y=prob_threshold, color='cyan', linewidth=2, linestyle='--')

    ax = axes[2]
    frame_norm = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
    ax.imshow(frame_norm, cmap='gray')

    np.random.seed(42)
    for inst_id in np.unique(instances):
        if inst_id == 0:
            continue

        mask = (instances == inst_id).astype(np.uint8)
        boundary = ndimage.binary_erosion(mask) ^ mask
        boundary_coords = np.where(boundary)

        if len(boundary_coords[0]) > 0:
            color = np.random.rand(3,)
            ax.scatter(boundary_coords[1], boundary_coords[0], c=[color], s=0.1, alpha=1.0)

    ax.set_title(f'Instance Outlines ({instances.max()} cells)')
    ax.axis('off')

    ax = axes[3]
    ax.imshow(frame_norm, cmap='gray')

    np.random.seed(42)
    for cell_id, prop in enumerate(props, 1):
        color = np.random.rand(3,)
        r0, c0, r1, c1 = prop.bbox
        rect = patches.Rectangle((c0, r0), c1-c0, r1-r0, linewidth=2,
                                edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    ax.set_title(f'Cell Bounding Boxes')
    ax.axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def tune_inference_params_multi_frame(model, frames, metrics, param_grid, max_frames=None, device=None):
    """Test parameter grid on multiple frames and return results."""
    from coastal.segment import TwoPassSegmentationInference

    if max_frames:
        frames = frames[:max_frames]
        metrics = metrics[:max_frames]

    # Generate all parameter combinations
    param_keys = list(param_grid.keys())
    param_lists = [param_grid[k] for k in param_keys]

    results = []

    for combo in itertools.product(*param_lists):
        params = dict(zip(param_keys, combo))

        # Test on each frame
        for frame_idx, (frame, metric_dict) in enumerate(zip(frames, metrics)):
            seg = TwoPassSegmentationInference(model=model, device=device, **params)
            prob_map, instances, props = seg.predict_frame(frame, metric_dict)
            n_cells = len(np.unique(instances)) - 1

            row = params.copy()
            row['frame_idx'] = frame_idx
            row['n_cells'] = n_cells
            row['instances'] = instances
            results.append(row)

    return pd.DataFrame(results)


def visualize_parameter_comparison(df, frames, max_combos=10):
    """Visualize top parameter combinations by cell count."""

    # Group by parameter combination (exclude frame_idx, n_cells, instances)
    param_cols = [c for c in df.columns if c not in ['frame_idx', 'n_cells', 'instances']]
    grouped = df.groupby(param_cols)['n_cells'].mean().reset_index()
    grouped = grouped.sort_values('n_cells', ascending=False).head(max_combos)

    n_combos = min(len(grouped), max_combos)
    fig, axes = plt.subplots(n_combos, 2, figsize=(12, 5*n_combos))
    if n_combos == 1:
        axes = axes.reshape(1, -1)

    for row, (idx, combo_row) in enumerate(grouped.iterrows()):
        frame_idx = 0
        frame = frames[frame_idx]
        frame_norm = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)

        # Get instances for this combo
        matching = df[df[param_cols].eq(combo_row[param_cols]).all(axis=1)]
        if len(matching) > 0:
            instances = matching.iloc[0]['instances']
            n_cells = matching.iloc[0]['n_cells']
        else:
            continue

        # Instances
        ax = axes[row, 0]
        ax.imshow(frame_norm, cmap='gray')
        for inst_id in np.unique(instances):
            if inst_id == 0:
                continue
            mask = (instances == inst_id).astype(np.uint8)
            boundary = ndimage.binary_erosion(mask) ^ mask
            coords = np.where(boundary)
            if len(coords[0]) > 0:
                ax.scatter(coords[1], coords[0], c='red', s=0.5, alpha=0.8)
        ax.set_title(f'{n_cells} cells')
        ax.axis('off')

        # Parameters
        ax = axes[row, 1]
        ax.axis('off')
        txt = '\n'.join([f'{k}: {v}' for k, v in combo_row[param_cols].items()])
        ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', family='monospace')

    plt.tight_layout()
    plt.show()


def _plot_rgb_frame(frames_mc, results, frame_idx=0, ch_rgb=(2, 1, 0), figsize=(14, 5)):
    """Helper: create RGB plot for a single frame."""
    r = results[frame_idx]
    instances = r['instances']
    num_cells = r['num_cells']

    # Create RGB composite from channels
    # Handle [T, H, W] (grayscale), [T, H, W, C], and [T, C, H, W]
    if frames_mc.ndim == 3:  # [T, H, W] grayscale
        frame = frames_mc[frame_idx].astype(np.float32)
        # Normalize and repeat to RGB
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-5)
        rgb = np.stack([frame, frame, frame], axis=2)
    elif frames_mc.ndim == 4:
        if frames_mc.shape[1] in (3, 4):  # [T, C, H, W]
            rgb = np.stack([frames_mc[frame_idx, c] for c in ch_rgb], axis=2).astype(np.float32) / 255.0
        else:  # [T, H, W, C]
            rgb = np.stack([frames_mc[frame_idx, :, :, c] for c in ch_rgb], axis=2).astype(np.float32) / 255.0
    else:
        raise ValueError(f"frames_mc must be 3D [T, H, W] or 4D [T, H, W, C] or [T, C, H, W], got shape {frames_mc.shape}")

    rgb = np.clip(rgb, 0, 1)  # Normalize to [0, 1]

    # Create overlay with yellow boundaries
    overlay = rgb.copy()
    boundaries = np.zeros(instances.shape, dtype=bool)
    for label in np.unique(instances[instances > 0]):
        mask = instances == label
        boundaries |= mask & ~ndimage.binary_erosion(mask, iterations=1)
    overlay[boundaries] = [1.0, 1.0, 0.0]  # Yellow

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(rgb)
    axes[0].set_title(f'Frame {frame_idx}\nRGB (R=ch{ch_rgb[0]}, G=ch{ch_rgb[1]}, B=ch{ch_rgb[2]})')
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title(f'{num_cells} instances (yellow boundaries)')
    axes[1].axis('off')

    axes[2].imshow(r['prob_map'], cmap='hot')
    axes[2].set_title('Probability map')
    axes[2].axis('off')

    plt.tight_layout()
    return fig


def plot_rgb_with_segmentation(frames_multi, results, frame_indices=None,
                               ch_rgb=(2, 1, 0), min_cell_size=100,
                               purity_threshold=0.7, figsize=(14, 5)):
    """
    RGB composite with segmentation boundaries color-coded by cell classification.
    Produces one figure per requested frame index.

      Green  — good: large cell with a clearly dominant fluorescent channel
      Red    — merged: large cell with mixed channels (likely two cells fused)
      Orange — fragmented: too small to be a real cell (over-segmentation artefact)

    Args:
        frames_multi:     [T, C, H, W] raw multi-channel frames
        results:          list of result dicts from predict_sequence / predict_frame
        frame_indices:    list of frame indices to plot (default: [0, 4, 8] clipped to len)
        ch_rgb:           (R_ch, G_ch, B_ch) channel indices — default (2,1,0) = tdTomato/GFP/CMAC
        min_cell_size:    pixel threshold below which a label is "fragmented" (default 100)
        purity_threshold: dominant-channel fraction above which a large cell is "good" (default 0.7)
        figsize:          figure size per frame (default (14, 5))
    """
    CLASS_COLORS = {
        'good':       np.array([0.1, 0.9, 0.1]),
        'merged':     np.array([0.95, 0.1, 0.1]),
        'fragmented': np.array([1.0, 0.55, 0.0]),
    }

    frames_arr = np.asarray(frames_multi, dtype=np.float32)

    if frame_indices is None:
        frame_indices = [i for i in [0, 4, 8] if i < len(results)]

    for idx in frame_indices:
        r = results[idx]
        instances = r['instances']
        frame = frames_arr[idx]  # [C, H, W]

        # RGB composite — normalise per-frame to [0, 1]
        C = frame.shape[0]
        rgb_chs = [min(c, C - 1) for c in ch_rgb]
        rgb = np.stack([frame[c] for c in rgb_chs], axis=2)
        rgb = rgb - rgb.min()
        rgb = rgb / (rgb.max() + 1e-8)
        rgb = np.clip(rgb, 0, 1)

        # Classify each label using channel-mean-normalised intensities
        ch_mean = frame.mean(axis=(1, 2))
        frame_norm = frame / (ch_mean[:, None, None] + 1e-6)

        label_class = {}
        for label in np.unique(instances):
            if label == 0:
                continue
            mask = instances == label
            size = int(mask.sum())
            if size < min_cell_size:
                label_class[label] = 'fragmented'
                continue
            mean_ch = frame_norm[:, mask].mean(axis=1)
            total = mean_ch.sum()
            if total < 1e-6:
                label_class[label] = 'fragmented'
                continue
            purity = float((mean_ch / total).max())
            label_class[label] = 'good' if purity >= purity_threshold else 'merged'

        n_good = sum(1 for v in label_class.values() if v == 'good')
        n_merged = sum(1 for v in label_class.values() if v == 'merged')
        n_fragmented = sum(1 for v in label_class.values() if v == 'fragmented')
        n_total = n_good + n_merged + n_fragmented

        # Build colour-coded boundary overlay
        overlay = rgb.copy()
        for label, cls in label_class.items():
            mask = (instances == label).astype(np.uint8)
            boundary = mask.astype(bool) & ~ndimage.binary_erosion(mask, iterations=1)
            overlay[boundary] = CLASS_COLORS[cls]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].imshow(rgb)
        axes[0].set_title(f'Frame {idx}\nRGB (R=ch{rgb_chs[0]}, G=ch{rgb_chs[1]}, B=ch{rgb_chs[2]})')
        axes[0].axis('off')

        axes[1].imshow(overlay)
        good_frac = f'{n_good / n_total:.2f}' if n_total > 0 else 'n/a'
        axes[1].set_title(
            f'{n_total} labels — good={n_good} | merged={n_merged} | frag={n_fragmented}\n'
            f'good fraction = {good_frac}'
        )
        axes[1].axis('off')
        from matplotlib.patches import Patch
        axes[1].legend(handles=[
            Patch(color=CLASS_COLORS['good'],       label=f'good ({n_good})'),
            Patch(color=CLASS_COLORS['merged'],      label=f'merged ({n_merged})'),
            Patch(color=CLASS_COLORS['fragmented'],  label=f'fragmented ({n_fragmented})'),
        ], loc='lower right', framealpha=0.8, fontsize=9)

        axes[2].imshow(r['prob_map'], cmap='hot')
        axes[2].set_title('Probability map')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()


def create_rgb_segmentation_movie(frames_mc, results, output_path='segmentation_movie.mp4',
                                 fps=10, max_frames=None, ch_rgb=(2, 1, 0), figsize=(14, 5)):
    """
    Create RGB composite movie with instance boundaries.

    Args:
        frames_mc: Frames in any of these formats:
                   - [T, H, W] grayscale (repeated to RGB)
                   - [T, H, W, C] multi-channel
                   - [T, C, H, W] multi-channel (channels first)
        results: segmentation results from segment_frames()
        output_path: path for output video file (e.g., 'movie.mp4')
        fps: frames per second for output video
        max_frames: max number of frames to include (None = all)
        ch_rgb: (R_channel, G_channel, B_channel) indices (ignored for grayscale)
                Default: (2, 1, 0) = R=channel2, G=channel1, B=channel0
        figsize: figure size per frame

    Returns:
        None (writes file to disk)

    Example:
        # Grayscale
        create_rgb_segmentation_movie(
            frames,  # [T, H, W]
            results,
            output_path='segmentation_rgb.mp4',
            fps=15
        )

        # Multi-channel
        create_rgb_segmentation_movie(
            frames_multi,  # [T, H, W, C]
            results,
            output_path='segmentation_rgb.mp4',
            fps=15,
            ch_rgb=(2, 1, 0)  # R=tdTomato, G=GFP, B=CMAC
        )
    """
    import cv2
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image

    if max_frames is None:
        max_frames = len(results)

    max_frames = min(max_frames, len(frames_mc), len(results))

    # Get first frame to determine video dimensions
    temp_fig = _plot_rgb_frame(
        frames_mc, results, frame_idx=0, ch_rgb=ch_rgb, figsize=figsize
    )

    # Convert matplotlib figure to image to get dimensions
    buf = BytesIO()
    temp_fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    frame_width, frame_height = img.size
    plt.close(temp_fig)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Creating RGB movie: {output_path}")
    print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")

    # Write frames
    for t in range(max_frames):
        fig = _plot_rgb_frame(
            frames_mc, results, frame_idx=t, ch_rgb=ch_rgb, figsize=figsize
        )

        # Convert matplotlib figure to numpy array (BGR for OpenCV)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Write frame to video
        out.write(img_bgr)

        plt.close(fig)

        if (t + 1) % max(1, max_frames // 10) == 0:
            print(f"  Processed {t + 1}/{max_frames} frames")

    # Release video writer
    out.release()
    print(f"✓ Movie saved to: {output_path}")


def create_classified_segmentation_movie(
    frames_multi, results, output_path='segmentation_classified.mp4',
    fps=10, max_frames=None, ch_rgb=(2, 1, 0),
    min_cell_size=100, purity_threshold=0.7, figsize=(14, 5),
):
    """Create movie with colour-coded segmentation boundaries (green/red/orange).

    Args:
        frames_multi:     [T, C, H, W] multi-channel frames (uint8)
        results:          list of result dicts from predict_sequence
        output_path:      output .mp4 path
        fps:              frames per second
        max_frames:       cap number of frames (None = all)
        ch_rgb:           (R_ch, G_ch, B_ch) channel indices
        min_cell_size:    pixel threshold below which a label is "fragmented"
        purity_threshold: dominant-channel fraction for "good" classification
        figsize:          matplotlib figure size per frame
    """
    import cv2
    from io import BytesIO
    from PIL import Image
    from matplotlib.patches import Patch

    CLASS_COLORS = {
        'good':       np.array([0.1, 0.9, 0.1]),
        'merged':     np.array([0.95, 0.1, 0.1]),
        'fragmented': np.array([1.0, 0.55, 0.0]),
    }

    frames_arr = np.asarray(frames_multi, dtype=np.float32)
    n_frames = min(
        max_frames if max_frames is not None else len(results),
        len(results), len(frames_arr)
    )

    def _render_frame(t):
        r = results[t]
        instances = r['instances']
        frame = frames_arr[t]  # [C, H, W]

        C = frame.shape[0]
        rgb_chs = [min(c, C - 1) for c in ch_rgb]
        rgb = np.stack([frame[c] for c in rgb_chs], axis=2)
        rgb = rgb - rgb.min()
        rgb = rgb / (rgb.max() + 1e-8)
        rgb = np.clip(rgb, 0, 1)

        ch_mean = frame.mean(axis=(1, 2))
        frame_norm = frame / (ch_mean[:, None, None] + 1e-6)

        label_class = {}
        for label in np.unique(instances):
            if label == 0:
                continue
            mask = instances == label
            if int(mask.sum()) < min_cell_size:
                label_class[label] = 'fragmented'
                continue
            mean_ch = frame_norm[:, mask].mean(axis=1)
            total = mean_ch.sum()
            if total < 1e-6:
                label_class[label] = 'fragmented'
                continue
            purity = float((mean_ch / total).max())
            label_class[label] = 'good' if purity >= purity_threshold else 'merged'

        n_good = sum(1 for v in label_class.values() if v == 'good')
        n_merged = sum(1 for v in label_class.values() if v == 'merged')
        n_frag = sum(1 for v in label_class.values() if v == 'fragmented')
        n_total = n_good + n_merged + n_frag

        overlay = rgb.copy()
        for label, cls in label_class.items():
            mask = (instances == label).astype(np.uint8)
            boundary = mask.astype(bool) & ~ndimage.binary_erosion(mask, iterations=1)
            overlay[boundary] = CLASS_COLORS[cls]

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes[0].imshow(rgb)
        axes[0].set_title(f'Frame {t}  RGB (R=ch{rgb_chs[0]}, G=ch{rgb_chs[1]}, B=ch{rgb_chs[2]})')
        axes[0].axis('off')
        axes[1].imshow(overlay)
        good_frac = f'{n_good / n_total:.2f}' if n_total > 0 else 'n/a'
        axes[1].set_title(
            f'{n_total} labels — good={n_good} | merged={n_merged} | frag={n_frag}\n'
            f'good fraction = {good_frac}'
        )
        axes[1].axis('off')
        axes[1].legend(handles=[
            Patch(color=CLASS_COLORS['good'],       label=f'good ({n_good})'),
            Patch(color=CLASS_COLORS['merged'],      label=f'merged ({n_merged})'),
            Patch(color=CLASS_COLORS['fragmented'],  label=f'frag ({n_frag})'),
        ], loc='lower right', framealpha=0.8, fontsize=9)
        axes[2].imshow(r['prob_map'], cmap='hot')
        axes[2].set_title('Probability map')
        axes[2].axis('off')
        plt.tight_layout()
        return fig

    # Probe dimensions from first frame
    probe_fig = _render_frame(0)
    buf = BytesIO()
    probe_fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    frame_width, frame_height = img.size
    plt.close(probe_fig)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Creating classified movie: {output_path}")
    print(f"Dimensions: {frame_width}x{frame_height}, FPS: {fps}, frames: {n_frames}")

    for t in range(n_frames):
        fig = _render_frame(t)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_array = np.array(Image.open(buf))
        out.write(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        plt.close(fig)
        if (t + 1) % max(1, n_frames // 10) == 0:
            print(f"  {t + 1}/{n_frames} frames")

    out.release()
    print(f"✓ Saved: {output_path}")


def create_segmentation_movie(frames, results, output_path='segmentation_movie.mp4',
                             fps=10, max_frames=None, figsize=(16, 5), prob_threshold=0.5):
    """
    Create a movie from segmentation results.
    
    Args:
        frames: [T, H, W] frames
        results: segmentation results from segment_frames()
        output_path: path for output video file (e.g., 'movie.mp4' or 'movie.avi')
        fps: frames per second for output video
        max_frames: max number of frames to include (None = all)
        figsize: figure size per frame
        prob_threshold: probability threshold for visualization
    
    Returns:
        None (writes file to disk)
    
    Example:
        create_segmentation_movie(
            frames_prep, results,
            output_path='segmentation_results.mp4',
            fps=15,
            max_frames=100
        )
    """
    import cv2
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image
    
    if max_frames is None:
        max_frames = len(frames)
    
    max_frames = min(max_frames, len(frames), len(results))
    
    # Get first frame to determine video dimensions
    temp_fig = visualize_frame_segmentation(
        frame=frames[0],
        prob_map=results[0]['prob_map'],
        instances=results[0]['instances'],
        props=results[0]['props'],
        title=f'Frame 0: {results[0]["num_cells"]} cells',
        figsize=figsize,
        prob_threshold=prob_threshold
    )
    
    # Convert matplotlib figure to image to get dimensions
    buf = BytesIO()
    temp_fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    frame_width, frame_height = img.size
    plt.close(temp_fig)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'MJPG' for .avi
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"Creating movie: {output_path}")
    print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")
    
    # Write frames
    for t in range(max_frames):
        frame = frames[t]
        result = results[t]
        
        # Create figure
        fig = visualize_frame_segmentation(
            frame=frame,
            prob_map=result['prob_map'],
            instances=result['instances'],
            props=result['props'],
            title=f'Frame {t}: {result["num_cells"]} cells',
            figsize=figsize,
            prob_threshold=prob_threshold
        )
        
        # Convert matplotlib figure to numpy array (BGR for OpenCV)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Write frame to video
        out.write(img_bgr)
        
        plt.close(fig)
        
        if (t + 1) % 10 == 0:
            print(f"  Processed {t + 1}/{max_frames} frames")
    
    # Release video writer
    out.release()
    print(f"✓ Movie saved to: {output_path}")
