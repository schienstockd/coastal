"""Inference classes for 2D and 3D segmentation.

Streamlined merging logic: embeddings + prob_map regions.
- Region growing: embedding affinity + optional prob_map relaxation
- Merging: embedding affinity + prob_map gate (bright regions easier to merge)

See TUNING_GUIDE.md for decision-tree parameter tuning.
"""

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import binary_dilation, gaussian_filter, maximum_filter, distance_transform_edt
from skimage.measure import regionprops

from coastal.device import resolve_device

from coastal.utils import match_masks_3d


class LearnedAffinityInference:
    """Fast region growing with fragment merging guided by embeddings + probability map.
    
    PARAMETER SET (11 total)
    ═══════════════════════════════════════════════════════════════════════════════
    
    CORE TUNING (5 parameters, start here):
    
    ▶ REGION GROWING (pixels expand from seeds):
    
      affinity_threshold [0.3–0.7]
        Base cosine similarity required to grow into neighbor pixel.
        Higher → more conservative, fewer pixels recruited, more fragments.
        Lower → more aggressive, pixels bleed into wrong regions.
        START: 0.5
    
      prob_weight [0.0–0.5]
        [REGION GROWING ONLY] Relaxes affinity_threshold in bright regions.
        Formula: required = affinity_threshold - prob[pixel] * prob_weight
        0.0 = off (strict) | 0.3 = mild | 0.5 = strong (lenient)
        Use when cells are touching/clustering (bright pixels help growth).
        Lower this if distinct cells are being connected during growth.
        START: 0.3
    
    ▶ FRAGMENT MERGING (fragments combine after growing):
    
      merge_affinity_threshold [0.5–0.8]
        Cosine similarity required to merge two fragments.
        Embeddings must be similar. Higher → stricter, cells stay split.
        START: 0.65

      merge_max_distance [0.5–3.0]
        Max pixel distance between fragments to search for merge candidates.
        Larger → search farther, more aggressive merging.
        START: 1.5

      merge_contact_brightness_threshold [0.5–0.7]
        [MERGING ONLY] Contact region must be bright (> this) to merge.
        Combined with affinity threshold: both must pass (AND logic).
        Higher → stricter, fewer merges. Increase to prevent false merges.
        START: 0.60
    
    SECONDARY TUNING (2 parameters, tune after core):

      seed_size [10–15]
        Initial seed finding window diameter (pixels).
        Larger → fewer seeds → fewer initial fragments.
        START: 12

      min_boundary_pixels [1–3]
        Minimum contact pixels required to consider merging.
        Hard floor (soft gate via contact brightness threshold).
        START: 1
    
    OPTIONAL TUNING (4 parameters, rarely needed):
    
      prob_threshold [0.2–0.5]
        Binary mask cutoff. Pixels below never grown into.
        Raise to ignore dim/noisy regions.
        START: 0.3
    
      embedding_blur_sigma [0.5–2.5]
        Gaussian blur on embeddings.
        Larger → smoother, more merging. Smaller → respect fine details.
        START: 1.5
    
      max_iter [50–500]
        Region growing iterations. Usually stops naturally before max.
        START: 200
    
      min_component_size [5–50]
        Discard fragments smaller than this.
        Larger → more aggressive noise filtering.
        START: 20
    
    ═══════════════════════════════════════════════════════════════════════════════
    
    TUNING WORKFLOW (see TUNING_GUIDE.md):
    
    1. Run with defaults
    2. Are cells SPLIT (fragmenting)?
       → Decrease merge_contact_brightness_threshold or merge_affinity_threshold
       → Or increase merge_max_distance
    3. Are cells MERGED (unrelated touching)?
       → Increase merge_affinity_threshold or affinity_threshold
       → Or increase merge_contact_brightness_threshold
    4. Both issues?
       → Fix split first (step 2), then merged (step 3)
    5. One parameter change per test
    
    ═══════════════════════════════════════════════════════════════════════════════
    """

    def __init__(self, model, device=None,
                 affinity_threshold=0.5,
                 merge_affinity_threshold=0.65,
                 merge_max_distance=1.5,
                 prob_weight=0.3,
                 merge_contact_brightness_threshold=0.60,
                 seed_size=12,
                 min_boundary_pixels=1,
                 prob_threshold=0.3,
                 embedding_blur_sigma=1.5,
                 max_iter=200,
                 min_component_size=20):
        """
        Args:
            model: trained UNet
            device: 'cuda' or 'cpu'
            
            ▶ REGION GROWING PARAMETERS:
            
            affinity_threshold: base cosine similarity to grow pixels [0.3–0.7]
                Higher → more conservative, more fragments.
                Lower → more aggressive, may bleed into wrong cells.
                Default: 0.5
            
            prob_weight: [REGION GROWING ONLY] relax threshold in bright regions [0.0–0.5]
                Formula: required = affinity_threshold - prob[pixel] * prob_weight
                0.0 = off (strict) | 0.3 = mild | 0.5 = strong (lenient)
                Use when cells touch/cluster (bright pixels help expand).
                Lower if distinct cells connect during growing.
                Default: 0.3
            
            ▶ FRAGMENT MERGING PARAMETERS:
            
            merge_affinity_threshold: base cosine similarity to merge fragments [0.5–0.8]
                Usually keep ≥ affinity_threshold.
                Higher → stricter, cells stay split.
                Lower → lenient, cells merge more.
                Default: 0.65
            
            merge_max_distance: max pixel distance for merge candidates [0.5–3.0]
                Larger → search farther for fragments. Default: 1.5
            
            merge_contact_brightness_threshold: [MERGING ONLY] contact brightness required [0.5–0.7]
                Contact region must be bright (> this) to merge.
                Combined with affinity threshold: both must pass (AND logic).
                Higher → stricter, fewer merges.
                Default: 0.60
            
            SECONDARY TUNING PARAMETERS (2):
            
            seed_size: seed finding window size [10–15]
                Larger → fewer seeds → fewer fragments but may miss small cells.
                Default: 12
            
            min_boundary_pixels: minimum contact pixels required to consider merging [1–3]
                Soft floor; contact brightness is the primary filter. Default: 1

            NOTE: merging uses AND logic (contact brightness ≥ threshold AND embedding affinity
            ≥ merge_affinity_threshold), NOT a weighted "hybrid score". Earlier docstrings here
            documented `contact_ratio_threshold` / `hybrid_score_threshold` / `contact_prob_weight`
            / `contact_ratio_weight` / `affinity_weight` — none of those are constructor params;
            that scoring scheme was abandoned. See `_merge_split_instances`.

            OPTIONAL TUNING PARAMETERS (4):
            
            prob_threshold: binary mask cutoff [0.2–0.5]
                Pixels below this prob are never grown into.
                Raise to ignore dim/noisy regions. Default: 0.3
            
            embedding_blur_sigma: gaussian blur on embeddings [0.5–2.5]
                Larger → smoother embeddings, more merging.
                Smaller → respect fine embedding details. Default: 1.5
            
            max_iter: region growing iterations [50–500]
                Usually stops naturally before max. Rarely needs tuning. Default: 200
            
            min_component_size: discard fragments < this size [5–50]
                Larger → more aggressive filtering of noise. Default: 20
        """
        device = resolve_device(device)
        self.model = model.to(device).eval()
        self.device = device

        # Core tuning (4)
        self.affinity_threshold = affinity_threshold
        self.merge_affinity_threshold = merge_affinity_threshold
        self.merge_max_distance = merge_max_distance
        self.prob_weight = prob_weight
        
        # Secondary tuning (2)
        self.seed_size = seed_size
        self.min_boundary_pixels = min_boundary_pixels
        self.merge_contact_brightness_threshold = merge_contact_brightness_threshold
        
        # Optional tuning (4)
        self.prob_threshold = prob_threshold
        self.embedding_blur_sigma = embedding_blur_sigma
        self.max_iter = max_iter
        self.min_component_size = min_component_size

    def predict_frame(self, frame, metrics_dict):
        """Segment using seed-based region growing with embedding+prob-based merging."""
        frame_norm = (frame - frame.min()) / (frame.max() - frame.min() + 1e-5)
        frame_tensor = torch.from_numpy(frame_norm).float().unsqueeze(0).unsqueeze(0)

        metric_list = []
        for name in sorted(metrics_dict.keys()):
            arr = metrics_dict[name]
            if isinstance(arr, np.ndarray):
                arr = torch.from_numpy(arr).float()
            else:
                arr = arr.float()
            metric_list.append(arr)

        H, W = frame_tensor.shape[2:]
        if metric_list:
            metrics_stacked = torch.stack(metric_list, dim=0).unsqueeze(0)
        else:
            metrics_stacked = torch.zeros(1, 1, H, W)

        # Pad with zeros for variance channels not available at inference.
        n_variance = max(0, self.model.num_metrics - len(metric_list))
        variance_zeros = torch.zeros(1, n_variance, H, W)
        frame_and_metrics = torch.cat([frame_tensor, metrics_stacked, variance_zeros], dim=1).to(self.device)

        with torch.no_grad():
            prob, embeddings = self.model(frame_and_metrics)
            prob_map = torch.sigmoid(prob)[0, 0].cpu().numpy()
            emb_np = embeddings[0].permute(1, 2, 0).cpu().numpy()

        H, W, D = emb_np.shape

        binary = prob_map > self.prob_threshold
        if not binary.any():
            return prob_map, np.zeros((H, W), dtype=np.int32), []

        local_max = maximum_filter(prob_map, size=self.seed_size) == prob_map
        seeds_binary = local_max & binary

        # Every connected component above prob_threshold must have at least one seed.
        components, n_components = ndimage.label(binary)
        for comp_id in range(1, n_components + 1):
            comp_mask = components == comp_id
            if not seeds_binary[comp_mask].any():
                peak = np.unravel_index(
                    np.where(comp_mask, prob_map, -1).argmax(), prob_map.shape
                )
                seeds_binary[peak] = True

        seeds, n_seeds = ndimage.label(seeds_binary)

        if n_seeds == 0:
            return prob_map, np.zeros((H, W), dtype=np.int32), []

        emb_norm = emb_np / (np.linalg.norm(emb_np, axis=2, keepdims=True) + 1e-5)

        emb_smoothed = np.zeros_like(emb_norm)
        for d in range(emb_norm.shape[2]):
            emb_smoothed[:, :, d] = gaussian_filter(emb_norm[:, :, d], sigma=self.embedding_blur_sigma)

        instances = self._grow_regions_fast(emb_smoothed, seeds, binary, prob_map)

        instances = self._fill_holes(instances)

        if self.merge_max_distance > 0:
            instances = self._merge_split_instances(instances, emb_smoothed, prob_map)

        props = regionprops(instances)

        return prob_map, instances, props

    def _grow_regions_fast(self, embeddings, seeds, mask, prob_map):
        """REGION GROWING: expand seeds based on embedding affinity (vectorized).

        Processes all boundary pixels simultaneously with numpy instead of a
        Python pixel loop — releases the GIL and is ~10-50× faster on large images.
        """
        H, W, D = embeddings.shape
        instances = seeds.copy()

        for _ in range(self.max_iter):
            unlabeled = (instances == 0) & mask
            if not unlabeled.any():
                break

            labeled_dilated = binary_dilation(instances > 0)
            boundary = unlabeled & labeled_dilated
            bh, bw = np.where(boundary)
            if len(bh) == 0:
                break

            best_labels = np.zeros(len(bh), dtype=instances.dtype)
            best_affinities = np.full(len(bh), -1.0, dtype=np.float32)

            for dh, dw in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nh = bh + dh
                nw = bw + dw
                in_bounds = (nh >= 0) & (nh < H) & (nw >= 0) & (nw < W)
                if not in_bounds.any():
                    continue

                labeled_mask = np.zeros(len(bh), dtype=bool)
                labeled_mask[in_bounds] = instances[nh[in_bounds], nw[in_bounds]] > 0
                valid = in_bounds & labeled_mask
                if not valid.any():
                    continue

                curr_emb = embeddings[bh[valid], bw[valid]]        # [N, D]
                neigh_emb = embeddings[nh[valid], nw[valid]]        # [N, D]
                aff = (curr_emb * neigh_emb).sum(axis=1)            # [N]
                neigh_labels = instances[nh[valid], nw[valid]]

                valid_idx = np.where(valid)[0]
                better = aff > best_affinities[valid_idx]
                update = valid_idx[better]
                best_affinities[update] = aff[better]
                best_labels[update] = neigh_labels[better]

            required = self.affinity_threshold - prob_map[bh, bw] * self.prob_weight
            accept = (best_affinities > required) & (best_labels > 0)
            if not accept.any():
                break

            instances[bh[accept], bw[accept]] = best_labels[accept]

        instances = self._remove_small_components(instances)
        return instances

    def _fill_holes(self, instances):
        """Fill holes in instance labels (donut-shaped regions become solid)."""
        from scipy.ndimage import binary_fill_holes

        instances_filled = instances.copy()

        for inst_id in np.unique(instances):
            if inst_id == 0:
                continue

            mask = (instances == inst_id).astype(np.uint8)
            mask_filled = binary_fill_holes(mask).astype(np.uint8)
            instances_filled[mask_filled == 1] = inst_id

        return instances_filled

    def _merge_split_instances(self, instances, embeddings, prob_map):
        """
        Merge nearby fragments using AND logic:

        Merge if:
        1. Sufficient contact (n_contact >= min_boundary_pixels)
        2. Contact is bright (contact_prob > merge_contact_brightness_threshold)
        3. Embeddings are similar (affinity > merge_affinity_threshold)

        Both brightness AND affinity must pass.
        """
        unique_ids = np.unique(instances)
        unique_ids = unique_ids[unique_ids > 0]

        merges = {}  # id_remove → id_keep

        for inst_id in unique_ids:
            mask = instances == inst_id
            dist1 = distance_transform_edt(~mask)

            # Candidates: pixels of other fragments within merge_max_distance
            candidate_pixels = (dist1 <= self.merge_max_distance) & ~mask & (instances > 0)
            neighbors = np.unique(instances[candidate_pixels])

            for neighbor_id in neighbors:
                if neighbor_id in merges:
                    continue

                contact_prob, _, _ = self._compute_boundary_intensity(
                    instances, prob_map, inst_id, neighbor_id, dist1
                )
                n_contact = self._count_contact_pixels(instances, inst_id, neighbor_id, dist1)

                # Minimum contact requirement
                if n_contact < self.min_boundary_pixels:
                    continue

                # AND logic: both brightness AND affinity must pass
                if contact_prob < self.merge_contact_brightness_threshold:
                    continue

                affinity = self._compute_fragment_affinity(instances, embeddings, inst_id, neighbor_id)
                if affinity < self.merge_affinity_threshold:
                    continue

                # Both conditions met: merge
                id_keep, id_remove = min(inst_id, neighbor_id), max(inst_id, neighbor_id)
                merges[id_remove] = id_keep

        # Resolve transitive chains before applying
        def find_root(x):
            while x in merges:
                x = merges[x]
            return x

        instances_merged = instances.copy()
        for id_remove in merges:
            instances_merged[instances_merged == id_remove] = find_root(id_remove)

        instances_merged = self._remove_small_components(instances_merged)

        return instances_merged

    def _compute_fragment_affinity(self, instances, embeddings, id1, id2):
        """Compute cosine similarity between mean embeddings of two fragments."""
        mask1 = (instances == id1)
        mask2 = (instances == id2)

        emb1_mean = embeddings[mask1].mean(axis=0)
        emb2_mean = embeddings[mask2].mean(axis=0)

        emb1_mean = emb1_mean / (np.linalg.norm(emb1_mean) + 1e-5)
        emb2_mean = emb2_mean / (np.linalg.norm(emb2_mean) + 1e-5)

        affinity = np.dot(emb1_mean, emb2_mean)
        return affinity

    def _compute_boundary_intensity(self, instances, prob_map, id1, id2, dist1=None):
        """
        Compute brightness of contact and gap regions between two fragments.
        
        Returns (contact_prob, gap_prob, has_gap) where:
        - contact_prob: mean prob at direct touching pixels (1px contact)
        - gap_prob: mean prob in the gap region between fragments
        - has_gap: boolean, True if there's a gap (not directly touching)
        
        Used by _merge_split_instances:
        - If has_gap=True and gap_prob is bright: allow merge (bridge between fragments)
        - If has_gap=False: don't relax (touching doesn't indicate same cell)
        """
        mask1 = instances == id1
        mask2 = instances == id2

        if dist1 is None:
            dist1 = distance_transform_edt(~mask1)
        dist2 = distance_transform_edt(~mask2)

        # Gap: pixels between fragments, within merge_max_distance of both
        gap = (dist1 <= self.merge_max_distance) & (dist2 <= self.merge_max_distance) & ~mask1 & ~mask2

        if not gap.any():
            # Directly touching: no gap between them
            contact_mask = (dist1 <= 1.0) & mask2
            if contact_mask.any():
                contact_prob = float(np.clip(prob_map[contact_mask].mean(), 0, 1))
                return contact_prob, contact_prob, False  # has_gap=False
            return 0.5, 0.5, False

        # Gap exists: compute gap and contact probs
        gap_prob = float(np.clip(prob_map[gap].mean(), 0, 1))
        contact_mask = (dist1 <= 1.0) & mask2
        if contact_mask.any():
            contact_prob = float(np.clip(prob_map[contact_mask].mean(), 0, 1))
        else:
            contact_prob = gap_prob

        return contact_prob, gap_prob, True  # has_gap=True

    def _count_contact_pixels(self, instances, id1, id2, dist1):
        """Count minimum contact pixels between two fragments."""
        mask2 = instances == id2
        # Pixels of id2 that are within 1px of id1
        contact_mask = (dist1 <= 1.0) & mask2
        return int(contact_mask.sum())

    def _remove_small_components(self, instances):
        """Remove instances smaller than min_component_size and reindex."""
        unique_labels = np.unique(instances)
        for label_id in unique_labels:
            if label_id == 0:
                continue
            if (instances == label_id).sum() < self.min_component_size:
                instances[instances == label_id] = 0

        unique_labels = np.unique(instances)
        instances_reindexed = np.zeros_like(instances)
        for new_id, old_id in enumerate(unique_labels[1:], 1):
            instances_reindexed[instances == old_id] = new_id

        return instances_reindexed

    def predict_volume_3d(self, volume_3d, metrics_3d):
        """
        Segment 3D volume by processing each Z-slice independently,
        then matching labels across Z using IOU overlap.

        Args:
            volume_3d: [Z, H, W] 3D volume
            metrics_3d: list of Z metric dicts (one per slice)

        Returns:
            instances_3d_matched: [Z, H, W] with consistent labels across Z
            results_per_slice: list of Z result dicts
        """
        Z = volume_3d.shape[0]
        masks_2d_list = []
        results_per_slice = []

        print(f"\nProcessing 3D volume: {Z} slices")

        for z in range(Z):
            frame = volume_3d[z]
            metrics = metrics_3d[z] if z < len(metrics_3d) else {}

            prob_map, instances, props = self.predict_frame(frame, metrics)

            masks_2d_list.append(instances)
            results_per_slice.append({
                'z': z,
                'prob_map': prob_map,
                'instances': instances,
                'props': props,
                'num_cells': len(props)
            })

            if z % max(1, Z // 10) == 0 or z == Z - 1:
                print(f"  Slice {z}/{Z-1}: {len(props)} cells")

        instances_3d = np.stack(masks_2d_list, axis=0)
        instances_3d_matched = match_masks_3d(instances_3d, stitch_threshold=0.0)

        return instances_3d_matched, results_per_slice

    def predict_sequence(self, frames, temporal_metrics_norm, show_progress=True):
        """
        Segment 2D temporal sequence (e.g., time-lapse movie).
        For 3D volumes, use predict_volume_3d instead.

        Args:
            frames: [T, H, W] temporal sequence
            temporal_metrics_norm: list of T metric dicts

        Returns:
            results: list of T result dicts
        """
        from tqdm import tqdm
        results = []

        it = tqdm(enumerate(frames), total=len(frames), desc="Segmenting", leave=False) \
            if show_progress else enumerate(frames)

        for t, frame in it:
            metrics = temporal_metrics_norm[t] if t < len(temporal_metrics_norm) else {}
            prob_map, instances, props = self.predict_frame(frame, metrics)

            results.append({
                'prob_map': prob_map,
                'instances': instances,
                'props': props,
                'num_cells': len(props),
                'frame_idx': t,
            })

        return results

    def update_params(self, **kwargs):
        """Update any parameters (e.g., for parameter sweeps/tuning)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE: All parameters with defaults (start here)
# ═══════════════════════════════════════════════════════════════════════════════

# segmentor = LearnedAffinityInference(
#     model=model,
#     device='cuda' if torch.cuda.is_available() else 'cpu',
#
#     # ▶ REGION GROWING (pixels expand from seeds)
#     affinity_threshold=0.5,         # Growing strictness [0.3–0.7]
#     prob_weight=0.3,                # Relax in bright regions [0.0–0.5]
#
#     # ▶ FRAGMENT MERGING (AND logic: contact bright AND embeddings similar)
#     merge_max_distance=1.5,         # Fragment search radius [0.5–3.0]
#     merge_affinity_threshold=0.65,  # Embeddings must agree [0.5–0.8]
#     merge_contact_brightness_threshold=0.60,  # Contact must be bright [0.5–0.7]
#     min_boundary_pixels=1,          # Min contact pixels [1–3]
#
#     # SECONDARY TUNING (1) — rarely needed
#     seed_size=12,                   # Seed window size [10–15]
#
#     # OPTIONAL TUNING (4) — rarely needed
#     prob_threshold=0.3,             # Binary mask cutoff [0.2–0.5]
#     embedding_blur_sigma=1.5,       # Embedding smoothness [0.5–2.5]
#     max_iter=200,                   # Region growing iterations [50–500]
#     min_component_size=20,          # Min fragment size [5–50]
# )
#
# TUNING WORKFLOW (see TUNING_GUIDE.md for decision tree):
#
# If cells are SPLIT (fragmenting):
#   → Decrease merge_contact_brightness_threshold or merge_affinity_threshold
#   → Or increase merge_max_distance (search farther)
#
# If cells are MERGED (distinct cells connected):
#   → Increase merge_affinity_threshold
#   → Or increase merge_contact_brightness_threshold
#
# One parameter change per test. Track what helps.

# ═══════════════════════════════════════════════════════════════════════════════


class TwoPassSegmentationInference:
    """Two-pass instance segmentation: large cells, then small fragments.
    
    Both passes use dual-criteria merging (affinity + intensity continuity).
    """

    def __init__(self, model, device=None,
                 prob_threshold=0.3,
                 seed_size_large=24,
                 affinity_threshold_large=0.7,
                 embedding_blur_sigma_large=1.5,
                 merge_max_distance_large=1.5,
                 merge_affinity_threshold_large=0.65,
                 prob_weight_large=0.3,
                 merge_contact_brightness_threshold_large=0.60,
                 seed_size_small=10,
                 affinity_threshold_small=0.4,
                 embedding_blur_sigma_small=1.5,
                 merge_max_distance_small=1.5,
                 merge_affinity_threshold_small=0.60,
                 prob_weight_small=0.3,
                 merge_contact_brightness_threshold_small=0.60,
                 max_iter=100,
                 min_component_size=20,
                 min_boundary_pixels=1):
        """
        Two-pass inference with different parameters for large and small cells.

        Args:
            model: trained UNet model
            device: cuda or cpu
            prob_threshold: probability threshold for both passes

            # Pass 1 parameters (large cells)
            seed_size_large: seed window size
            affinity_threshold_large: affinity threshold for region growing
            merge_max_distance_large: max Euclidean pixel gap for fragment merge candidates
            merge_affinity_threshold_large: affinity threshold for merging fragments
            prob_weight_large: region-growing relaxation — how much pixel prob lowers the
                required grow affinity (required = affinity_threshold - prob * prob_weight; 0 = off)
            merge_contact_brightness_threshold_large: contact must be bright to merge [0.5–0.7]

            # Pass 2 parameters (small fragments)
            seed_size_small: seed window size
            affinity_threshold_small: affinity threshold for region growing
            merge_max_distance_small: max Euclidean pixel gap for fragment merge candidates
            merge_affinity_threshold_small: affinity threshold for merging fragments
            prob_weight_small: region-growing relaxation — how much pixel prob lowers the
                required grow affinity (required = affinity_threshold - prob * prob_weight; 0 = off)
            merge_contact_brightness_threshold_small: contact must be bright to merge [0.5–0.7]

            # Shared
            max_iter: max iterations for region growing
            min_component_size: minimum component size to keep
            min_boundary_pixels: minimum contact pixels required (default 1)
        """
        device = resolve_device(device)
        self.model = model
        self.device = device
        self.prob_threshold = prob_threshold

        self.pass1 = LearnedAffinityInference(
            model=model,
            device=device,
            prob_threshold=prob_threshold,
            seed_size=seed_size_large,
            affinity_threshold=affinity_threshold_large,
            max_iter=max_iter,
            min_component_size=min_component_size,
            embedding_blur_sigma=embedding_blur_sigma_large,
            merge_max_distance=merge_max_distance_large,
            merge_affinity_threshold=merge_affinity_threshold_large,
            prob_weight=prob_weight_large,
            merge_contact_brightness_threshold=merge_contact_brightness_threshold_large,
            min_boundary_pixels=min_boundary_pixels,
        )

        self.pass2 = LearnedAffinityInference(
            model=model,
            device=device,
            prob_threshold=prob_threshold,
            seed_size=seed_size_small,
            affinity_threshold=affinity_threshold_small,
            max_iter=max_iter,
            min_component_size=min_component_size,
            embedding_blur_sigma=embedding_blur_sigma_small,
            merge_max_distance=merge_max_distance_small,
            merge_affinity_threshold=merge_affinity_threshold_small,
            prob_weight=prob_weight_small,
            merge_contact_brightness_threshold=merge_contact_brightness_threshold_small,
            min_boundary_pixels=min_boundary_pixels,
        )

    def predict_frame(self, frame, metrics_dict):
        """Two-pass segmentation: large cells → small fragments."""
        prob_map, instances_pass1, props1 = self.pass1.predict_frame(frame, metrics_dict)

        mask_remaining = (instances_pass1 == 0) & (prob_map > self.prob_threshold)

        if not mask_remaining.any():
            return prob_map, instances_pass1, props1

        prob_map_p2, instances_pass2, props2 = self.pass2.predict_frame(frame, metrics_dict)
        instances_pass2[~mask_remaining] = 0

        max_label_p1 = instances_pass1.max()
        instances_pass2[instances_pass2 > 0] += max_label_p1

        instances_merged = instances_pass1.copy()
        instances_merged[mask_remaining] = instances_pass2[mask_remaining]

        props_merged = regionprops(instances_merged)

        return prob_map, instances_merged, props_merged

    def predict_sequence(self, frames, temporal_metrics_norm, show_progress=True):
        """Segment 2D temporal sequence with two-pass approach."""
        from tqdm import tqdm
        results = []

        it = tqdm(enumerate(frames), total=len(frames), desc="Segmenting", leave=False) \
            if show_progress else enumerate(frames)

        for t, frame in it:
            metrics = temporal_metrics_norm[t] if t < len(temporal_metrics_norm) else {}
            prob_map, instances, props = self.predict_frame(frame, metrics)

            results.append({
                'prob_map': prob_map,
                'instances': instances,
                'props': props,
                'num_cells': len(props),
                'frame_idx': t,
            })

        return results

    def predict_volume_3d(self, volume_3d, metrics_3d):
        """
        Segment 3D volume with two-pass approach on each slice,
        then match labels across Z using IOU overlap.

        Args:
            volume_3d: [Z, H, W] 3D volume
            metrics_3d: list of Z metric dicts (one per slice)

        Returns:
            instances_3d_matched: [Z, H, W] with consistent labels across Z
            results_per_slice: list of Z result dicts
        """
        Z = volume_3d.shape[0]
        masks_2d_list = []
        results_per_slice = []

        print(f"\nProcessing 3D volume (two-pass): {Z} slices")

        for z in range(Z):
            frame = volume_3d[z]
            metrics = metrics_3d[z] if z < len(metrics_3d) else {}

            prob_map, instances, props = self.predict_frame(frame, metrics)

            masks_2d_list.append(instances)
            results_per_slice.append({
                'z': z,
                'prob_map': prob_map,
                'instances': instances,
                'props': props,
                'num_cells': len(props)
            })

            if z % max(1, Z // 10) == 0 or z == Z - 1:
                print(f"  Slice {z}/{Z-1}: {len(props)} cells")

        instances_3d = np.stack(masks_2d_list, axis=0)
        instances_3d_matched = match_masks_3d(instances_3d, stitch_threshold=0.0)

        return instances_3d_matched, results_per_slice

    def update_params(self, pass_num=None, **kwargs):
        """Update parameters for pass 1, pass 2, or both (if pass_num=None)."""
        target = [self.pass1, self.pass2] if pass_num is None else [self.pass1 if pass_num == 1 else self.pass2]
        for engine in target:
            engine.update_params(**kwargs)


class Inference3D:
    """
    3D instance segmentation: 2D per-slice + IOU-based label matching.

    Workflow:
    1. Segment each Z-slice independently
    2. Match labels across Z dimension using IOU-based stitching
    3. Return 3D instance map with consistent labels
    """

    def __init__(self, model, device=None, **inference_kwargs):
        """
        Args:
            model: trained UNet model
            device: torch device; None/'auto' → cuda→mps→cpu (see coastal.device.resolve_device)
            **inference_kwargs: passed to LearnedAffinityInference
        """
        device = resolve_device(device)
        self.model = model
        self.device = device
        self.inferencer_2d = LearnedAffinityInference(model, device=device, **inference_kwargs)

    def predict_stack(self, stack, metrics_stack, stitch_threshold=0.0):
        """
        Segment 3D stack and match labels across Z.

        Args:
            stack: [Z, H, W] image stack
            metrics_stack: list of Z dicts (one per slice), computed per-stack
            stitch_threshold: minimum IOU for label matching (default 0.0)

        Returns:
            instances_3d: [Z, H, W] matched instance labels
            results: list of Z result dicts
        """
        results = []
        masks_2d = []

        print(f"\n3D Segmentation: Processing {len(stack)} slices...")

        for z, (frame, metrics) in enumerate(zip(stack, metrics_stack)):
            prob_map, instances, props = self.inferencer_2d.predict_frame(frame, metrics)

            results.append({
                'prob_map': prob_map,
                'instances': instances,
                'props': props,
                'num_cells': len(props),
                'slice_idx': z,
            })

            masks_2d.append(instances)
            print(f"  Slice {z}: {len(props)} cells")

        print(f"Matching labels across Z dimension...")
        masks_2d_matched = match_masks_3d(masks_2d, stitch_threshold=stitch_threshold)

        instances_3d = np.stack(masks_2d_matched, axis=0)

        for z, instances_matched in enumerate(masks_2d_matched):
            props = regionprops(instances_matched)
            results[z]['instances'] = instances_matched
            results[z]['props'] = props
            results[z]['num_cells'] = len(props)

        print(f"3D segmentation complete. Total unique labels: {len(np.unique(instances_3d)) - 1}\n")

        return instances_3d, results

    def predict_temporal_volume(
        self,
        volume,
        ch_indices=None,
        stitch_threshold=0.0,
        gap_tolerance=1,
        gap_iou_threshold=0.3,
        temporal_scales=[1, 2, 4],
        cumulative_window=2,
        n_workers=4,
    ):
        """Segment a 4D volume [T, C, Z, Y, X] over time.

        For each z-slice: computes temporal optical flow then runs predict_sequence.
        For each timepoint: stitches Z labels with IOU matching so cells have
        consistent IDs across Z-planes.

        Args:
            volume:            [T, C, Z, Y, X] array
            ch_indices:        channel indices to use for projection/flow (None = all)
            stitch_threshold:  min IOU for label matching across Z (default 0.0)
            gap_tolerance:     bridge chains broken by up to this many bad slices (default 1)
            gap_iou_threshold: min IOU to accept a gap bridge (default 0.3)
            temporal_scales:   Farneback multi-scale parameters
            cumulative_window: cumulative displacement window

        Returns:
            instances_4d: [T, Z, H, W] int32 matched instance labels
            results_per_z: list of Z predict_sequence result lists (one per z-slice)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from coastal.flow import prepare_data_for_unet, normalize_and_project

        T, C, Z, H, W = volume.shape  # dask-safe: no full load

        print(f"\n4D Temporal Segmentation: {T} timepoints × {Z} z-slices × {H}×{W} px")
        print(f"  Using {n_workers} parallel workers\n")

        def _process_z(z):
            print(f"  Z {z+1:2d}/{Z}: computing flow...", flush=True)
            seq = np.asarray(volume[:, :, z, :, :])  # load one z-slice
            _, frames_proj = normalize_and_project(seq, ch_indices)

            frames_prep, _, _, temporal_metrics = prepare_data_for_unet(
                frames_proj,
                temporal_scales=temporal_scales,
                cumulative_window=cumulative_window,
                verbose=False,
            )

            print(f"  Z {z+1:2d}/{Z}: segmenting...", flush=True)
            results_z = self.inferencer_2d.predict_sequence(
                frames_prep, temporal_metrics, show_progress=False
            )
            avg_cells = np.mean([r['num_cells'] for r in results_z])
            print(f"  Z {z+1:2d}/{Z}: done — {avg_cells:.0f} cells/frame avg", flush=True)
            return z, results_z

        results_per_z = [None] * Z
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_process_z, z): z for z in range(Z)}
            for future in as_completed(futures):
                z, results_z = future.result()
                results_per_z[z] = results_z

        print(f"\n  Stitching Z labels at each of {T} timepoints...")
        instances_4d = np.zeros((T, Z, H, W), dtype=np.int32)

        def _stitch_t(t):
            masks_at_t = [results_per_z[z][t]['instances'] for z in range(Z)]
            return t, match_masks_3d(
                masks_at_t,
                stitch_threshold=stitch_threshold,
                gap_tolerance=gap_tolerance,
                gap_iou_threshold=gap_iou_threshold,
            )

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for t, masks_matched in ex.map(_stitch_t, range(T)):
                for z, inst in enumerate(masks_matched):
                    instances_4d[t, z] = inst

        n_cells_per_t = [int(len(np.unique(instances_4d[t])) - 1) for t in range(T)]
        print(f"  Done. Cells/timepoint: min={min(n_cells_per_t)}, "
              f"max={max(n_cells_per_t)}, mean={np.mean(n_cells_per_t):.0f}\n")

        return instances_4d, results_per_z

    def predict_movies(self, movies, movies_metrics, stitch_threshold=0.0):
        """
        Segment multiple 3D stacks.

        Args:
            movies: list of [Z, H, W] stacks
            movies_metrics: list of lists (metrics per-stack)
            stitch_threshold: minimum IOU for matching

        Returns:
            results_all: list of (instances_3d, results_list) tuples
        """
        results_all = []

        for movie_idx, (stack, metrics_stack) in enumerate(zip(movies, movies_metrics)):
            print(f"\n{'='*80}")
            print(f"Movie {movie_idx}")
            print(f"{'='*80}")

            instances_3d, results = self.predict_stack(stack, metrics_stack, stitch_threshold)
            results_all.append((instances_3d, results))

        return results_all
