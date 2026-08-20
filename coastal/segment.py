"""Inference classes for 2D and 3D segmentation.

Streamlined merging logic: embeddings + prob_map regions.
- Region growing: embedding affinity + optional prob_map relaxation
- Merging: embedding affinity + prob_map gate (bright regions easier to merge)

See TUNING_GUIDE.md for decision-tree parameter tuning.
"""

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import (binary_dilation, gaussian_filter, maximum_filter,
                           distance_transform_edt, find_objects)
from skimage.measure import regionprops

from coastal.device import resolve_device

from coastal.utils import match_masks_3d


# How often the GPU region-growing loop asks whether it has converged. Every `accept.any()` is a
# device sync, and the loop is otherwise a few microseconds of elementwise kernels, so asking every
# iteration cost more than the work. 8 bounds the wasted passes to 7; an iteration that accepts
# nothing writes nothing, so overshooting is free of consequence beyond its own time.
_GROW_CHECK_EVERY = 8


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
                 prob_blur_sigma=0.0,
                 seed_blur_sigma=0.0,
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
            
            seed_blur_sigma: gaussian sigma for the SEED map only, leaving the outline sharp.
                0.0 (default) = seeds come from the same map as the mask, i.e. previous behaviour.
                Raise it when the prob map fragments each cell into several components: the count
                has a hard floor at the component count, and blurring `prob_blur_sigma` to fix that
                rounds the boundary away. Cell-diameter-ish is a sensible start.

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
        self.seed_blur_sigma = seed_blur_sigma
        self.min_boundary_pixels = min_boundary_pixels
        self.merge_contact_brightness_threshold = merge_contact_brightness_threshold
        
        # Optional tuning (4)
        self.prob_threshold = prob_threshold
        self.embedding_blur_sigma = embedding_blur_sigma
        self.prob_blur_sigma = prob_blur_sigma
        self.max_iter = max_iter
        self.min_component_size = min_component_size

    def predict_frame(self, frame, metrics_dict, variance_metrics=None):
        """Segment using seed-based region growing with embedding+prob-based merging.

        Args:
            frame:            [H, W] projected frame
            metrics_dict:     the temporal (flow) metrics for this frame
            variance_metrics: optional cross-channel (confetti) metrics from
                              `flow.compute_variance_metrics`. Training feeds these as the
                              trailing input channels — `[frame, sorted(temporal),
                              sorted(variance)]` — with channel dropout so the model also
                              works without them, and inference has historically passed
                              zeros there. Supplying them here fills those channels with the
                              real signal instead, in the same order training used.
        """
        frame_norm = (frame - frame.min()) / (frame.max() - frame.min() + 1e-5)
        frame_tensor = torch.from_numpy(frame_norm).float().unsqueeze(0).unsqueeze(0)

        def _stack(d):
            out = []
            for name in sorted(d.keys()):
                arr = d[name]
                out.append(torch.from_numpy(arr).float() if isinstance(arr, np.ndarray)
                           else arr.float())
            return out

        # Order matters: training concatenates the temporal block then the variance block,
        # each sorted within itself (see train.py::_stack_metrics), NOT one merged sort.
        metric_list = _stack(metrics_dict)
        n_temporal = len(metric_list)
        if variance_metrics:
            metric_list += _stack(variance_metrics)

        H, W = frame_tensor.shape[2:]
        if metric_list:
            metrics_stacked = torch.stack(metric_list, dim=0).unsqueeze(0)
        else:
            # ZERO channels, not one. A `torch.zeros(1, 1, H, W)` here fabricates a metric plane
            # that the model never had, so a model trained with no flow metrics at all (in_channels=1
            # — a flow ablation) is handed 2 channels and the first conv raises. The zero-fill below
            # is the only thing allowed to add channels, and it is driven by `model.num_metrics`.
            metrics_stacked = torch.zeros(1, 0, H, W)

        # Zero-fill whatever the model expects beyond what was supplied.
        n_variance = max(0, self.model.num_metrics - len(metric_list))
        variance_zeros = torch.zeros(1, n_variance, H, W)
        frame_and_metrics = torch.cat([frame_tensor, metrics_stacked, variance_zeros], dim=1).to(self.device)

        with torch.no_grad():
            prob, embeddings = self.model(frame_and_metrics)
            prob_map = torch.sigmoid(prob)[0, 0].cpu().numpy()
            emb_np = embeddings[0].permute(1, 2, 0).cpu().numpy()

        # Speckle suppression. The prob head resolves cells (~15-20 px blobs) but sits on a
        # 1-3 px noise floor that also crosses prob_threshold, so the foreground mask comes
        # out as ~3600 blobs of median 3 px per frame. Cells and speckle differ by scale, so
        # a blur at cell scale separates them where a threshold cannot.
        #
        # Judge this at matched FOREGROUND AREA, not at matched threshold: the blur lowers every
        # prob value, so a fixed threshold silently runs it at a stricter operating point and
        # makes it look worse than it is. At equal area (4 movies x 2 z-planes x 3 frames, recall
        # scored against raw-grey cell seeds), sigma=1 beats sigma=0 outright — 70.5% vs 68.1%
        # recall at 1% foreground, with half the blobs (47 vs 104). Larger sigma trades the low-
        # area end for the high: sigma=3 gives 60.9% at 1% area but 90.7% vs 87.8% at 5%, with 16x
        # fewer blobs.
        #
        # This composes with, rather than replaces, cleaning the INPUT — see
        # denoise.denoise_preserving_ratio. Denoised input plus sigma=1 is the best measured
        # combination (77.2% recall at 1% area, 42 blobs). See docs/SEGMENTATION.md.
        prob_raw = prob_map
        if self.prob_blur_sigma > 0:
            prob_map = gaussian_filter(prob_map, sigma=self.prob_blur_sigma)

        # SEEDING and the OUTLINE want opposite amounts of blur, and until `seed_blur_sigma` they
        # shared `prob_blur_sigma`.
        #
        #   Seeding wants HEAVY blur. Seeds are local maxima of the prob map, and every connected
        #   component of the thresholded map is GUARANTEED one (below) — so the instance count has
        #   a hard floor at the component count. On photon-limited intravital data that floor was
        #   ~64 components against ~30 real cells, and no `seed_size` gets under it: `seed_size`
        #   only removes surplus seeds WITHIN a component.
        #
        #   The outline wants NONE. The same blur that fuses those components also rounds the
        #   boundary: on zolIMa/fXgbTl mem-TOM, sigma=8 gave the right cell count at solidity 0.97
        #   and circularity 0.79-0.86 — ellipses drawn near the cells rather than their shape. For
        #   a cytoplasmic reporter the shape IS the readout.
        #
        # So seeds come from `seed_map`, the mask and outline from `prob_map`. Measured on that
        # data: identical counts (42/31), circularity 0.79 -> 0.54, solidity 0.97 -> 0.87.
        # Default 0.0 reproduces the previous behaviour exactly (seed_map IS prob_map).
        seed_map = (gaussian_filter(prob_raw, sigma=self.seed_blur_sigma)
                    if self.seed_blur_sigma > 0 else prob_map)

        H, W, D = emb_np.shape

        binary = prob_map > self.prob_threshold
        if not binary.any():
            return prob_map, np.zeros((H, W), dtype=np.int32), []

        seed_binary = seed_map > self.prob_threshold
        local_max = maximum_filter(seed_map, size=self.seed_size) == seed_map
        # a seed outside the growth mask can never grow, so intersect with it
        seeds_binary = local_max & seed_binary & binary

        # Every connected component of the SEED map must have at least one seed. Scoped to each
        # component's bounding box: the brightest pixel of a component is inside its own box, and
        # row-major order within the box preserves the whole-frame argmax tie-break, so the seed
        # chosen is the same one.
        #
        # Deliberately the seed map's components, not the mask's. When the mask is sharper than the
        # seed map it breaks into speckle fragments containing no seed — those SHOULD go unlabelled
        # rather than each being handed its own cell, which is the fragmentation this parameter
        # exists to stop.
        components, n_components = ndimage.label(seed_binary)
        for comp_id, sl in enumerate(find_objects(components), start=1):
            comp_mask = components[sl] == comp_id
            if not seeds_binary[sl][comp_mask].any():
                # restrict to the growth mask, else the forced seed is stranded outside it
                scored = np.where(comp_mask & binary[sl], seed_map[sl], -1.0)
                if scored.max() < 0:
                    continue
                local = np.unravel_index(scored.argmax(), scored.shape)
                seeds_binary[local[0] + sl[0].start, local[1] + sl[1].start] = True

        seeds, n_seeds = ndimage.label(seeds_binary)

        if n_seeds == 0:
            return prob_map, np.zeros((H, W), dtype=np.int32), []

        emb_norm = emb_np / (np.linalg.norm(emb_np, axis=2, keepdims=True) + 1e-5)

        # One filter call over [H, W, D] with sigma=0 on the channel axis (an identity
        # kernel) instead of D separate 2D calls — same separable passes per channel,
        # without the per-channel Python/scipy overhead (D is 64 in practice).
        emb_smoothed = gaussian_filter(
            emb_norm, sigma=(self.embedding_blur_sigma, self.embedding_blur_sigma, 0))

        instances = self._grow_regions_fast(emb_smoothed, seeds, binary, prob_map)

        instances = self._fill_holes(instances)

        if self.merge_max_distance > 0:
            instances = self._merge_split_instances(instances, emb_smoothed, prob_map)

        props = regionprops(instances)

        return prob_map, instances, props

    # The four 4-connected neighbour offsets, in the order the affinity tie-break depends on:
    # a pixel takes the FIRST neighbour that strictly beats the running best, so reordering this
    # changes which label an equal-affinity pixel joins. Shared by both region-growing paths so the
    # GPU one cannot drift from the CPU one.
    _GROW_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def _grow_regions_fast(self, embeddings, seeds, mask, prob_map):
        """REGION GROWING: expand seeds based on embedding affinity (vectorized).

        Processes all boundary pixels simultaneously with numpy instead of a
        Python pixel loop — releases the GIL and is ~10-50× faster on large images.

        Dispatches to `_grow_regions_torch` on a CUDA device, which is the same algorithm laid out
        densely instead of over the boundary index set — measured bit-identical, 3.6x faster. See
        that method for why dense is the wrong shape for the numpy path and the right one here.
        """
        if getattr(self.device, 'type', str(self.device)).startswith('cuda'):
            return self._remove_small_components(
                self._grow_regions_torch(embeddings, seeds, mask, prob_map))

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

    def _grow_regions_torch(self, embeddings, seeds, mask, prob_map):
        """`_grow_regions_fast` on the GPU. Same result, bit for bit — verified, not assumed.

        Two changes, both of which only pay off on a GPU:

        * **The four affinity maps are computed ONCE.** A pixel's affinity to its neighbour in a
          given direction is a function of the embedding field alone — nothing in the loop touches
          it — yet the numpy path recomputes it every iteration, and there are up to `max_iter`
          (200) of those. Hoisting it out is not a win on CPU: numpy gathers only the current
          BOUNDARY pixels, a sparse subset, so computing all four maps densely costs more than the
          repeated sparse gathers (310 ms/plane vs 263 measured). On a GPU the dense form is the
          cheap one and the hoist is nearly free.
        * **The convergence check is batched.** `if not accept.any()` forces a device sync, and at
          three syncs an iteration that dominated the port's first draft (111 ms/plane, only 2.3x).
          Checking every `_GROW_CHECK_EVERY` iterations instead costs at most that many no-op
          passes — an iteration with nothing to accept leaves `instances` untouched, so the extra
          passes cannot change the result, only the time.

        Tie-breaking, the strict `>` comparisons and the direction order are kept exactly as the
        numpy path has them; `tests/test_grow_regions_parity.py` pins the two together on real
        embeddings. 73 ms/plane vs 263 on a 420x441 frame (RTX 2000 Ada).
        """
        dev = self.device
        emb = torch.as_tensor(np.ascontiguousarray(embeddings), device=dev)
        instances = torch.as_tensor(np.ascontiguousarray(seeds).astype(np.int32), device=dev)
        mask_t = torch.as_tensor(np.ascontiguousarray(mask), device=dev)
        required = (self.affinity_threshold
                    - torch.as_tensor(np.ascontiguousarray(prob_map), device=dev)
                    * self.prob_weight)

        def shifted(x, dh, dw):
            """`x` at offset `(dh, dw)` — i.e. each pixel's neighbour — zero outside the frame.

            Zero-fill is what makes the bounds check disappear: an out-of-frame neighbour gets
            label 0, and every use below already requires a label > 0.
            """
            out = torch.zeros_like(x)
            hs = slice(max(dh, 0), x.shape[0] + min(dh, 0))
            hd = slice(max(-dh, 0), x.shape[0] + min(-dh, 0))
            ws = slice(max(dw, 0), x.shape[1] + min(dw, 0))
            wd = slice(max(-dw, 0), x.shape[1] + min(-dw, 0))
            out[hd, wd] = x[hs, ws]
            return out

        affinities = [(shifted(emb, dh, dw) * emb).sum(-1) for dh, dw in self._GROW_DIRS]

        for it in range(self.max_iter):
            labeled = instances > 0
            unlabeled = (~labeled) & mask_t

            # binary_dilation with the cross structuring element, as four in-place ors
            dilated = labeled.clone()
            dilated[1:] |= labeled[:-1]
            dilated[:-1] |= labeled[1:]
            dilated[:, 1:] |= labeled[:, :-1]
            dilated[:, :-1] |= labeled[:, 1:]
            boundary = unlabeled & dilated

            best_affinities = torch.full(instances.shape, -1.0, device=dev)
            best_labels = torch.zeros_like(instances)
            for (dh, dw), affinity in zip(self._GROW_DIRS, affinities):
                neighbour_labels = shifted(instances, dh, dw)
                update = boundary & (neighbour_labels > 0) & (affinity > best_affinities)
                best_affinities = torch.where(update, affinity, best_affinities)
                best_labels = torch.where(update, neighbour_labels, best_labels)

            accept = boundary & (best_affinities > required) & (best_labels > 0)
            instances = torch.where(accept, best_labels, instances)

            if it % _GROW_CHECK_EVERY == _GROW_CHECK_EVERY - 1 and not accept.any():
                break

        return instances.cpu().numpy().astype(np.int32)

    def _fill_holes(self, instances):
        """Fill holes in instance labels (donut-shaped regions become solid).

        Each label is processed inside its own bounding box (`find_objects`) with a 1-px
        zero border, which reproduces the whole-frame result exactly: the label lies
        entirely within its box, so a background component that escapes the box also
        escapes to the frame edge, and neither gets filled. Labels are still applied in
        ascending order, so a hole containing a lower-numbered label is overwritten just
        as before. Whole-frame `binary_fill_holes` per label was ~45% of segmentation
        time at ~600 labels/frame.
        """
        from scipy.ndimage import binary_fill_holes, find_objects

        instances_filled = instances.copy()

        for inst_id, sl in enumerate(find_objects(instances), start=1):
            if sl is None:                      # label id absent from this frame
                continue

            sub = instances[sl] == inst_id
            padded = np.zeros((sub.shape[0] + 2, sub.shape[1] + 2), dtype=bool)
            padded[1:-1, 1:-1] = sub
            filled = binary_fill_holes(padded)[1:-1, 1:-1]
            instances_filled[sl][filled] = inst_id   # basic slice -> writes through

        return instances_filled

    def _merge_split_instances(self, instances, embeddings, prob_map):
        """
        Merge nearby fragments using AND logic:

        Merge if:
        1. Sufficient contact (n_contact >= min_boundary_pixels)
        2. Contact is bright (contact_prob > merge_contact_brightness_threshold)
        3. Embeddings are similar (affinity > merge_affinity_threshold)

        Both brightness AND affinity must pass.

        Per-fragment work happens inside a padded bounding box instead of on the whole
        frame (whole-frame EDTs per fragment were ~30% of segmentation time at ~600
        fragments/frame). Every predicate below thresholds a distance at
        <= max(1.0, merge_max_distance), so the box only has to be wide enough that those
        distances match their whole-frame values: a pixel within merge_max_distance of
        *both* fragments puts the other fragment's nearest pixel within 2x that of this
        one, so `pad` covers it. Beyond the box the true distance exceeds the threshold,
        and a cropped transform can only over-estimate, so the predicates stay False
        either way.
        """
        unique_ids = np.unique(instances)
        unique_ids = unique_ids[unique_ids > 0]

        pad = int(np.ceil(max(1.0, 2.0 * self.merge_max_distance))) + 1
        boxes = find_objects(instances)
        merges = {}  # id_remove → id_keep

        for inst_id in unique_ids:
            sl = boxes[inst_id - 1]
            if sl is None:
                continue
            box = tuple(slice(max(0, s.start - pad), min(dim, s.stop + pad))
                        for s, dim in zip(sl, instances.shape))

            inst_box = instances[box]
            prob_box = prob_map[box]
            mask = inst_box == inst_id
            dist1 = distance_transform_edt(~mask)

            # Candidates: pixels of other fragments within merge_max_distance
            candidate_pixels = (dist1 <= self.merge_max_distance) & ~mask & (inst_box > 0)
            neighbors = np.unique(inst_box[candidate_pixels])

            for neighbor_id in neighbors:
                if neighbor_id in merges:
                    continue

                contact_prob, _, _ = self._compute_boundary_intensity(
                    inst_box, prob_box, inst_id, neighbor_id, dist1
                )
                n_contact = self._count_contact_pixels(inst_box, inst_id, neighbor_id, dist1)

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
        """Compute cosine similarity between mean embeddings of two fragments.

        Takes the **whole-frame** instances/embeddings on purpose: the means are over every
        pixel of each fragment, so unlike the distance predicates in
        `_merge_split_instances` this one cannot be evaluated on a bounding-box crop.
        """
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
        """Remove instances smaller than min_component_size and reindex.

        Sizes come from one `bincount` and the relabel is a single lookup-table gather,
        replacing two full-frame comparisons per label (~600 of them per frame). Surviving
        labels are still renumbered 1..n in ascending order of their old id, so the output
        is identical to the per-label loop.
        """
        counts = np.bincount(instances.ravel())
        keep = counts >= self.min_component_size
        keep[0] = False                                    # background is not a label

        lut = np.zeros(counts.size, dtype=instances.dtype)
        surviving = np.flatnonzero(keep)                   # ascending old ids
        lut[surviving] = np.arange(1, surviving.size + 1, dtype=instances.dtype)

        return lut[instances]

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
                 seed_blur_sigma_large=0.0,
                 merge_max_distance_large=1.5,
                 merge_affinity_threshold_large=0.65,
                 prob_weight_large=0.3,
                 merge_contact_brightness_threshold_large=0.60,
                 seed_size_small=10,
                 affinity_threshold_small=0.4,
                 embedding_blur_sigma_small=1.5,
                 seed_blur_sigma_small=0.0,
                 merge_max_distance_small=1.5,
                 merge_affinity_threshold_small=0.60,
                 prob_weight_small=0.3,
                 merge_contact_brightness_threshold_small=0.60,
                 max_iter=100,
                 min_component_size=20,
                 min_component_size_small=None,
                 min_boundary_pixels=1):
        """
        Two-pass inference with different parameters for large and small cells.

        Args:
            model: trained UNet model
            device: cuda or cpu
            prob_threshold: probability threshold for both passes

            seed_blur_sigma_large / seed_blur_sigma_small: per-pass `seed_blur_sigma` (see
                LearnedAffinityInference). The two passes want opposite values on photon-limited
                data: the large pass needs heavy seed blur to stop one cell fragmenting into
                several, while the small pass is looking FOR small objects and must not have them
                blurred together. 0.0 on both = previous behaviour.

            min_component_size_small: size floor for pass 2 only (default: same as
                `min_component_size`). Apoptotic bodies are legitimately far smaller than a cell,
                so the floor that suppresses speckle in pass 1 also deletes them in pass 2.

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
            seed_blur_sigma=seed_blur_sigma_large,
        )

        self.pass2 = LearnedAffinityInference(
            model=model,
            device=device,
            prob_threshold=prob_threshold,
            seed_size=seed_size_small,
            affinity_threshold=affinity_threshold_small,
            max_iter=max_iter,
            min_component_size=(min_component_size if min_component_size_small is None
                                else min_component_size_small),
            embedding_blur_sigma=embedding_blur_sigma_small,
            merge_max_distance=merge_max_distance_small,
            merge_affinity_threshold=merge_affinity_threshold_small,
            prob_weight=prob_weight_small,
            merge_contact_brightness_threshold=merge_contact_brightness_threshold_small,
            min_boundary_pixels=min_boundary_pixels,
            seed_blur_sigma=seed_blur_sigma_small,
        )

    def predict_frame(self, frame, metrics_dict, return_provenance=False):
        """Two-pass segmentation: large cells → small fragments.

        Args:
            return_provenance: also return a [H, W] uint8 map — 0 background, 1 pass 1 (cells),
                2 pass 2 (small objects). The merged labels alone cannot answer "which pass found
                this", and object SIZE is not a substitute: pass 1 can return a small object and
                pass 2 a larger one. Returned rather than stashed on `self` because the 3D/4D
                paths call this from several threads on one instance.
        """
        prob_map, instances_pass1, props1 = self.pass1.predict_frame(frame, metrics_dict)

        mask_remaining = (instances_pass1 == 0) & (prob_map > self.prob_threshold)

        if not mask_remaining.any():
            if return_provenance:
                return (prob_map, instances_pass1, props1,
                        (instances_pass1 > 0).astype(np.uint8))
            return prob_map, instances_pass1, props1

        prob_map_p2, instances_pass2, props2 = self.pass2.predict_frame(frame, metrics_dict)
        instances_pass2[~mask_remaining] = 0

        max_label_p1 = instances_pass1.max()
        instances_pass2[instances_pass2 > 0] += max_label_p1

        instances_merged = instances_pass1.copy()
        instances_merged[mask_remaining] = instances_pass2[mask_remaining]

        props_merged = regionprops(instances_merged)

        if return_provenance:
            provenance = np.zeros(instances_merged.shape, dtype=np.uint8)
            provenance[instances_pass1 > 0] = 1
            provenance[mask_remaining & (instances_pass2 > 0)] = 2
            return prob_map, instances_merged, props_merged, provenance

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
        keep_results=False,
    ):
        """Segment a 4D volume [T, C, Z, Y, X] over time.

        For each z-slice: computes temporal optical flow then runs predict_sequence.
        For each timepoint: stitches Z labels with IOU matching so cells have
        consistent IDs across Z-planes.

        Memory: `volume` should be passed **lazily** (a dask array straight from
        cecelia) — only one z-slice is materialised at a time, so `np.asarray(volume)`
        at the call site defeats the streaming and costs T×C×Z×H×W×2 bytes. The two
        terms that do scale are:

          * per worker  — the z-slice's flow fields (multi-scale + cumulative, ~1.8 GB
            at T=180, 531×586) held for the lifetime of that slice. The 14 metric planes
            per frame are computed on demand by `flow.TemporalMetrics`, so only the
            current frame's are resident. `n_workers` multiplies this, so it is still
            the dominant cost (see docs/SEGMENTATION.md).
          * the output  — the [T, Z, H, W] int32 label buffer, written in place.

        Per-slice labels are copied into that buffer and the rest of each
        `predict_sequence` result (prob maps, regionprops) is dropped as soon as the
        slice finishes; pass `keep_results=True` to retain them for inspection, which
        costs a further ~2× the output buffer plus one regionprops list per frame.

        Args:
            volume:            [T, C, Z, Y, X] array; lazy (dask) is preferred
            ch_indices:        channel indices to use for projection/flow (None = all)
            stitch_threshold:  min IOU for label matching across Z (default 0.0)
            gap_tolerance:     bridge chains broken by up to this many bad slices (default 1)
            gap_iou_threshold: min IOU to accept a gap bridge (default 0.3)
            temporal_scales:   Farneback multi-scale parameters
            cumulative_window: cumulative displacement window
            n_workers:         z-slices segmented concurrently (memory multiplier)
            keep_results:      keep the full per-slice result dicts (default False)

        Returns:
            instances_4d: [T, Z, H, W] int32 matched instance labels
            results_per_z: list of Z predict_sequence result lists (one per z-slice),
                           or None when keep_results is False
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from coastal.flow import prepare_data_for_unet, normalize_and_project

        T, C, Z, H, W = volume.shape  # dask-safe: no full load

        print(f"\n4D Temporal Segmentation: {T} timepoints × {Z} z-slices × {H}×{W} px")
        print(f"  Using {n_workers} parallel workers\n")

        # Single buffer for the labels: filled per z-slice, then stitched in place.
        instances_4d = np.zeros((T, Z, H, W), dtype=np.int32)
        results_per_z = [None] * Z if keep_results else None

        def _process_z(z):
            print(f"  Z {z+1:2d}/{Z}: computing flow...", flush=True)
            seq = np.asarray(volume[:, :, z, :, :])  # load one z-slice
            _, frames_proj = normalize_and_project(seq, ch_indices)
            del seq

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

            # Keep only what the stitching below needs; release the prob maps and
            # regionprops of this slice before the next one is loaded.
            for t, r in enumerate(results_z):
                instances_4d[t, z] = r['instances']
            avg_cells = np.mean([r['num_cells'] for r in results_z])
            print(f"  Z {z+1:2d}/{Z}: done — {avg_cells:.0f} cells/frame avg", flush=True)
            return z, (results_z if keep_results else None)

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_process_z, z): z for z in range(Z)}
            for future in as_completed(futures):
                z, results_z = future.result()
                if keep_results:
                    results_per_z[z] = results_z

        print(f"\n  Stitching Z labels at each of {T} timepoints...")

        # match_masks_3d copies its input, so writing the result back into the same
        # timepoint is safe; timepoints are independent, hence the thread pool.
        def _stitch_t(t):
            instances_4d[t] = match_masks_3d(
                instances_4d[t],
                stitch_threshold=stitch_threshold,
                gap_tolerance=gap_tolerance,
                gap_iou_threshold=gap_iou_threshold,
            )
            return t

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for _t in ex.map(_stitch_t, range(T)):
                pass

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
