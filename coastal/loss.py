"""Loss functions for training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntensityLoss(nn.Module):
    """Probability guidance using intensity + contrast + edges."""

    def forward(self, pred_prob, frame):
        """
        Args:
            pred_prob: [B, 1, H, W] raw model logits (before sigmoid)
            frame: [B, 1, H, W] intensity images
        """
        # Build target in float32 regardless of AMP dtype
        frame = frame.float()

        frame_mean = frame.view(frame.size(0), -1).mean(dim=1, keepdim=True).view(-1, 1, 1, 1)
        frame_std = frame.view(frame.size(0), -1).std(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-5
        bright = (frame > frame_mean + frame_std).float()

        frame_unfold = F.unfold(frame, kernel_size=5, padding=2)
        local_std = frame_unfold.std(dim=1, keepdim=True) + 1e-5
        contrast = local_std.view_as(frame)

        contrast_mean = contrast.view(contrast.size(0), -1).mean(dim=1, keepdim=True).view(-1, 1, 1, 1)
        contrast_std = contrast.view(contrast.size(0), -1).std(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-5
        contrast_norm = torch.clamp((contrast - contrast_mean) / contrast_std, 0, 1)

        edge_y = torch.abs(frame[:, :, 1:, :] - frame[:, :, :-1, :])
        edge_x = torch.abs(frame[:, :, :, 1:] - frame[:, :, :, :-1])
        edge_y = F.pad(edge_y, (0, 0, 0, 1))
        edge_x = F.pad(edge_x, (0, 1, 0, 0))
        edge = (edge_x + edge_y) / 2

        edge_mean = edge.view(edge.size(0), -1).mean(dim=1, keepdim=True).view(-1, 1, 1, 1)
        edge_std = edge.view(edge.size(0), -1).std(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-5
        edge_norm = torch.clamp((edge - edge_mean) / edge_std, 0, 1)

        cell_target = 0.5 * bright + 0.3 * contrast_norm + 0.2 * edge_norm
        cell_target = cell_target ** 0.5  # push cell regions toward 1.0

        # binary_cross_entropy_with_logits is AMP-safe (fuses sigmoid internally)
        return F.binary_cross_entropy_with_logits(pred_prob.float(), cell_target)


def _contrastive_metric_loss(metric_emb, metrics_dict, k_neighbors, margin, max_pixels=5000):
    B, D, H, W = metric_emb.shape

    total_loss = 0.0

    for b in range(B):
        # Extract metrics for batch b
        if isinstance(metrics_dict, list):
            metrics_dict_b = metrics_dict[b]
        else:
            metrics_dict_b = metrics_dict

        metric_list = []
        for name in sorted(metrics_dict_b.keys()):
            arr = metrics_dict_b[name]
            if isinstance(arr, torch.Tensor):
                tensor = arr.float().to(metric_emb.device)
            else:
                tensor = torch.from_numpy(arr).float().to(metric_emb.device)
            metric_list.append(tensor)

        if not metric_list:
            continue

        metrics_stacked = torch.stack(metric_list, dim=0)  # [M, H, W]
        num_metrics = len(metric_list)

        emb_flat = metric_emb[b].view(D, -1).T  # [H*W, D]
        metrics_flat = metrics_stacked.view(num_metrics, -1).T  # [H*W, M]

        n_pixels = len(metrics_flat)
        if n_pixels > max_pixels:
            indices = torch.randperm(n_pixels, device=metrics_flat.device)[:max_pixels]
            metrics_flat = metrics_flat[indices]
            emb_flat = emb_flat[indices]
            n_pixels = max_pixels

        metrics_norm = F.normalize(metrics_flat, dim=1, p=2)
        emb_norm = F.normalize(emb_flat, dim=1, p=2)

        metric_dist = torch.cdist(metrics_norm, metrics_norm, p=2)
        emb_sim = torch.mm(emb_norm, emb_norm.T)

        k = min(k_neighbors, n_pixels - 1)
        if k < 1:
            continue

        _, sorted_indices = torch.sort(metric_dist, dim=1)
        pos_indices = sorted_indices[:, 1:k + 1]
        neg_indices = sorted_indices[:, -k:]

        pos_sims = emb_sim.gather(1, pos_indices)
        neg_sims = emb_sim.gather(1, neg_indices)

        loss_pos = torch.clamp(1.0 - pos_sims, min=0.0).mean()
        loss_neg = torch.clamp(margin + neg_sims, min=0.0).mean()
        total_loss += loss_pos + loss_neg

    return total_loss / B if B > 0 else torch.tensor(0.0, device=metric_emb.device, requires_grad=True)
  

class TemporalMetricsLoss(nn.Module):
    """Embeddings preserve temporal (optical flow) metric structure via contrastive learning."""

    def __init__(self, k_neighbors=10, margin=0.5, max_pixels=2000):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.margin = margin
        self.max_pixels = max_pixels

    def forward(self, metric_emb, metrics_dict):
        return _contrastive_metric_loss(
            metric_emb, metrics_dict, self.k_neighbors, self.margin, self.max_pixels
        )


class WarpConsistencyLoss(nn.Module):
    """Self-supervised temporal embedding consistency via optical flow warping.

    Pulls emb[t, y, x] toward emb[t+1, y+v, x+u] using bilinear warping.
    Operates on foreground pixels only (prob_t > prob_threshold).
    Uses cosine distance to avoid norm collapse.
    """

    def __init__(self, prob_threshold: float = 0.3):
        super().__init__()
        self.prob_threshold = prob_threshold

    def forward(
        self,
        emb_t:   "torch.Tensor",   # [B, D, H, W]
        emb_t1:  "torch.Tensor",   # [B, D, H, W]
        flow_uv: "torch.Tensor",   # [B, 2, H, W]  [0]=u(x-dir), [1]=v(y-dir) in pixels
        prob_t:  "torch.Tensor",   # [B, 1, H, W]  logits
    ) -> "torch.Tensor":
        B, D, H, W = emb_t.shape
        device = emb_t.device

        # Build identity grid in normalised [-1, 1] coordinates
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')   # [H, W]
        base = torch.stack([grid_x, grid_y], dim=-1)              # [H, W, 2]
        base = base.unsqueeze(0).expand(B, -1, -1, -1)            # [B, H, W, 2]

        # Convert pixel-space displacement to normalised coordinates
        u_norm = flow_uv[:, 0] / (W / 2.0)   # [B, H, W]
        v_norm = flow_uv[:, 1] / (H / 2.0)   # [B, H, W]
        # grid_sample grid[..., 0] = x (column), grid[..., 1] = y (row)
        disp = torch.stack([u_norm, v_norm], dim=-1)  # [B, H, W, 2]

        # Backward-warp: sample emb_t1 at locations shifted by the flow
        grid = (base + disp).clamp(-1, 1)
        emb_t1_warped = F.grid_sample(
            emb_t1.float(), grid, mode='bilinear', align_corners=True, padding_mode='border'
        )  # [B, D, H, W]

        # Foreground mask
        mask = (prob_t.sigmoid() > self.prob_threshold).squeeze(1)  # [B, H, W]
        if not mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        e_t  = emb_t.permute(0, 2, 3, 1)[mask]          # [P, D]
        e_w  = emb_t1_warped.permute(0, 2, 3, 1)[mask]  # [P, D]
        cos  = F.cosine_similarity(e_t.float(), e_w.float(), dim=-1)  # [P]
        return (1.0 - cos).mean()


class VarianceMetricsLoss(nn.Module):
    """Windowed contrastive loss on cross-channel variance metrics.

    Compares pixels only within random spatial windows so the loss sees both
    same-cell positives and cross-cell negatives within each patch.
    """

    def __init__(self, k_neighbors=10, margin=0.5, window_size=32, max_tiles=8):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.margin = margin
        self.window_size = window_size
        self.max_tiles = max_tiles

    def forward(self, metric_emb, metrics_dict, frame_indices=None, max_pixels_per_patch=512):
        B, D, H, W = metric_emb.shape

        pw = min(self.window_size, H, W)
        all_tiles = [(y, x) for y in range(0, max(1, H - pw + 1), pw)
                             for x in range(0, max(1, W - pw + 1), pw)]

        total_loss = torch.tensor(0.0, device=metric_emb.device)

        for b in range(B):
            seed = frame_indices[b] if frame_indices is not None else 0
            if len(all_tiles) > self.max_tiles:
                g = torch.Generator()
                g.manual_seed(seed)
                perm = torch.randperm(len(all_tiles), generator=g)[:self.max_tiles].tolist()
                tiles = [all_tiles[i] for i in perm]
            else:
                tiles = all_tiles
            metrics_dict_b = metrics_dict[b] if isinstance(metrics_dict, list) else metrics_dict
            metric_list = []
            for name in sorted(metrics_dict_b.keys()):
                arr = metrics_dict_b[name]
                tensor = arr.float().to(metric_emb.device) if isinstance(arr, torch.Tensor) \
                    else torch.from_numpy(arr).float().to(metric_emb.device)
                metric_list.append(tensor)

            if not metric_list:
                continue

            met_b = torch.stack(metric_list, dim=0)  # [M, H, W]
            M = len(metric_list)
            patch_losses = []

            for y0, x0 in tiles:
                emb_patch = metric_emb[b, :, y0:y0+pw, x0:x0+pw].reshape(D, -1).T  # [pw², D]
                met_patch = met_b[:, y0:y0+pw, x0:x0+pw].reshape(M, -1).T          # [pw², M]

                n_px = met_patch.shape[0]
                if n_px > max_pixels_per_patch:
                    step = max(1, n_px // max_pixels_per_patch)
                    idx = torch.arange(0, n_px, step, device=met_patch.device)[:max_pixels_per_patch]
                    met_patch = met_patch[idx]
                    emb_patch = emb_patch[idx]
                    n_px = len(idx)

                k = min(self.k_neighbors, n_px - 1)
                if k < 1:
                    continue

                met_norm = F.normalize(met_patch, dim=1, p=2)
                emb_norm = F.normalize(emb_patch, dim=1, p=2)

                met_dist = torch.cdist(met_norm, met_norm, p=2)
                emb_sim = torch.mm(emb_norm, emb_norm.T)

                _, sorted_idx = torch.sort(met_dist, dim=1)
                neg_sims = emb_sim.gather(1, sorted_idx[:, -k:])

                loss_neg = torch.clamp(self.margin + neg_sims, min=0.0).mean()
                patch_losses.append(loss_neg)

            if patch_losses:
                total_loss = total_loss + torch.stack(patch_losses).mean()

        return total_loss / B
