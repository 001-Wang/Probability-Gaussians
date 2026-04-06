#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def project_gaussians(means: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor, h: int, w: int):
    ones = torch.ones((means.shape[0], 1), device=means.device, dtype=means.dtype)
    world_h = torch.cat([means, ones], dim=-1)
    cam = (extrinsics @ world_h.t()).transpose(0, 1)
    z = cam[:, 2]
    z_safe = z.clamp_min(1e-6)
    u = intrinsics[0, 0] * (cam[:, 0] / z_safe) + intrinsics[0, 2]
    v = intrinsics[1, 1] * (cam[:, 1] / z_safe) + intrinsics[1, 2]
    ui = u.round().long().clamp(0, w - 1)
    vi = v.round().long().clamp(0, h - 1)
    in_img = (z > 0) & (u >= 0) & (u <= (w - 1)) & (v >= 0) & (v <= (h - 1))
    return u, v, ui, vi, z, in_img


class LocalViewFeatureEncoder(nn.Module):
    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, feat_dim, kernel_size=1),
        )

    def forward(self, rgb: torch.Tensor, col: torch.Tensor, dmg: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        depth_valid = (depth > 0).float().unsqueeze(1)
        depth_norm = depth.unsqueeze(1)
        denom = depth_norm.amax(dim=(2, 3), keepdim=True).clamp_min(1e-4)
        depth_norm = depth_norm / denom
        x = torch.cat([rgb, col.unsqueeze(1), dmg.unsqueeze(1), depth_norm, depth_valid], dim=1)
        return self.net(x)


class ProjectedLocalCrossAttention(nn.Module):
    def __init__(self, gauss_dim: int, token_dim: int, attn_dim: int, heads: int, hidden_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(gauss_dim, attn_dim)
        self.key_proj = nn.Linear(token_dim, attn_dim)
        self.value_proj = nn.Linear(token_dim, attn_dim)
        self.attn = nn.MultiheadAttention(attn_dim, heads, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(attn_dim),
            nn.Linear(attn_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, gaussian_feat: torch.Tensor, local_tokens: torch.Tensor, visible_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_gauss, n_view, _ = local_tokens.shape
        delta = torch.zeros((n_gauss, 2), device=gaussian_feat.device, dtype=gaussian_feat.dtype)
        weights = torch.zeros((n_gauss, n_view), device=gaussian_feat.device, dtype=gaussian_feat.dtype)
        valid = visible_mask.any(dim=1)
        if not valid.any():
            return delta, weights
        q = self.query_proj(gaussian_feat[valid]).unsqueeze(1)
        k = self.key_proj(local_tokens[valid])
        v = self.value_proj(local_tokens[valid])
        key_padding_mask = ~visible_mask[valid]
        attended, valid_weights = self.attn(q, k, v, key_padding_mask=key_padding_mask, need_weights=True)
        delta[valid] = self.head(attended.squeeze(1))
        weights[valid] = valid_weights.squeeze(1)
        return delta, weights


def dilate_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return mask
    pad = kernel_size // 2
    return F.max_pool2d(mask.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=pad).squeeze(1)


def build_candidate_mask(
    means: torch.Tensor,
    depth_batch: torch.Tensor,
    intr_batch: torch.Tensor,
    extr_batch: torch.Tensor,
    col_batch: torch.Tensor,
    dmg_batch: torch.Tensor,
    opacity_prob: torch.Tensor,
    column_prob: torch.Tensor,
    damage_prob: torch.Tensor,
    dilate_kernel: int,
    opacity_thresh: float,
    col_prior_thresh: float,
    dmg_prior_thresh: float,
    min_candidate_count: int,
    rel_depth_thresh: float,
    abs_depth_thresh: float,
) -> torch.Tensor:
    device = means.device
    col_dil = dilate_mask(col_batch, dilate_kernel)
    dmg_dil = dilate_mask(dmg_batch, dilate_kernel)
    visible_any = torch.zeros((means.shape[0],), dtype=torch.bool, device=device)
    hit_col = torch.zeros_like(visible_any)
    hit_dmg = torch.zeros_like(visible_any)
    near_col = torch.zeros_like(visible_any)
    near_dmg = torch.zeros_like(visible_any)

    for b in range(depth_batch.shape[0]):
        h, w = depth_batch[b].shape
        _, _, ui, vi, z, in_img = project_gaussians(means, extr_batch[b], intr_batch[b], h, w)
        gt_depth = depth_batch[b, vi, ui]
        depth_ok = gt_depth > 0
        depth_err = (z - gt_depth).abs()
        vis = in_img & depth_ok & (depth_err <= torch.maximum(rel_depth_thresh * gt_depth.abs(), torch.full_like(gt_depth, abs_depth_thresh)))
        visible_any |= vis
        hit_col |= vis & (col_batch[b, vi, ui] > 0.5)
        hit_dmg |= vis & (dmg_batch[b, vi, ui] > 0.5)
        near_col |= vis & (col_dil[b, vi, ui] > 0.5)
        near_dmg |= vis & (dmg_dil[b, vi, ui] > 0.5)

    candidate = visible_any & (
        hit_col | hit_dmg | near_col | near_dmg |
        (opacity_prob >= opacity_thresh) |
        (column_prob >= col_prior_thresh) |
        (damage_prob >= dmg_prior_thresh)
    )
    if candidate.sum().item() < min_candidate_count:
        priority = (
            hit_dmg.float() * 8.0 + hit_col.float() * 6.0 + near_dmg.float() * 4.0 + near_col.float() * 3.0 +
            visible_any.float() + opacity_prob * 0.5 + column_prob * 1.5 + damage_prob * 2.0
        )
        k = min(max(min_candidate_count, 1), means.shape[0])
        topk = torch.topk(priority, k=k).indices
        candidate = torch.zeros_like(candidate)
        candidate[topk] = True
    return candidate


def build_local_attention_tokens(
    means: torch.Tensor,
    intr_batch: torch.Tensor,
    extr_batch: torch.Tensor,
    depth_batch: torch.Tensor,
    col_batch: torch.Tensor,
    dmg_batch: torch.Tensor,
    feat_maps: torch.Tensor,
    rel_depth_thresh: float,
    abs_depth_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, feat_dim, h, w = feat_maps.shape
    n_gauss = means.shape[0]
    tokens = []
    visibility = []
    cam_centers = camera_centers_from_extrinsics(extr_batch)

    for b in range(bsz):
        u, v, ui, vi, z, in_img = project_gaussians(means, extr_batch[b], intr_batch[b], h, w)
        gt_depth = depth_batch[b, vi, ui]
        depth_ok = gt_depth > 0
        depth_err = (z - gt_depth).abs()
        vis = in_img & depth_ok & (depth_err <= torch.maximum(rel_depth_thresh * gt_depth.abs(), torch.full_like(gt_depth, abs_depth_thresh)))
        grid_x = (u / max(w - 1, 1)) * 2.0 - 1.0
        grid_y = (v / max(h - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, n_gauss, 1, 2)
        sampled = F.grid_sample(feat_maps[b:b + 1], grid, mode='bilinear', align_corners=True, padding_mode='zeros')
        sampled = sampled[0, :, :, 0].transpose(0, 1)
        col_val = col_batch[b, vi, ui].unsqueeze(-1)
        dmg_val = dmg_batch[b, vi, ui].unsqueeze(-1)
        depth_delta = ((z - gt_depth) / gt_depth.abs().clamp_min(1e-3)).unsqueeze(-1)
        cam_center = cam_centers[b].unsqueeze(0).expand(n_gauss, -1)
        token = torch.cat([sampled, col_val, dmg_val, depth_delta, cam_center], dim=-1)
        token = torch.where(vis.unsqueeze(-1), token, torch.zeros_like(token))
        tokens.append(token)
        visibility.append(vis)

    return torch.stack(tokens, dim=1), torch.stack(visibility, dim=1)


def camera_centers_from_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
    rot = extrinsics[:, :3, :3]
    trans = extrinsics[:, :3, 3]
    return -(rot.transpose(1, 2) @ trans.unsqueeze(-1)).squeeze(-1)


def compute_attention_delta_for_candidates(
    means: torch.Tensor,
    gaussian_feat: torch.Tensor,
    intr_batch: torch.Tensor,
    extr_batch: torch.Tensor,
    depth_batch: torch.Tensor,
    col_batch: torch.Tensor,
    dmg_batch: torch.Tensor,
    feat_maps: torch.Tensor,
    candidate_mask: torch.Tensor,
    attn: nn.Module,
    residual_scale: float,
    rel_depth_thresh: float,
    abs_depth_thresh: float,
    chunk_size: int | None = None,
) -> tuple[torch.Tensor, int]:
    delta = torch.zeros((means.shape[0], 2), device=means.device, dtype=gaussian_feat.dtype)
    cand_idx = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
    if cand_idx.numel() == 0:
        return delta, 0
    if chunk_size is None:
        chunk_size = cand_idx.numel()
    for start in range(0, cand_idx.numel(), chunk_size):
        idx = cand_idx[start:start + chunk_size]
        local_tokens, visible_tokens = build_local_attention_tokens(
            means[idx], intr_batch, extr_batch, depth_batch, col_batch, dmg_batch, feat_maps,
            rel_depth_thresh, abs_depth_thresh,
        )
        chunk_delta, _ = attn(gaussian_feat[idx], local_tokens, visible_tokens)
        delta[idx] = residual_scale * chunk_delta
    return delta, int(cand_idx.numel())
