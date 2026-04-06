#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from depth_anything_3.model.utils.gs_renderer import render_3dgs  # noqa: E402
from depth_anything_3.specs import Gaussians  # noqa: E402
from depth_anything_3.utils.gsply_helpers import export_ply  # noqa: E402
from output.column.column_attention_and_candidates import (
    LocalViewFeatureEncoder,
    ProjectedLocalCrossAttention,
    build_candidate_mask,
    camera_centers_from_extrinsics,
    compute_attention_delta_for_candidates,
    project_gaussians,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser('GS damage optimization with coarse-to-fine candidate attention')
    p.add_argument('--gs-ply', type=Path, required=True)
    p.add_argument('--label-root', type=Path, required=True)
    p.add_argument('--view-root', type=Path, required=True)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--stage1-steps', type=int, default=1000)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-train-side', type=int, default=1024)
    p.add_argument('--damage-label-id', type=int, default=2)
    p.add_argument('--spall-only-supervision', action='store_true')
    p.add_argument('--sem-column-only', action='store_true', default=True)
    p.add_argument('--photo-weight', type=float, default=0.10)
    p.add_argument('--depth-weight', type=float, default=0.05)
    p.add_argument('--column-weight', type=float, default=3.0)
    p.add_argument('--semantic-weight', type=float, default=1.2)
    p.add_argument('--mv-column-weight', type=float, default=2.0)
    p.add_argument('--mv-damage-weight', type=float, default=3.0)
    p.add_argument('--mv-var-weight', type=float, default=0.25)
    p.add_argument('--column-pos-weight', type=float, default=3.0)
    p.add_argument('--sem-pos-weight', type=float, default=14.0)
    p.add_argument('--outside-column-penalty', type=float, default=1.5)
    p.add_argument('--damage-outside-column-weight', type=float, default=2.0)
    p.add_argument('--sparsity-weight', type=float, default=1e-5)
    p.add_argument('--learn-labels-only', action='store_true', default=True)
    p.add_argument('--allow-gs-updates', action='store_false', dest='learn_labels_only')
    p.add_argument('--freeze-geo-steps', type=int, default=3000)
    p.add_argument('--lr-geo', type=float, default=5e-6)
    p.add_argument('--lr-color', type=float, default=1e-4)
    p.add_argument('--lr-column', type=float, default=2e-3)
    p.add_argument('--lr-damage', type=float, default=2e-3)
    p.add_argument('--lr-attn', type=float, default=8e-4)
    p.add_argument('--lr-encoder', type=float, default=8e-4)
    p.add_argument('--geom-reg-weight', type=float, default=5e-3)
    p.add_argument('--color-reg-weight', type=float, default=2e-3)
    p.add_argument('--opacity-reg-weight', type=float, default=1e-3)
    p.add_argument('--scale-reg-weight', type=float, default=8e-3)
    p.add_argument('--scale-min-mult', type=float, default=0.80)
    p.add_argument('--scale-max-mult', type=float, default=1.25)
    p.add_argument('--opacity-max', type=float, default=0.95)
    p.add_argument('--column-init-logit', type=float, default=-2.0)
    p.add_argument('--damage-init-logit', type=float, default=-3.5)
    p.add_argument('--column-thresh-train', type=float, default=0.55)
    p.add_argument('--column-thresh-export', type=float, default=0.90)
    p.add_argument('--soft-column-gate', action='store_true', default=True)
    p.add_argument('--hard-column-gate', action='store_false', dest='soft_column_gate')
    p.add_argument('--attn-dim', type=int, default=64)
    p.add_argument('--attn-heads', type=int, default=4)
    p.add_argument('--attn-hidden-dim', type=int, default=96)
    p.add_argument('--attn-residual-scale', type=float, default=1.0)
    p.add_argument('--local-feat-dim', type=int, default=32)
    p.add_argument('--proj-depth-rel-thresh', type=float, default=0.08)
    p.add_argument('--proj-depth-abs-thresh', type=float, default=0.20)
    p.add_argument('--far-view-blend', type=float, default=0.70)
    p.add_argument('--candidate-dilate', type=int, default=17)
    p.add_argument('--opacity-candidate-thresh', type=float, default=0.08)
    p.add_argument('--column-prior-candidate-thresh', type=float, default=0.10)
    p.add_argument('--damage-prior-candidate-thresh', type=float, default=0.05)
    p.add_argument('--min-candidate-count', type=int, default=50000)
    p.add_argument('--train-attn-chunk', type=int, default=100000)
    p.add_argument('--final-attn-chunk', type=int, default=200000)
    p.add_argument('--print-every', type=int, default=50)
    p.add_argument('--debug-every', type=int, default=200)
    p.add_argument('--debug-dir', type=Path, default=Path('output/column/debug_pred_full'))
    p.add_argument('--save-ply', type=Path, default=Path('output/column/gs_ply_full/0000_damage_opt_mfull.ply'))
    p.add_argument('--save-highlight-ply', type=Path, default=Path('output/column/gs_ply_full/0000_damage_highlight_full.ply'))
    p.add_argument('--save-column-highlight-ply', type=Path, default=Path('output/column/gs_ply_full/0000_column_highlight_full.ply'))
    p.add_argument('--save-damage-only-ply', type=Path, default=Path('output/column/gs_ply_full/0000_damage_only_full.ply'))
    p.add_argument('--save-damage', type=Path, default=Path('output/column/gs_ply_full/0000_damage_prob_full.npy'))
    p.add_argument('--save-column', type=Path, default=Path('output/column/gs_ply_full/0000_column_prob_full.npy'))
    p.add_argument('--damage-thresh', type=float, default=0.15)
    p.add_argument('--min-damage-count', type=int, default=3000)
    p.add_argument('--highlight-alpha', type=float, default=0.85)
    p.add_argument('--highlight-min-alpha', type=float, default=0.30)
    p.add_argument('--highlight-binary', action='store_true')
    p.add_argument('--highlight-transparency', type=float, default=0.70)
    p.add_argument('--highlight-opacity-boost', type=float, default=0.25)
    p.add_argument('--damage-only-opacity', type=float, default=0.22)
    return p.parse_args()


def infer_damage_label_id(label_root: Path, fallback: int) -> int:
    candidate_names = ('damage', 'spalling', 'spaling')
    for dataset_dir in sorted(label_root.glob('*_dataset')):
        mapping_path = dataset_dir / 'label_name_to_value.txt'
        if not mapping_path.exists():
            continue
        parsed = {}
        for raw_line in mapping_path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                parsed[parts[0].strip().lower()] = int(parts[-1])
            except ValueError:
                continue
        for name in candidate_names:
            if name in parsed:
                print(f'[info] inferred damage label id={parsed[name]} from {mapping_path}')
                return parsed[name]
    print(f'[info] using damage label id={fallback}')
    return fallback


def load_custom_supervision(label_root: Path, view_root: Path, damage_label_id: int, spall_only_supervision: bool, max_train_side: int):
    dataset_dirs = sorted(label_root.glob('*_dataset'))
    if not dataset_dirs:
        raise RuntimeError(f'No label datasets found under: {label_root}')
    view_names, images, depths, intrinsics, extrinsics, spall_masks, col_masks = [], [], [], [], [], [], []
    for dataset_dir in dataset_dirs:
        view_name = dataset_dir.name.removesuffix('_dataset')
        img_path = dataset_dir / 'img.png'
        label_path = dataset_dir / 'label.png'
        cam_path = view_root / f'{view_name}_camera.json'
        depth_path = view_root / f'{view_name}_depth.npy'
        missing = [str(p) for p in (img_path, label_path, cam_path, depth_path) if not p.exists()]
        if missing:
            print(f'[warn] skipping {view_name}, missing: {", ".join(missing)}')
            continue
        img = np.asarray(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        label = np.asarray(Image.open(label_path).convert('L'), dtype=np.uint8)
        depth = np.load(depth_path).astype(np.float32)
        camera = json.loads(cam_path.read_text(encoding='utf-8'))
        intr = np.asarray(camera['intrinsics'], dtype=np.float32)
        extr = np.asarray(camera['w2c_extrinsics'], dtype=np.float32)
        h, w = img.shape[:2]
        if label.shape != (h, w) or depth.shape != (h, w):
            raise ValueError(f'shape mismatch in {dataset_dir}')
        spall = (label == damage_label_id).astype(np.float32)
        col = spall.copy() if spall_only_supervision else (label > 0).astype(np.float32)
        view_names.append(view_name)
        images.append(img)
        depths.append(depth)
        intrinsics.append(intr)
        extrinsics.append(extr)
        spall_masks.append(spall)
        col_masks.append(col)
    if not view_names:
        raise RuntimeError('No valid labeled GS views were found.')
    target_h = min(x.shape[0] for x in images)
    target_w = min(x.shape[1] for x in images)
    max_side = max(int(max_train_side), 1)
    if max(target_h, target_w) > max_side:
        scale = max_side / float(max(target_h, target_w))
        target_h = max(1, int(round(target_h * scale)))
        target_w = max(1, int(round(target_w * scale)))
    out_images, out_depths, out_intrinsics, out_spall, out_col = [], [], [], [], []
    for img, depth, intr, spall, col in zip(images, depths, intrinsics, spall_masks, col_masks):
        src_h, src_w = img.shape[:2]
        sx = target_w / src_w
        sy = target_h / src_h
        if (src_h, src_w) != (target_h, target_w):
            img = np.asarray(Image.fromarray(img).resize((target_w, target_h), Image.BILINEAR), dtype=np.uint8)
            depth = np.asarray(Image.fromarray(depth.astype(np.float32), mode='F').resize((target_w, target_h), Image.BILINEAR), dtype=np.float32)
            spall = np.asarray(Image.fromarray((spall * 255).astype(np.uint8), mode='L').resize((target_w, target_h), Image.NEAREST), dtype=np.uint8)
            col = np.asarray(Image.fromarray((col * 255).astype(np.uint8), mode='L').resize((target_w, target_h), Image.NEAREST), dtype=np.uint8)
            spall = (spall > 127).astype(np.float32)
            col = (col > 127).astype(np.float32)
        intr = intr.copy()
        intr[0, 0] *= sx
        intr[0, 2] *= sx
        intr[1, 1] *= sy
        intr[1, 2] *= sy
        out_images.append(img)
        out_depths.append(depth)
        out_intrinsics.append(intr)
        out_spall.append(spall)
        out_col.append(col)
    return {
        'view_names': view_names,
        'images': np.stack(out_images, axis=0),
        'depths': np.stack(out_depths, axis=0),
        'intrinsics': np.stack(out_intrinsics, axis=0),
        'extrinsics': np.stack(extrinsics, axis=0),
        'spall_masks': np.stack(out_spall, axis=0),
        'col_masks': np.stack(out_col, axis=0),
    }


def load_gs_ply(path: Path, device: torch.device):
    ply = PlyData.read(str(path))
    v = ply['vertex']
    return {
        'means': torch.tensor(np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32), device=device),
        'sh_dc': torch.tensor(np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1).astype(np.float32), device=device),
        'log_scales': torch.tensor(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1).astype(np.float32), device=device),
        'quats': torch.tensor(np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1).astype(np.float32), device=device),
        'opacity_logit': torch.tensor(np.asarray(v['opacity'], dtype=np.float32), device=device),
    }


def build_gaussians(means, log_scales, quats, sh_dc, opacity_logit):
    return Gaussians(
        means=means[None],
        scales=torch.exp(log_scales).clamp(1e-5, 30.0)[None],
        rotations=F.normalize(quats, dim=-1)[None],
        harmonics=sh_dc[..., None][None],
        opacities=torch.sigmoid(opacity_logit).clamp(1e-4, 1 - 1e-4)[None],
    )


def render_view(gaussians, extrinsics, intrinsics, h, w, use_sh):
    intr_normed = intrinsics.clone()
    intr_normed[:, 0, :] /= w
    intr_normed[:, 1, :] /= h
    rgb, depth = render_3dgs(extrinsics=extrinsics, intrinsics=intr_normed, image_shape=(h, w), gaussian=gaussians, use_sh=use_sh, num_view=1, color_mode='RGB+D')
    return rgb[0], depth[0]


def _logit(x):
    x = x.clamp(1e-4, 1 - 1e-4)
    return torch.log(x / (1 - x))


def weighted_bce_prob(prob, target, pos_weight):
    prob = prob.clamp(1e-4, 1 - 1e-4)
    return -(pos_weight * target * torch.log(prob) + (1.0 - target) * torch.log(1.0 - prob)).mean()


def save_gray(path: Path, x: torch.Tensor) -> None:
    arr = (x.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def select_far_view_batch(camera_centers: torch.Tensor, batch_size: int, blend: float) -> list[int]:
    n_view = camera_centers.shape[0]
    if n_view <= batch_size:
        return list(range(n_view))
    dist = torch.cdist(camera_centers.detach(), camera_centers.detach())
    selected = [random.randrange(n_view)]
    remaining = set(range(n_view)) - set(selected)
    while len(selected) < batch_size and remaining:
        best_idx = None
        best_score = -1.0
        for idx in remaining:
            min_dist = dist[idx, selected].min().item()
            mean_dist = dist[idx, selected].mean().item()
            score = blend * min_dist + (1.0 - blend) * mean_dist + 1e-3 * random.random()
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(int(best_idx))
        remaining.remove(int(best_idx))
    return selected


def build_gaussian_features(means, sh_dc, opacity_logit):
    xyz = means - means.mean(dim=0, keepdim=True)
    xyz = xyz / xyz.std(dim=0, keepdim=True).clamp_min(1e-4)
    rgb = torch.sigmoid(sh_dc)
    opacity = torch.sigmoid(opacity_logit).unsqueeze(-1)
    radius = torch.norm(xyz, dim=-1, keepdim=True)
    return torch.cat([xyz, rgb, opacity, radius], dim=-1)


def projection_consistency_losses(means, column_prob, damage_prob_gated, depth_gt_batch, col_gt_batch, dmg_gt_batch, intr_batch, extr_batch, h, w, rel_thresh, abs_thresh, column_pos_weight, sem_pos_weight):
    col_obs = []
    dmg_obs = []
    vis_obs = []
    for b in range(depth_gt_batch.shape[0]):
        _, _, ui, vi, z, in_img = project_gaussians(means, extr_batch[b], intr_batch[b], h, w)
        gt_depth = depth_gt_batch[b, vi, ui]
        depth_ok = gt_depth > 0
        depth_err = (z - gt_depth).abs()
        vis = in_img & depth_ok & (depth_err <= torch.maximum(rel_thresh * gt_depth.abs(), torch.full_like(gt_depth, abs_thresh)))
        col_obs.append(col_gt_batch[b, vi, ui])
        dmg_obs.append(dmg_gt_batch[b, vi, ui])
        vis_obs.append(vis.float())
    col_obs_t = torch.stack(col_obs, dim=0)
    dmg_obs_t = torch.stack(dmg_obs, dim=0)
    vis_t = torch.stack(vis_obs, dim=0)
    vis_count = vis_t.sum(dim=0)
    visible_mask = vis_count > 0
    if not visible_mask.any():
        zero = means.new_tensor(0.0)
        return zero, zero, zero, 0
    col_target = (col_obs_t * vis_t).sum(dim=0) / vis_count.clamp_min(1.0)
    dmg_target = (dmg_obs_t * vis_t).sum(dim=0) / vis_count.clamp_min(1.0)
    mv_col = weighted_bce_prob(column_prob[visible_mask], col_target[visible_mask], column_pos_weight)
    mv_dmg = weighted_bce_prob(damage_prob_gated[visible_mask], dmg_target[visible_mask], sem_pos_weight)
    col_dev = (((col_obs_t - col_target.unsqueeze(0)) ** 2) * vis_t).sum() / vis_t.sum().clamp_min(1.0)
    dmg_dev = (((dmg_obs_t - dmg_target.unsqueeze(0)) ** 2) * vis_t).sum() / vis_t.sum().clamp_min(1.0)
    return mv_col, mv_dmg, 0.5 * (col_dev + dmg_dev), int(visible_mask.sum().item())


def make_optimizer(means, log_scales, quats, sh_dc, opacity_logit, column_logit, damage_logit, attn, encoder, args):
    groups = [
        {'params': [column_logit], 'lr': args.lr_column},
        {'params': [damage_logit], 'lr': args.lr_damage},
        {'params': list(attn.parameters()), 'lr': args.lr_attn},
        {'params': list(encoder.parameters()), 'lr': args.lr_encoder},
    ]
    if not args.learn_labels_only:
        groups.extend([
            {'params': [means, log_scales, quats], 'lr': args.lr_geo},
            {'params': [sh_dc, opacity_logit], 'lr': args.lr_color},
        ])
    return torch.optim.Adam(groups)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[info] device={device}')
    print(f'[info] learn_labels_only={args.learn_labels_only}')
    args.damage_label_id = infer_damage_label_id(args.label_root, args.damage_label_id)
    sup = load_custom_supervision(args.label_root, args.view_root, args.damage_label_id, args.spall_only_supervision, args.max_train_side)
    images = sup['images']
    depths = sup['depths']
    intrinsics = sup['intrinsics']
    extrinsics = sup['extrinsics']
    spall_masks = sup['spall_masks']
    col_masks = sup['col_masks']
    view_names = sup['view_names']
    n_view, h, w, _ = images.shape
    rgb_gt = torch.from_numpy(images).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    depth_gt = torch.from_numpy(depths).to(device=device, dtype=torch.float32)
    intr_t = torch.from_numpy(intrinsics).to(device=device, dtype=torch.float32)
    extr_t = torch.from_numpy(extrinsics).to(device=device, dtype=torch.float32)
    sem_spall_gt = torch.from_numpy(spall_masks).to(device=device, dtype=torch.float32)
    sem_col_gt = torch.from_numpy(col_masks).to(device=device, dtype=torch.float32)
    camera_centers = camera_centers_from_extrinsics(extr_t)
    gs_init = load_gs_ply(args.gs_ply, device)
    means = torch.nn.Parameter(gs_init['means'].clone())
    sh_dc = torch.nn.Parameter(gs_init['sh_dc'].clone())
    log_scales = torch.nn.Parameter(gs_init['log_scales'].clone())
    quats = torch.nn.Parameter(gs_init['quats'].clone())
    opacity_logit = torch.nn.Parameter(gs_init['opacity_logit'].clone())
    column_logit = torch.nn.Parameter(torch.full((means.shape[0],), args.column_init_logit, device=device))
    damage_logit = torch.nn.Parameter(torch.full((means.shape[0],), args.damage_init_logit, device=device))
    means_ref = gs_init['means'].clone().detach()
    log_scales_ref = gs_init['log_scales'].clone().detach()
    sh_dc_ref = gs_init['sh_dc'].clone().detach()
    opacity_ref = gs_init['opacity_logit'].clone().detach()
    quats_ref = gs_init['quats'].clone().detach()
    encoder = LocalViewFeatureEncoder(args.local_feat_dim).to(device)
    token_dim = args.local_feat_dim + 1 + 1 + 1 + 3
    attn = ProjectedLocalCrossAttention(gauss_dim=8, token_dim=token_dim, attn_dim=args.attn_dim, heads=args.attn_heads, hidden_dim=args.attn_hidden_dim).to(device)
    optimizer = make_optimizer(means, log_scales, quats, sh_dc, opacity_logit, column_logit, damage_logit, attn, encoder, args)
    args.debug_dir.mkdir(parents=True, exist_ok=True)
    args.save_ply.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        in_stage1 = step <= args.stage1_steps
        batch_idx = select_far_view_batch(camera_centers, min(args.batch_size, n_view), args.far_view_blend)
        batch_idx_t = torch.tensor(batch_idx, device=device, dtype=torch.long)
        rgb_batch = rgb_gt[batch_idx_t]
        depth_batch = depth_gt[batch_idx_t]
        col_batch = sem_col_gt[batch_idx_t]
        dmg_batch = sem_spall_gt[batch_idx_t]
        intr_batch = intr_t[batch_idx_t]
        extr_batch = extr_t[batch_idx_t]
        feat_maps = encoder(rgb_batch, col_batch, dmg_batch, depth_batch)
        cur_means = means_ref if args.learn_labels_only else means
        cur_sh = sh_dc_ref if args.learn_labels_only else sh_dc
        cur_opacity = opacity_ref if args.learn_labels_only else opacity_logit
        gaussian_feat = build_gaussian_features(cur_means, cur_sh, cur_opacity)
        candidate_mask = build_candidate_mask(cur_means, depth_batch, intr_batch, extr_batch, col_batch, dmg_batch, torch.sigmoid(cur_opacity), torch.sigmoid(column_logit.detach()), torch.sigmoid(damage_logit.detach()), args.candidate_dilate, args.opacity_candidate_thresh, args.column_prior_candidate_thresh, args.damage_prior_candidate_thresh, args.min_candidate_count, args.proj_depth_rel_thresh, args.proj_depth_abs_thresh)
        attn_delta, n_candidate = compute_attention_delta_for_candidates(cur_means, gaussian_feat, intr_batch, extr_batch, depth_batch, col_batch, dmg_batch, feat_maps, candidate_mask, attn, args.attn_residual_scale, args.proj_depth_rel_thresh, args.proj_depth_abs_thresh, chunk_size=args.train_attn_chunk)
        column_prob_base = torch.sigmoid(column_logit + attn_delta[:, 0])
        damage_prob_base = torch.sigmoid(damage_logit + attn_delta[:, 1])
        col_gate_g = column_prob_base if args.soft_column_gate else (column_prob_base.detach() >= args.column_thresh_train).float()
        damage_prob_gated = damage_prob_base * col_gate_g
        g = build_gaussians(means, log_scales, quats, sh_dc, opacity_logit)
        col_harm = column_prob_base[:, None, None].repeat(1, 3, 1)
        dmg_harm = damage_prob_gated[:, None, None].repeat(1, 3, 1)
        col_g = Gaussians(means=g.means, scales=g.scales, rotations=g.rotations, opacities=g.opacities, harmonics=col_harm[None])
        dmg_g = Gaussians(means=g.means, scales=g.scales, rotations=g.rotations, opacities=g.opacities, harmonics=dmg_harm[None])
        loss_photo = torch.tensor(0.0, device=device)
        loss_depth = torch.tensor(0.0, device=device)
        loss_col = torch.tensor(0.0, device=device)
        loss_col_out = torch.tensor(0.0, device=device)
        loss_sem = torch.tensor(0.0, device=device)
        loss_out_col = torch.tensor(0.0, device=device)
        debug_cache = None
        for vi in batch_idx:
            rgb_pred, depth_pred = render_view(g, extr_t[vi:vi + 1], intr_t[vi:vi + 1], h, w, use_sh=True)
            col_rgb, _ = render_view(col_g, extr_t[vi:vi + 1], intr_t[vi:vi + 1], h, w, use_sh=False)
            sem_rgb, _ = render_view(dmg_g, extr_t[vi:vi + 1], intr_t[vi:vi + 1], h, w, use_sh=False)
            col_pred = col_rgb.mean(dim=0).clamp(1e-4, 1 - 1e-4)
            sem_pred = sem_rgb.mean(dim=0).clamp(1e-4, 1 - 1e-4)
            col_gt = sem_col_gt[vi]
            dmg_gt = sem_spall_gt[vi]
            depth_valid = depth_gt[vi] > 0
            loss_photo = loss_photo + torch.mean(torch.abs(rgb_pred - rgb_gt[vi]))
            loss_depth = loss_depth + (F.smooth_l1_loss(depth_pred[depth_valid], depth_gt[vi][depth_valid]) if depth_valid.any() else torch.tensor(0.0, device=device))
            valid_col = col_gt > 0.5
            out_col = col_gt < 0.5
            if valid_col.any():
                loss_col = loss_col + weighted_bce_prob(col_pred[valid_col], col_gt[valid_col], args.column_pos_weight)
            if out_col.any():
                loss_col_out = loss_col_out + col_pred[out_col].mean()
            if not in_stage1:
                valid_sem = (col_gt > 0.5) if args.sem_column_only else torch.ones_like(col_gt, dtype=torch.bool)
                if valid_sem.any():
                    loss_sem = loss_sem + weighted_bce_prob(sem_pred[valid_sem], dmg_gt[valid_sem], args.sem_pos_weight)
                if out_col.any():
                    loss_out_col = loss_out_col + sem_pred[out_col].mean()
            if debug_cache is None:
                debug_cache = (vi, col_pred, sem_pred, col_gt, dmg_gt)
        batch_div = float(len(batch_idx))
        loss_photo /= batch_div
        loss_depth /= batch_div
        loss_col /= batch_div
        loss_col_out /= batch_div
        loss_sem /= batch_div
        loss_out_col /= batch_div
        mv_col, mv_dmg, mv_var, mv_visible = projection_consistency_losses(cur_means, column_prob_base, damage_prob_gated, depth_batch, col_batch, dmg_batch, intr_batch, extr_batch, h, w, args.proj_depth_rel_thresh, args.proj_depth_abs_thresh, args.column_pos_weight, args.sem_pos_weight)
        loss_sparse = damage_prob_base.mean() if not in_stage1 else torch.tensor(0.0, device=device)
        loss_damage_out = (damage_prob_base * (1.0 - column_prob_base)).mean() if not in_stage1 else torch.tensor(0.0, device=device)
        loss_geom = F.smooth_l1_loss(means, means_ref) + F.smooth_l1_loss(log_scales, log_scales_ref)
        loss_color_reg = F.smooth_l1_loss(sh_dc, sh_dc_ref)
        loss_opacity_reg = F.smooth_l1_loss(opacity_logit, opacity_ref)
        loss_scale_reg = F.smooth_l1_loss(log_scales, log_scales_ref)
        loss = args.photo_weight * loss_photo + args.depth_weight * loss_depth + args.column_weight * loss_col + args.semantic_weight * loss_sem + args.mv_column_weight * mv_col + (0.0 if in_stage1 else args.mv_damage_weight * mv_dmg) + args.mv_var_weight * mv_var + args.outside_column_penalty * loss_col_out + args.outside_column_penalty * loss_out_col + args.damage_outside_column_weight * loss_damage_out + args.sparsity_weight * loss_sparse + args.geom_reg_weight * loss_geom + args.color_reg_weight * loss_color_reg + args.opacity_reg_weight * loss_opacity_reg + args.scale_reg_weight * loss_scale_reg
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if in_stage1:
            damage_logit.grad = None
        if args.learn_labels_only or step <= args.freeze_geo_steps:
            means.grad = None
            log_scales.grad = None
            quats.grad = None
            sh_dc.grad = None
            opacity_logit.grad = None
        optimizer.step()
        with torch.no_grad():
            if not args.learn_labels_only:
                log_scales.data.clamp_(min=log_scales_ref + math.log(max(args.scale_min_mult, 1e-3)), max=log_scales_ref + math.log(max(args.scale_max_mult, 1.0)))
                opacity_logit.data.clamp_(min=_logit(torch.tensor(1e-4, device=device)), max=_logit(torch.tensor(args.opacity_max, device=device)))
            else:
                means.data.copy_(means_ref)
                log_scales.data.copy_(log_scales_ref)
                quats.data.copy_(quats_ref)
                sh_dc.data.copy_(sh_dc_ref)
                opacity_logit.data.copy_(opacity_ref)
        if (step % args.debug_every == 0 or step == 1) and debug_cache is not None:
            vi, col_pred, sem_pred, col_gt, dmg_gt = debug_cache
            stem = f'step{step:05d}_{view_names[vi]}'
            save_gray(args.debug_dir / f'{stem}_colpred.png', col_pred)
            save_gray(args.debug_dir / f'{stem}_gtcol.png', col_gt)
            save_gray(args.debug_dir / f'{stem}_sempred.png', sem_pred)
            save_gray(args.debug_dir / f'{stem}_gtdmg.png', dmg_gt)
        if step % args.print_every == 0 or step == 1:
            batch_names = ','.join(view_names[i] for i in batch_idx)
            print(f"[{step:05d}/{args.steps}][{'S1' if in_stage1 else 'S2'}] photo={loss_photo.item():.4f} depth={loss_depth.item():.4f} col={loss_col.item():.4f} sem={loss_sem.item():.4f} mv_col={mv_col.item():.4f} mv_dmg={mv_dmg.item():.4f} mv_var={mv_var.item():.4f} col_out={loss_col_out.item():.4f} out={loss_out_col.item():.4f} d_out={loss_damage_out.item():.4f} sparse={loss_sparse.item():.4f} visible_g={mv_visible} cand_g={n_candidate} total={loss.item():.4f} views={batch_names}")
    with torch.no_grad():
        final_means = means_ref if args.learn_labels_only else means
        final_scales = torch.exp(log_scales_ref if args.learn_labels_only else log_scales).clamp(1e-5, 30.0)
        final_quats = F.normalize(quats_ref if args.learn_labels_only else quats, dim=-1)
        final_harm = (sh_dc_ref if args.learn_labels_only else sh_dc)[..., None]
        final_opacity = opacity_ref if args.learn_labels_only else opacity_logit
        export_ply(means=final_means, scales=final_scales, rotations=final_quats, harmonics=final_harm, opacities=final_opacity, path=args.save_ply, shift_and_scale=False, save_sh_dc_only=True, match_3dgs_mcmc_dev=False)
        full_feat_maps = encoder(rgb_gt, sem_col_gt, sem_spall_gt, depth_gt)
        gaussian_feat = build_gaussian_features(final_means, sh_dc_ref if args.learn_labels_only else sh_dc, final_opacity)
        candidate_mask = build_candidate_mask(final_means, depth_gt, intr_t, extr_t, sem_col_gt, sem_spall_gt, torch.sigmoid(final_opacity), torch.sigmoid(column_logit), torch.sigmoid(damage_logit), args.candidate_dilate, args.opacity_candidate_thresh, args.column_prior_candidate_thresh, args.damage_prior_candidate_thresh, args.min_candidate_count, args.proj_depth_rel_thresh, args.proj_depth_abs_thresh)
        attn_delta, _ = compute_attention_delta_for_candidates(final_means, gaussian_feat, intr_t, extr_t, depth_gt, sem_col_gt, sem_spall_gt, full_feat_maps, candidate_mask, attn, args.attn_residual_scale, args.proj_depth_rel_thresh, args.proj_depth_abs_thresh, chunk_size=args.final_attn_chunk)
        damage_prob_t = torch.sigmoid(damage_logit + attn_delta[:, 1])
        column_prob_t = torch.sigmoid(column_logit + attn_delta[:, 0])
        column_mask_t = column_prob_t >= args.column_thresh_export
        gated_damage_t = damage_prob_t * column_mask_t.float()
        damage_score = ((gated_damage_t - args.damage_thresh) / (1.0 - args.damage_thresh + 1e-6)).clamp(0.0, 1.0)
        damage_mask_bool = damage_score > 0
        if not damage_mask_bool.any():
            k = min(max(args.min_damage_count, 1), damage_prob_t.numel())
            topk_idx = torch.topk(gated_damage_t, k=k).indices
            damage_mask_bool = torch.zeros_like(damage_prob_t, dtype=torch.bool)
            damage_mask_bool[topk_idx] = True
            damage_score = damage_mask_bool.float()
        base_rgb = torch.sigmoid(sh_dc_ref if args.learn_labels_only else sh_dc)
        base_opacity_prob = torch.sigmoid(final_opacity)
        highlight_rgb = torch.tensor([1.0, 0.05, 0.05], device=device).unsqueeze(0)
        alpha_score = damage_mask_bool.float() if args.highlight_binary else damage_score
        alpha_score = torch.where(damage_mask_bool, torch.maximum(alpha_score, torch.full_like(alpha_score, args.highlight_min_alpha)), torch.zeros_like(alpha_score)).clamp(0.0, 1.0)
        alpha = (args.highlight_alpha * alpha_score).unsqueeze(-1).clamp(0.0, 1.0)
        blended_rgb = (1.0 - alpha) * base_rgb + alpha * highlight_rgb
        highlight_harm = _logit(blended_rgb)[..., None]
        overlay_opacity_prob = (base_opacity_prob * (1.0 - args.highlight_transparency * alpha.squeeze(-1)) + args.highlight_opacity_boost * alpha.squeeze(-1)).clamp(1e-4, 1.0 - 1e-4)
        overlay_opacity_logit = _logit(overlay_opacity_prob)
        export_ply(means=final_means, scales=final_scales, rotations=final_quats, harmonics=highlight_harm, opacities=overlay_opacity_logit, path=args.save_highlight_ply, shift_and_scale=False, save_sh_dc_only=True, match_3dgs_mcmc_dev=False)
        column_score = ((column_prob_t - args.column_thresh_export) / (1.0 - args.column_thresh_export + 1e-6)).clamp(0.0, 1.0)
        column_mask_bool = column_prob_t >= args.column_thresh_export
        column_alpha_score = torch.where(column_mask_bool, torch.maximum(column_score, torch.full_like(column_score, args.highlight_min_alpha)), torch.zeros_like(column_score)).clamp(0.0, 1.0)
        column_alpha = (args.highlight_alpha * column_alpha_score).unsqueeze(-1).clamp(0.0, 1.0)
        column_rgb = torch.tensor([0.05, 1.0, 0.10], device=device).unsqueeze(0)
        column_blended_rgb = (1.0 - column_alpha) * base_rgb + column_alpha * column_rgb
        column_highlight_harm = _logit(column_blended_rgb)[..., None]
        column_overlay_opacity_prob = (base_opacity_prob * (1.0 - args.highlight_transparency * column_alpha.squeeze(-1)) + args.highlight_opacity_boost * column_alpha.squeeze(-1)).clamp(1e-4, 1.0 - 1e-4)
        column_overlay_opacity_logit = _logit(column_overlay_opacity_prob)
        export_ply(means=final_means, scales=final_scales, rotations=final_quats, harmonics=column_highlight_harm, opacities=column_overlay_opacity_logit, path=args.save_column_highlight_ply, shift_and_scale=False, save_sh_dc_only=True, match_3dgs_mcmc_dev=False)
        sel_means = final_means[damage_mask_bool]
        sel_scales = final_scales[damage_mask_bool]
        sel_quats = final_quats[damage_mask_bool]
        sel_opacity = _logit(torch.full((sel_means.shape[0],), args.damage_only_opacity, device=device))
        sel_color = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 3).repeat(sel_means.shape[0], 1)
        sel_harm = _logit(sel_color)[..., None]
        export_ply(means=sel_means, scales=sel_scales, rotations=sel_quats, harmonics=sel_harm, opacities=sel_opacity, path=args.save_damage_only_ply, shift_and_scale=False, save_sh_dc_only=True, match_3dgs_mcmc_dev=False)
        np.save(args.save_damage, damage_prob_t.detach().cpu().numpy())
        np.save(args.save_column, column_prob_t.detach().cpu().numpy())
        print(f'[done] saved optimized ply: {args.save_ply}')
        print(f'[done] saved damage highlight ply: {args.save_highlight_ply}')
        print(f'[done] saved column highlight ply: {args.save_column_highlight_ply}')
        print(f'[done] saved damage-only ply: {args.save_damage_only_ply}')
        print(f'[done] saved damage prob: {args.save_damage}')
        print(f'[done] saved column prob: {args.save_column}')


if __name__ == '__main__':
    main()
