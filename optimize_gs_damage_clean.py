#!/usr/bin/env python3
"""
Clean GS damage optimization script.

Designed for the case where labels are drawn on the SAME exported GS-model views.
So this version uses only:
- gs ply
- labeled view folders: <name>_dataset/{img.png,label.png}
- exported camera/depth files: <name>_camera.json, <name>_depth.npy

Removed on purpose:
- DA3 NPZ view matching
- point-cloud cross-attention prior
- densification
- unused arguments/functions
- extra image-space damage * column multiplication
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from depth_anything_3.model.utils.gs_renderer import render_3dgs  # noqa: E402
from depth_anything_3.specs import Gaussians  # noqa: E402
from depth_anything_3.utils.gsply_helpers import export_ply  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Clean GS damage optimization")

    # Required inputs
    p.add_argument("--gs-ply", type=Path, required=True)
    p.add_argument("--label-root", type=Path, required=True)
    p.add_argument("--view-root", type=Path, required=True)

    # Training setup
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--stage1-steps", type=int, default=1200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--labeled-sample-prob", type=float, default=1.0)
    p.add_argument("--damage-label-id", type=int, default=2)
    p.add_argument("--spall-only-supervision", action="store_true")
    p.add_argument("--photo-depth-column-only", action="store_true")
    p.add_argument("--sem-column-only", action="store_true", default=True)

    # Loss weights
    p.add_argument("--photo-weight", type=float, default=1.0)
    p.add_argument("--depth-weight", type=float, default=0.10)
    p.add_argument("--column-weight", type=float, default=4.0)
    p.add_argument("--semantic-weight", type=float, default=0.30)
    p.add_argument("--column-pos-weight", type=float, default=3.0)
    p.add_argument("--sem-pos-weight", type=float, default=12.0)
    p.add_argument("--outside-column-penalty", type=float, default=2.0)
    p.add_argument("--damage-outside-column-weight", type=float, default=2.0)
    p.add_argument("--sparsity-weight", type=float, default=1e-5)
    p.add_argument("--stage2-column-loss-scale", type=float, default=0.5)

    # Regularization / optimization
    p.add_argument("--freeze-geo-steps", type=int, default=1500)
    p.add_argument("--lr-geo", type=float, default=2e-5)
    p.add_argument("--lr-color", type=float, default=5e-4)
    p.add_argument("--lr-column", type=float, default=2e-3)
    p.add_argument("--lr-damage", type=float, default=2e-3)
    p.add_argument("--geom-reg-weight", type=float, default=2e-3)
    p.add_argument("--color-reg-weight", type=float, default=8e-4)
    p.add_argument("--opacity-reg-weight", type=float, default=1e-3)
    p.add_argument("--scale-reg-weight", type=float, default=5e-3)
    p.add_argument("--scale-min-mult", type=float, default=0.60)
    p.add_argument("--scale-max-mult", type=float, default=1.60)
    p.add_argument("--opacity-max", type=float, default=0.95)

    # Gating / export
    p.add_argument("--column-init-logit", type=float, default=-2.0)
    p.add_argument("--damage-init-logit", type=float, default=-3.0)
    p.add_argument("--column-thresh-train", type=float, default=0.55)
    p.add_argument("--column-thresh-export", type=float, default=0.90)
    p.add_argument("--soft-column-gate", action="store_true", default=True)
    p.add_argument("--hard-column-gate", action="store_false", dest="soft_column_gate")
    p.add_argument("--freeze-column-stage2", action="store_true")

    # Logging / outputs
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--debug-every", type=int, default=200)
    p.add_argument("--debug-dir", type=Path, default=Path("output/column/debug_pred"))
    p.add_argument("--save-ply", type=Path, default=Path("output/column/gs_ply_clean/0000_damage_opt_clean.ply"))
    p.add_argument("--save-highlight-ply", type=Path, default=Path("output/column/gs_ply_clean/0000_damage_highlight_clean.ply"))
    p.add_argument("--save-column-highlight-ply", type=Path, default=Path("output/column/gs_ply_clean/0000_column_highlight_clean.ply"))
    p.add_argument("--save-damage-only-ply", type=Path, default=Path("output/column/gs_ply_clean/0000_damage_only_clean.ply"))
    p.add_argument("--save-damage", type=Path, default=Path("output/column/gs_ply_clean/0000_damage_prob_clean.npy"))
    p.add_argument("--save-column", type=Path, default=Path("output/column/gs_ply_clean/0000_column_prob_clean.npy"))
    p.add_argument("--damage-thresh", type=float, default=0.15)
    p.add_argument("--min-damage-count", type=int, default=3000)
    p.add_argument("--highlight-alpha", type=float, default=0.85)
    p.add_argument("--highlight-min-alpha", type=float, default=0.30)
    p.add_argument("--highlight-binary", action="store_true")
    p.add_argument("--highlight-transparency", type=float, default=0.70)
    p.add_argument("--highlight-opacity-boost", type=float, default=0.25)
    p.add_argument("--damage-only-opacity", type=float, default=0.22)

    return p.parse_args()


def infer_damage_label_id(label_root: Path, fallback: int) -> int:
    candidate_names = ("damage", "spalling", "spaling")
    for dataset_dir in sorted(label_root.glob("*_dataset")):
        mapping_path = dataset_dir / "label_name_to_value.txt"
        if not mapping_path.exists():
            continue
        parsed: dict[str, int] = {}
        for raw_line in mapping_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0].strip().lower()
            try:
                value = int(parts[-1])
            except ValueError:
                continue
            parsed[name] = value
        for name in candidate_names:
            if name in parsed:
                print(f"[info] inferred damage label id={parsed[name]} from {mapping_path}")
                return parsed[name]
    print(f"[info] using damage label id={fallback}")
    return fallback


def load_custom_supervision(
    label_root: Path,
    view_root: Path,
    damage_label_id: int,
    spall_only_supervision: bool,
) -> dict[str, object]:
    dataset_dirs = sorted(label_root.glob("*_dataset"))
    if not dataset_dirs:
        raise RuntimeError(f"No label datasets found under: {label_root}")

    view_names: list[str] = []
    images: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    extrinsics: list[np.ndarray] = []
    spall_masks: list[np.ndarray] = []
    col_masks: list[np.ndarray] = []

    for dataset_dir in dataset_dirs:
        view_name = dataset_dir.name.removesuffix("_dataset")
        img_path = dataset_dir / "img.png"
        label_path = dataset_dir / "label.png"
        cam_path = view_root / f"{view_name}_camera.json"
        depth_path = view_root / f"{view_name}_depth.npy"

        missing = [str(p) for p in (img_path, label_path, cam_path, depth_path) if not p.exists()]
        if missing:
            print(f"[warn] skipping {view_name}, missing: {', '.join(missing)}")
            continue

        img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        label = np.asarray(Image.open(label_path).convert("L"), dtype=np.uint8)
        depth = np.load(depth_path).astype(np.float32)
        camera = json.loads(cam_path.read_text(encoding="utf-8"))
        intr = np.asarray(camera["intrinsics"], dtype=np.float32)
        extr = np.asarray(camera["w2c_extrinsics"], dtype=np.float32)

        h, w = img.shape[:2]
        if label.shape != (h, w):
            raise ValueError(f"Label shape mismatch for {label_path}: {label.shape} vs {(h, w)}")
        if depth.shape != (h, w):
            raise ValueError(f"Depth shape mismatch for {depth_path}: {depth.shape} vs {(h, w)}")
        if intr.shape != (3, 3):
            raise ValueError(f"Bad intrinsics shape for {cam_path}: {intr.shape}")
        if extr.shape != (4, 4):
            raise ValueError(f"Bad extrinsics shape for {cam_path}: {extr.shape}")

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
        raise RuntimeError("No valid labeled GS views were found.")

    target_h = min(x.shape[0] for x in images)
    target_w = min(x.shape[1] for x in images)
    mixed_res = any(x.shape[:2] != (target_h, target_w) for x in images)
    if mixed_res:
        print(f"[warn] mixed resolutions found, resizing all views to {target_h}x{target_w}")

    out_images: list[np.ndarray] = []
    out_depths: list[np.ndarray] = []
    out_intrinsics: list[np.ndarray] = []
    out_spall: list[np.ndarray] = []
    out_col: list[np.ndarray] = []

    for img, depth, intr, spall, col in zip(images, depths, intrinsics, spall_masks, col_masks):
        src_h, src_w = img.shape[:2]
        sx = target_w / src_w
        sy = target_h / src_h
        if (src_h, src_w) != (target_h, target_w):
            img = np.asarray(Image.fromarray(img).resize((target_w, target_h), Image.BILINEAR), dtype=np.uint8)
            depth = np.asarray(
                Image.fromarray(depth.astype(np.float32), mode="F").resize((target_w, target_h), Image.BILINEAR),
                dtype=np.float32,
            )
            spall = np.asarray(
                Image.fromarray((spall * 255).astype(np.uint8), mode="L").resize((target_w, target_h), Image.NEAREST),
                dtype=np.uint8,
            )
            col = np.asarray(
                Image.fromarray((col * 255).astype(np.uint8), mode="L").resize((target_w, target_h), Image.NEAREST),
                dtype=np.uint8,
            )
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

    print(f"[info] loaded {len(view_names)} labeled GS views, image={target_h}x{target_w}")
    print(f"[info] views: {view_names}")

    return {
        "view_names": view_names,
        "images": np.stack(out_images, axis=0),
        "depths": np.stack(out_depths, axis=0),
        "intrinsics": np.stack(out_intrinsics, axis=0),
        "extrinsics": np.stack(extrinsics, axis=0),
        "spall_masks": np.stack(out_spall, axis=0),
        "col_masks": np.stack(out_col, axis=0),
    }


def load_gs_ply(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    means = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    sh_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)
    log_scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32)
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)
    opacity_logit = np.asarray(v["opacity"], dtype=np.float32)
    return {
        "means": torch.tensor(means, device=device),
        "sh_dc": torch.tensor(sh_dc, device=device),
        "log_scales": torch.tensor(log_scales, device=device),
        "quats": torch.tensor(quats, device=device),
        "opacity_logit": torch.tensor(opacity_logit, device=device),
    }


def build_gaussians(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    quats: torch.Tensor,
    sh_dc: torch.Tensor,
    opacity_logit: torch.Tensor,
) -> Gaussians:
    scales = torch.exp(log_scales).clamp(1e-5, 30.0)
    quats = F.normalize(quats, dim=-1)
    opacities = torch.sigmoid(opacity_logit).clamp(1e-4, 1 - 1e-4)
    harmonics = sh_dc[..., None]
    return Gaussians(
        means=means[None],
        scales=scales[None],
        rotations=quats[None],
        harmonics=harmonics[None],
        opacities=opacities[None],
    )


def render_view(
    gaussians: Gaussians,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    h: int,
    w: int,
    use_sh: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    intr_normed = intrinsics.clone()
    intr_normed[:, 0, :] /= w
    intr_normed[:, 1, :] /= h
    rgb, depth = render_3dgs(
        extrinsics=extrinsics,
        intrinsics=intr_normed,
        image_shape=(h, w),
        gaussian=gaussians,
        use_sh=use_sh,
        num_view=1,
        color_mode="RGB+D",
    )
    return rgb[0], depth[0]


def _logit(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-4, 1 - 1e-4)
    return torch.log(x / (1 - x))


def weighted_bce_prob(prob: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    prob = prob.clamp(1e-4, 1 - 1e-4)
    return -(pos_weight * target * torch.log(prob) + (1.0 - target) * torch.log(1.0 - prob)).mean()


def make_optimizer(
    means: torch.nn.Parameter,
    log_scales: torch.nn.Parameter,
    quats: torch.nn.Parameter,
    sh_dc: torch.nn.Parameter,
    opacity_logit: torch.nn.Parameter,
    column_logit: torch.nn.Parameter,
    damage_logit: torch.nn.Parameter,
    args: argparse.Namespace,
) -> torch.optim.Adam:
    return torch.optim.Adam(
        [
            {"params": [means, log_scales, quats], "lr": args.lr_geo},
            {"params": [sh_dc, opacity_logit], "lr": args.lr_color},
            {"params": [column_logit], "lr": args.lr_column},
            {"params": [damage_logit], "lr": args.lr_damage},
        ]
    )


def save_gray(path: Path, x: torch.Tensor) -> None:
    arr = (x.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    args.damage_label_id = infer_damage_label_id(args.label_root, args.damage_label_id)
    sup = load_custom_supervision(
        label_root=args.label_root,
        view_root=args.view_root,
        damage_label_id=args.damage_label_id,
        spall_only_supervision=args.spall_only_supervision,
    )

    images = sup["images"]
    depths = sup["depths"]
    intrinsics = sup["intrinsics"]
    extrinsics = sup["extrinsics"]
    spall_masks = sup["spall_masks"]
    col_masks = sup["col_masks"]
    view_names = sup["view_names"]

    n_view, h, w, _ = images.shape
    rgb_gt = torch.from_numpy(images).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    depth_gt = torch.from_numpy(depths).to(device=device, dtype=torch.float32)
    intr_t = torch.from_numpy(intrinsics).to(device=device, dtype=torch.float32)
    extr_t = torch.from_numpy(extrinsics).to(device=device, dtype=torch.float32)
    sem_spall_gt = torch.from_numpy(spall_masks).to(device=device, dtype=torch.float32)
    sem_col_gt = torch.from_numpy(col_masks).to(device=device, dtype=torch.float32)

    gs_init = load_gs_ply(args.gs_ply, device=device)
    means = torch.nn.Parameter(gs_init["means"].clone())
    sh_dc = torch.nn.Parameter(gs_init["sh_dc"].clone())
    log_scales = torch.nn.Parameter(gs_init["log_scales"].clone())
    quats = torch.nn.Parameter(gs_init["quats"].clone())
    opacity_logit = torch.nn.Parameter(gs_init["opacity_logit"].clone())
    column_logit = torch.nn.Parameter(torch.full((means.shape[0],), args.column_init_logit, device=device))
    damage_logit = torch.nn.Parameter(torch.full((means.shape[0],), args.damage_init_logit, device=device))

    means_ref = gs_init["means"].clone().detach()
    log_scales_ref = gs_init["log_scales"].clone().detach()
    sh_dc_ref = gs_init["sh_dc"].clone().detach()
    opacity_ref = gs_init["opacity_logit"].clone().detach()

    optimizer = make_optimizer(
        means=means,
        log_scales=log_scales,
        quats=quats,
        sh_dc=sh_dc,
        opacity_logit=opacity_logit,
        column_logit=column_logit,
        damage_logit=damage_logit,
        args=args,
    )

    args.debug_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        in_stage1 = step <= args.stage1_steps
        in_stage2 = not in_stage1

        if random.random() < args.labeled_sample_prob:
            vi = random.randrange(n_view)
        else:
            vi = random.randrange(n_view)

        g = build_gaussians(means, log_scales, quats, sh_dc, opacity_logit)
        rgb_pred, depth_pred = render_view(g, extr_t[vi:vi + 1], intr_t[vi:vi + 1], h, w, use_sh=True)

        col_gt = sem_col_gt[vi]
        dmg_gt = sem_spall_gt[vi]
        labeled_mask = col_gt >= 0.0

        depth_valid = depth_gt[vi] > 0
        if args.photo_depth_column_only:
            photo_mask = col_gt > 0.5
            if photo_mask.any():
                loss_photo = torch.mean(torch.abs(rgb_pred - rgb_gt[vi]).mean(dim=0)[photo_mask])
                depth_valid = depth_valid & photo_mask
            else:
                loss_photo = torch.mean(torch.abs(rgb_pred - rgb_gt[vi]))
        else:
            loss_photo = torch.mean(torch.abs(rgb_pred - rgb_gt[vi]))
        loss_depth = F.smooth_l1_loss(depth_pred[depth_valid], depth_gt[vi][depth_valid]) if depth_valid.any() else torch.tensor(0.0, device=device)

        # Column render
        col_harm = torch.sigmoid(column_logit)[:, None, None].repeat(1, 3, 1)
        col_g = Gaussians(
            means=g.means,
            scales=g.scales,
            rotations=g.rotations,
            opacities=g.opacities,
            harmonics=col_harm[None],
        )
        col_rgb, _ = render_view(col_g, extr_t[vi:vi + 1], intr_t[vi:vi + 1], h, w, use_sh=False)
        col_pred = col_rgb.mean(dim=0).clamp(1e-4, 1 - 1e-4)

        valid_col = col_gt > 0.5
        out_col = col_gt < 0.5
        loss_col = weighted_bce_prob(col_pred[valid_col], col_gt[valid_col], args.column_pos_weight) if valid_col.any() else torch.tensor(0.0, device=device)
        loss_col_out = col_pred[out_col].mean() if out_col.any() else torch.tensor(0.0, device=device)

        # Damage render
        if args.soft_column_gate:
            col_gate_g = torch.sigmoid(column_logit)
        else:
            col_gate_g = (torch.sigmoid(column_logit).detach() >= args.column_thresh_train).float()
        damage_prob_g = torch.sigmoid(damage_logit)
        damage_prob_gated = damage_prob_g * col_gate_g

        loss_sem = torch.tensor(0.0, device=device)
        loss_out_col = torch.tensor(0.0, device=device)
        if in_stage2:
            sem_harm = damage_prob_gated[:, None, None].repeat(1, 3, 1)
            sem_g = Gaussians(
                means=g.means,
                scales=g.scales,
                rotations=g.rotations,
                opacities=g.opacities,
                harmonics=sem_harm[None],
            )
            sem_rgb, _ = render_view(sem_g, extr_t[vi:vi + 1], intr_t[vi:vi + 1], h, w, use_sh=False)
            sem_pred = sem_rgb.mean(dim=0).clamp(1e-4, 1 - 1e-4)
            valid_sem = (col_gt > 0.5) if args.sem_column_only else labeled_mask
            if valid_sem.any():
                loss_sem = weighted_bce_prob(sem_pred[valid_sem], dmg_gt[valid_sem], args.sem_pos_weight)
            if out_col.any():
                loss_out_col = sem_pred[out_col].mean()
        else:
            sem_pred = None

        loss_sparse = torch.sigmoid(damage_logit).mean() if in_stage2 else torch.tensor(0.0, device=device)
        loss_damage_out = (
            torch.sigmoid(damage_logit) * (1.0 - torch.sigmoid(column_logit))
        ).mean() if in_stage2 else torch.tensor(0.0, device=device)

        loss_geom = F.smooth_l1_loss(means, means_ref) + F.smooth_l1_loss(log_scales, log_scales_ref)
        loss_color_reg = F.smooth_l1_loss(sh_dc, sh_dc_ref)
        loss_opacity_reg = F.smooth_l1_loss(opacity_logit, opacity_ref)
        loss_scale_reg = F.smooth_l1_loss(log_scales, log_scales_ref)
        col_loss_weight = args.column_weight if in_stage1 else args.column_weight * args.stage2_column_loss_scale

        loss = (
            args.photo_weight * loss_photo
            + args.depth_weight * loss_depth
            + col_loss_weight * loss_col
            + args.semantic_weight * loss_sem
            + args.outside_column_penalty * loss_col_out
            + args.outside_column_penalty * loss_out_col
            + args.damage_outside_column_weight * loss_damage_out
            + args.sparsity_weight * loss_sparse
            + args.geom_reg_weight * loss_geom
            + args.color_reg_weight * loss_color_reg
            + args.opacity_reg_weight * loss_opacity_reg
            + args.scale_reg_weight * loss_scale_reg
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if in_stage1:
            damage_logit.grad = None
        if in_stage2 and args.freeze_column_stage2:
            column_logit.grad = None
        if step <= args.freeze_geo_steps:
            means.grad = None
            log_scales.grad = None
            quats.grad = None

        optimizer.step()

        with torch.no_grad():
            log_scales.data.clamp_(
                min=log_scales_ref + np.log(max(args.scale_min_mult, 1e-3)),
                max=log_scales_ref + np.log(max(args.scale_max_mult, 1.0)),
            )
            opacity_logit.data.clamp_(
                min=_logit(torch.tensor(1e-4, device=device)),
                max=_logit(torch.tensor(args.opacity_max, device=device)),
            )

        if step % args.debug_every == 0 or step == 1:
            stem = f"step{step:05d}_{view_names[vi]}"
            save_gray(args.debug_dir / f"{stem}_colpred.png", col_pred)
            save_gray(args.debug_dir / f"{stem}_gtcol.png", col_gt)
            if sem_pred is not None:
                save_gray(args.debug_dir / f"{stem}_sempred.png", sem_pred)
                save_gray(args.debug_dir / f"{stem}_gtdmg.png", dmg_gt)

        if step % args.print_every == 0 or step == 1:
            print(
                f"[{step:05d}/{args.steps}][{'S1' if in_stage1 else 'S2'}] "
                f"photo={loss_photo.item():.4f} depth={loss_depth.item():.4f} "
                f"col={loss_col.item():.4f} col_out={loss_col_out.item():.4f} "
                f"sem={loss_sem.item():.4f} out={loss_out_col.item():.4f} "
                f"d_out={loss_damage_out.item():.4f} sparse={loss_sparse.item():.4f} "
                f"total={loss.item():.4f} n_g={means.shape[0]} view={view_names[vi]}"
            )

    with torch.no_grad():
        final_scales = torch.exp(log_scales).clamp(1e-5, 30.0)
        final_quats = F.normalize(quats, dim=-1)
        final_harm = sh_dc[..., None]
        export_ply(
            means=means,
            scales=final_scales,
            rotations=final_quats,
            harmonics=final_harm,
            opacities=opacity_logit,
            path=args.save_ply,
            shift_and_scale=False,
            save_sh_dc_only=True,
            match_3dgs_mcmc_dev=False,
        )

        damage_prob_t = torch.sigmoid(damage_logit)
        column_prob_t = torch.sigmoid(column_logit)
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
            print(f"[warn] no gaussians above thresh, fallback to top-{k}")

        base_rgb = torch.sigmoid(sh_dc)
        base_opacity_prob = torch.sigmoid(opacity_logit)

        # damage highlight
        highlight_rgb = torch.tensor([1.0, 0.05, 0.05], device=device).unsqueeze(0)
        alpha_score = damage_mask_bool.float() if args.highlight_binary else damage_score
        alpha_score = torch.where(
            damage_mask_bool,
            torch.maximum(alpha_score, torch.full_like(alpha_score, args.highlight_min_alpha)),
            torch.zeros_like(alpha_score),
        ).clamp(0.0, 1.0)
        alpha = (args.highlight_alpha * alpha_score).unsqueeze(-1).clamp(0.0, 1.0)
        blended_rgb = (1.0 - alpha) * base_rgb + alpha * highlight_rgb
        highlight_harm = _logit(blended_rgb)[..., None]
        overlay_opacity_prob = (
            base_opacity_prob * (1.0 - args.highlight_transparency * alpha.squeeze(-1))
            + args.highlight_opacity_boost * alpha.squeeze(-1)
        ).clamp(1e-4, 1.0 - 1e-4)
        overlay_opacity_logit = _logit(overlay_opacity_prob)
        export_ply(
            means=means,
            scales=final_scales,
            rotations=final_quats,
            harmonics=highlight_harm,
            opacities=overlay_opacity_logit,
            path=args.save_highlight_ply,
            shift_and_scale=False,
            save_sh_dc_only=True,
            match_3dgs_mcmc_dev=False,
        )

        # column highlight
        column_score = ((column_prob_t - args.column_thresh_export) / (1.0 - args.column_thresh_export + 1e-6)).clamp(0.0, 1.0)
        column_mask_bool = column_prob_t >= args.column_thresh_export
        column_alpha_score = torch.where(
            column_mask_bool,
            torch.maximum(column_score, torch.full_like(column_score, args.highlight_min_alpha)),
            torch.zeros_like(column_score),
        ).clamp(0.0, 1.0)
        column_alpha = (args.highlight_alpha * column_alpha_score).unsqueeze(-1).clamp(0.0, 1.0)
        column_rgb = torch.tensor([0.05, 1.0, 0.10], device=device).unsqueeze(0)
        column_blended_rgb = (1.0 - column_alpha) * base_rgb + column_alpha * column_rgb
        column_highlight_harm = _logit(column_blended_rgb)[..., None]
        column_overlay_opacity_prob = (
            base_opacity_prob * (1.0 - args.highlight_transparency * column_alpha.squeeze(-1))
            + args.highlight_opacity_boost * column_alpha.squeeze(-1)
        ).clamp(1e-4, 1.0 - 1e-4)
        column_overlay_opacity_logit = _logit(column_overlay_opacity_prob)
        export_ply(
            means=means,
            scales=final_scales,
            rotations=final_quats,
            harmonics=column_highlight_harm,
            opacities=column_overlay_opacity_logit,
            path=args.save_column_highlight_ply,
            shift_and_scale=False,
            save_sh_dc_only=True,
            match_3dgs_mcmc_dev=False,
        )

        # damage only
        sel_means = means[damage_mask_bool]
        sel_scales = final_scales[damage_mask_bool]
        sel_quats = final_quats[damage_mask_bool]
        sel_opacity = _logit(torch.full((sel_means.shape[0],), args.damage_only_opacity, device=device))
        sel_color = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 3).repeat(sel_means.shape[0], 1)
        sel_harm = _logit(sel_color)[..., None]
        export_ply(
            means=sel_means,
            scales=sel_scales,
            rotations=sel_quats,
            harmonics=sel_harm,
            opacities=sel_opacity,
            path=args.save_damage_only_ply,
            shift_and_scale=False,
            save_sh_dc_only=True,
            match_3dgs_mcmc_dev=False,
        )

        args.save_damage.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_damage, damage_prob_t.detach().cpu().numpy())
        np.save(args.save_column, column_prob_t.detach().cpu().numpy())

        print(f"[done] saved optimized ply: {args.save_ply}")
        print(f"[done] saved damage highlight ply: {args.save_highlight_ply}")
        print(f"[done] saved column highlight ply: {args.save_column_highlight_ply}")
        print(f"[done] saved damage-only ply: {args.save_damage_only_ply}")
        print(f"[done] saved damage prob: {args.save_damage}")
        print(f"[done] saved column prob: {args.save_column}")


if __name__ == "__main__":
    main()
