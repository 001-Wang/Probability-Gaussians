#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from depth_anything_3.model.utils.gs_renderer import render_3dgs
from depth_anything_3.specs import Gaussians


def ensure_homo_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Unexpected extrinsics shape: {extrinsics.shape}")
    n = extrinsics.shape[0]
    ext_h = np.zeros((n, 4, 4), dtype=np.float32)
    ext_h[:, :3, :4] = extrinsics
    ext_h[:, 3, 3] = 1.0
    return ext_h


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
    harmonics_dc: torch.Tensor,
    opacity_logit: torch.Tensor,
) -> Gaussians:
    return Gaussians(
        means=means[None],
        scales=torch.exp(log_scales).clamp(1e-5, 30.0)[None],
        rotations=F.normalize(quats, dim=-1)[None],
        harmonics=harmonics_dc[..., None][None],
        opacities=torch.sigmoid(opacity_logit).clamp(1e-4, 1 - 1e-4)[None],
    )


def render_rgb(
    gaussians: Gaussians,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    h: int,
    w: int,
) -> np.ndarray:
    intr_normed = intrinsics.clone()
    intr_normed[:, 0, :] /= w
    intr_normed[:, 1, :] /= h
    rgb, _ = render_3dgs(
        extrinsics=extrinsics,
        intrinsics=intr_normed,
        image_shape=(h, w),
        gaussian=gaussians,
        use_sh=False,
        num_view=1,
        color_mode="RGB",
    )
    img = rgb[0].permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(img, 0.0, 1.0)


def empty_rgb(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.float32)


def project_points(
    xyz: torch.Tensor,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    h: int,
    w: int,
):
    xyz_h = torch.cat([xyz, torch.ones((xyz.shape[0], 1), device=xyz.device)], dim=-1)
    cam = xyz_h @ extrinsic.T
    z = cam[:, 2]
    valid_z = z > 1e-4

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    u = fx * (cam[:, 0] / z) + cx
    v = fy * (cam[:, 1] / z) + cy
    in_bounds = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, z, in_bounds


def to_uint8(img: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def blend_overlay(base: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.35) -> np.ndarray:
    out = (base * 255).astype(np.uint8).copy()
    m = mask > 0.5
    out[m] = ((1 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)).astype(np.uint8)
    return out


def load_view_from_npz(npz_path: Path, view_id: int):
    data = np.load(npz_path)
    images = data["image"]
    intrinsics = data["intrinsics"].astype(np.float32)
    extrinsics = ensure_homo_extrinsics(data["extrinsics"].astype(np.float32))
    n_view, h, w, _ = images.shape
    if not (0 <= view_id < n_view):
        raise ValueError(f"view-id must be in [0, {n_view - 1}]")
    name = f"view_{view_id:04d}"
    gt = images[view_id].astype(np.uint8)
    return name, gt, intrinsics[view_id], extrinsics[view_id], h, w


def load_view_from_custom_root(custom_view_root: Path, view_name: str):
    image_path = custom_view_root / f"{view_name}.png"
    camera_path = custom_view_root / f"{view_name}_camera.json"
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if not camera_path.exists():
        raise FileNotFoundError(camera_path)
    gt = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float32)
    extrinsics = np.asarray(camera["w2c_extrinsics"], dtype=np.float32)
    h, w = gt.shape[:2]
    return view_name, gt, intrinsics, extrinsics, h, w


def draw_panel_title(img: Image.Image, title: str) -> Image.Image:
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, min(out.width, 520), 28), fill=(0, 0, 0))
    draw.text((8, 6), title, fill=(255, 255, 255))
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=Path("output/column/exports/npz/results.npz"))
    parser.add_argument("--custom-view-root", type=Path, default=None)
    parser.add_argument("--gs-ply", type=Path, default=Path("output/column/gs_ply/0000.ply"))
    parser.add_argument("--column-prob", type=Path, default=Path("output/column/gs_ply/0000_column_prob.npy"))
    parser.add_argument("--damage-prob", type=Path, default=Path("output/column/gs_ply/0000_damage_prob.npy"))
    parser.add_argument("--view-id", type=int, default=None)
    parser.add_argument("--view-name", type=str, default=None)
    parser.add_argument("--thresh", type=float, default=0.7)
    parser.add_argument("--column-thresh", type=float, default=None)
    parser.add_argument("--damage-thresh", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--outdir", type=Path, default=Path("output/column/debug_proj"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.outdir.mkdir(parents=True, exist_ok=True)

    use_custom = args.custom_view_root is not None
    if use_custom:
        if args.view_name is None:
            raise ValueError("--view-name is required with --custom-view-root")
        view_name, gt_img, intrinsics_np, extrinsics_np, h, w = load_view_from_custom_root(
            args.custom_view_root, args.view_name
        )
    else:
        if args.view_id is None:
            raise ValueError("--view-id is required when using --npz")
        view_name, gt_img, intrinsics_np, extrinsics_np, h, w = load_view_from_npz(args.npz, args.view_id)

    intr_t = torch.from_numpy(intrinsics_np[None]).to(device=device)
    extr_t = torch.from_numpy(extrinsics_np[None]).to(device=device)

    gs = load_gs_ply(args.gs_ply, device=device)
    means = gs["means"]
    sh_dc = gs["sh_dc"]
    log_scales = gs["log_scales"]
    quats = gs["quats"]
    opacity_logit = gs["opacity_logit"]

    column_prob = torch.from_numpy(np.load(args.column_prob).astype(np.float32)).to(device=device)
    damage_prob = torch.from_numpy(np.load(args.damage_prob).astype(np.float32)).to(device=device)
    if column_prob.shape[0] != means.shape[0]:
        raise ValueError(
            f"column-prob length {column_prob.shape[0]} does not match gaussians {means.shape[0]}"
        )
    if damage_prob.shape[0] != means.shape[0]:
        raise ValueError(
            f"damage-prob length {damage_prob.shape[0]} does not match gaussians {means.shape[0]}"
        )
    column_thresh = args.column_thresh if args.column_thresh is not None else args.thresh
    damage_thresh = args.damage_thresh if args.damage_thresh is not None else args.thresh
    sel = column_prob >= column_thresh
    damage_sel = damage_prob >= damage_thresh
    print(
        f"[info] mode={'custom-view' if use_custom else 'npz'}, "
        f"view={view_name}, image={h}x{w}"
    )
    print(f"[info] selected column gaussians: {int(sel.sum().item())}/{sel.numel()} with thresh={column_thresh}")
    print(f"[info] selected damage gaussians: {int(damage_sel.sum().item())}/{damage_sel.numel()} with thresh={damage_thresh}")

    full_g = build_gaussians(means, log_scales, quats, sh_dc, opacity_logit)
    full_rgb = render_rgb(full_g, extr_t, intr_t, h, w)

    sel_means = means[sel]
    sel_scales = log_scales[sel]
    sel_quats = quats[sel]
    sel_opacity = opacity_logit[sel]

    if sel_means.shape[0] > 0:
        red_rgb = torch.zeros((sel_means.shape[0], 3), device=device)
        red_rgb[:, 0] = 1.0
        red_rgb = torch.logit(red_rgb.clamp(1e-4, 1 - 1e-4))
        col_g = build_gaussians(sel_means, sel_scales, sel_quats, red_rgb, sel_opacity)
        col_rgb = render_rgb(col_g, extr_t, intr_t, h, w)
        col_mask = (col_rgb[..., 0] > 0.05).astype(np.float32)
        u, v, z, valid = project_points(sel_means, extr_t[0], intr_t[0], h, w)
    else:
        col_rgb = empty_rgb(h, w)
        col_mask = np.zeros((h, w), dtype=np.float32)
        u = v = z = valid = None

    dmg_means = means[damage_sel]
    dmg_scales = log_scales[damage_sel]
    dmg_quats = quats[damage_sel]
    dmg_opacity = opacity_logit[damage_sel]
    if dmg_means.shape[0] > 0:
        yellow_rgb = torch.zeros((dmg_means.shape[0], 3), device=device)
        yellow_rgb[:, 0] = 1.0
        yellow_rgb[:, 1] = 1.0
        yellow_rgb = torch.logit(yellow_rgb.clamp(1e-4, 1 - 1e-4))
        dmg_g = build_gaussians(dmg_means, dmg_scales, dmg_quats, yellow_rgb, dmg_opacity)
        dmg_rgb = render_rgb(dmg_g, extr_t, intr_t, h, w)
        dmg_mask = (np.maximum(dmg_rgb[..., 0], dmg_rgb[..., 1]) > 0.05).astype(np.float32)
        du, dv, dz, dvalid = project_points(dmg_means, extr_t[0], intr_t[0], h, w)
    else:
        dmg_rgb = empty_rgb(h, w)
        dmg_mask = np.zeros((h, w), dtype=np.float32)
        du = dv = dz = dvalid = None

    gt_pil = Image.fromarray(gt_img)
    full_pil = to_uint8(full_rgb)
    col_pil = to_uint8(col_rgb)
    overlay_pil = Image.fromarray(blend_overlay(gt_img / 255.0, col_mask, color=(255, 0, 0), alpha=0.35))
    dmg_pil = to_uint8(dmg_rgb)
    dmg_overlay_pil = Image.fromarray(blend_overlay(gt_img / 255.0, dmg_mask, color=(255, 255, 0), alpha=0.35))

    pts_pil = gt_pil.copy()
    draw = ImageDraw.Draw(pts_pil)
    if u is not None:
        z_valid = z[valid]
        zmin = float(z_valid.min().item()) if z_valid.numel() > 0 else 0.0
        zmax = float(z_valid.max().item()) if z_valid.numel() > 0 else 1.0
        uu = u[valid].detach().cpu().numpy()
        vv = v[valid].detach().cpu().numpy()
        zz = z[valid].detach().cpu().numpy()
        for x, y, depth in zip(uu, vv, zz):
            t = (depth - zmin) / (zmax - zmin) if zmax > zmin else 0.5
            color = (255, int(255 * (1.0 - t)), int(255 * t))
            r = 2
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    dmg_pts_pil = gt_pil.copy()
    dmg_draw = ImageDraw.Draw(dmg_pts_pil)
    if du is not None:
        dz_valid = dz[dvalid]
        dzmin = float(dz_valid.min().item()) if dz_valid.numel() > 0 else 0.0
        dzmax = float(dz_valid.max().item()) if dz_valid.numel() > 0 else 1.0
        duu = du[dvalid].detach().cpu().numpy()
        dvv = dv[dvalid].detach().cpu().numpy()
        dzz = dz[dvalid].detach().cpu().numpy()
        for x, y, depth in zip(duu, dvv, dzz):
            t = (depth - dzmin) / (dzmax - dzmin) if dzmax > dzmin else 0.5
            color = (255, 255, int(255 * t))
            r = 2
            dmg_draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    gt_pil.save(args.outdir / f"{view_name}_gt.png")
    full_pil.save(args.outdir / f"{view_name}_full_render.png")
    col_pil.save(args.outdir / f"{view_name}_column_only_render.png")
    overlay_pil.save(args.outdir / f"{view_name}_column_overlay.png")
    pts_pil.save(args.outdir / f"{view_name}_projected_points.png")
    dmg_pil.save(args.outdir / f"{view_name}_damage_only_render.png")
    dmg_overlay_pil.save(args.outdir / f"{view_name}_damage_overlay.png")
    dmg_pts_pil.save(args.outdir / f"{view_name}_damage_projected_points.png")

    panel = Image.new("RGB", (w * 3, h * 3), (0, 0, 0))
    panel.paste(draw_panel_title(gt_pil, "reference"), (0, 0))
    panel.paste(draw_panel_title(full_pil, "full render"), (w, 0))
    panel.paste(draw_panel_title(col_pil, "column only"), (2 * w, 0))
    panel.paste(draw_panel_title(overlay_pil, "column overlay"), (0, h))
    panel.paste(draw_panel_title(pts_pil, "projected centers"), (w, h))
    panel.paste(draw_panel_title(dmg_pil, "damage only"), (2 * w, h))
    panel.paste(draw_panel_title(dmg_overlay_pil, "damage overlay"), (0, 2 * h))
    panel.paste(draw_panel_title(dmg_pts_pil, "damage centers"), (w, 2 * h))
    panel.save(args.outdir / f"{view_name}_panel.png")

    print(f"[done] saved to {args.outdir}")


if __name__ == "__main__":
    main()
