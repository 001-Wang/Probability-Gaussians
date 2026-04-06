#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData

GSPLAT_EXAMPLES = Path("/home/zuoxu/project/3dgs/gsplat/examples")
if str(GSPLAT_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(GSPLAT_EXAMPLES))

from gsplat.rendering import rasterization  # noqa: E402


@dataclass
class Metrics:
    tp: int
    fp: int
    fn: int
    tn: int
    iou: float
    precision: float
    recall: float
    f1: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate only the added viewer highlight area against GT.")
    p.add_argument("--gs-ply", type=Path, default=Path("output/column/gs_ply_clean/0000_damage_opt_clean.ply"))
    p.add_argument("--column-prob", type=Path, default=Path("output/column/gs_ply_clean/0000_column_prob_clean.npy"))
    p.add_argument("--damage-prob", type=Path, default=Path("output/column/gs_ply_clean/0000_damage_prob_clean.npy"))
    p.add_argument("--view-root", type=Path, default=Path("output/column/real_gs_saved_test"))
    p.add_argument("--gt-root", type=Path, default=Path("assets/examples/column/real_gs_saved_test_converted"))
    p.add_argument("--column-thresh", type=float, default=0.90)
    p.add_argument("--damage-thresh", type=float, default=0.15)
    p.add_argument("--min-damage-count", type=int, default=3000)
    p.add_argument("--highlight-alpha", type=float, default=0.85)
    p.add_argument("--highlight-min-alpha", type=float, default=0.30)
    p.add_argument("--highlight-transparency", type=float, default=0.70)
    p.add_argument("--highlight-opacity-boost", type=float, default=0.25)
    p.add_argument("--delta-r-min", type=float, default=0.20)
    p.add_argument("--delta-rg-min", type=float, default=0.15)
    p.add_argument("--delta-rb-min", type=float, default=0.15)
    p.add_argument("--highlight-red-min", type=float, default=0.35)
    p.add_argument("--damage-label-id", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-dir", type=Path, default=Path("output/column/test_metrics_viewer"))
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
            try:
                parsed[parts[0].strip().lower()] = int(parts[-1])
            except ValueError:
                continue
        for name in candidate_names:
            if name in parsed:
                print(f"[info] inferred damage label id={parsed[name]} from {mapping_path}")
                return parsed[name]
    print(f"[info] using damage label id={fallback}")
    return fallback


def load_ply_as_gsplat_tensors(path: Path, device: torch.device):
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    names = set(v.data.dtype.names or [])

    means = torch.tensor(np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32), device=device)
    quats = torch.tensor(
        np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32),
        device=device,
    )
    scales = torch.tensor(
        np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32)),
        device=device,
    )
    opacities = torch.tensor(np.asarray(v["opacity"], dtype=np.float32), device=device)
    sh0 = torch.tensor(
        np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32),
        device=device,
    )

    f_rest_names = sorted(
        [name for name in names if name.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if f_rest_names:
        shn_flat = np.stack([v[name] for name in f_rest_names], axis=-1).astype(np.float32)
        if shn_flat.shape[1] % 3 != 0:
            raise ValueError(f"Unexpected f_rest property count: {shn_flat.shape[1]}")
        shn = torch.tensor(shn_flat.reshape(shn_flat.shape[0], -1, 3), device=device)
        colors = torch.cat([sh0[:, None, :], shn], dim=1)
        sh_degree = int(math.sqrt(colors.shape[1]) - 1)
    else:
        colors = sh0[:, None, :]
        sh_degree = 0
    return means, F.normalize(quats, dim=-1), scales, opacities, colors, sh_degree


def make_highlight_colors(
    base_colors: torch.Tensor,
    score: torch.Tensor,
    highlight_rgb: tuple[float, float, float],
    alpha_scale: float,
    alpha_floor: float,
) -> torch.Tensor:
    base_rgb = torch.sigmoid(base_colors[:, 0, :]).clamp(0.0, 1.0)
    active = score > 0
    alpha = torch.where(
        active,
        torch.maximum(score, torch.full_like(score, alpha_floor)),
        torch.zeros_like(score),
    )
    alpha = (alpha_scale * alpha).unsqueeze(-1).clamp(0.0, 1.0)
    target_rgb = torch.tensor(highlight_rgb, device=base_colors.device, dtype=base_colors.dtype).unsqueeze(0)
    blended_rgb = (1.0 - alpha) * base_rgb + alpha * target_rgb

    out = base_colors.clone()
    out[:, 0, :] = torch.logit(blended_rgb.clamp(1e-4, 1.0 - 1e-4))
    if out.shape[1] > 1:
        out[:, 1:, :] = torch.where(active[:, None, None], torch.zeros_like(out[:, 1:, :]), out[:, 1:, :])
    return out


def make_highlight_opacity(
    base_opacity_logit: torch.Tensor,
    score: torch.Tensor,
    alpha_scale: float,
    alpha_floor: float,
    transparency: float,
    boost: float,
) -> torch.Tensor:
    active = score > 0
    alpha = torch.where(
        active,
        torch.maximum(score, torch.full_like(score, alpha_floor)),
        torch.zeros_like(score),
    )
    alpha = (alpha_scale * alpha).clamp(0.0, 1.0)
    base_prob = torch.sigmoid(base_opacity_logit)
    out_prob = (
        base_prob * (1.0 - transparency * alpha)
        + boost * alpha
    ).clamp(1e-4, 1.0 - 1e-4)
    return torch.logit(out_prob)


@torch.no_grad()
def render_rgb(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacity_logit: torch.Tensor,
    colors: torch.Tensor,
    sh_degree: int,
    w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    h: int,
    w: int,
) -> np.ndarray:
    render_colors, _, _ = rasterization(
        means,
        quats,
        scales,
        torch.sigmoid(opacity_logit),
        colors,
        w2c[None],
        intrinsics[None],
        w,
        h,
        sh_degree=sh_degree,
        near_plane=1e-2,
        far_plane=1e2,
        radius_clip=0.0,
        eps2d=0.3,
        backgrounds=torch.zeros((1, 3), device=means.device),
        render_mode="RGB",
        rasterize_mode="classic",
        camera_model="pinhole",
        packed=False,
    )
    return render_colors[0].clamp(0, 1).detach().cpu().numpy()


def extract_added_highlight_mask(base_rgb: np.ndarray, hi_rgb: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    base_r = base_rgb[..., 0]
    base_g = base_rgb[..., 1]
    base_b = base_rgb[..., 2]
    hi_r = hi_rgb[..., 0]
    hi_g = hi_rgb[..., 1]
    hi_b = hi_rgb[..., 2]

    delta_r = hi_r - base_r
    delta_rg = (hi_r - hi_g) - (base_r - base_g)
    delta_rb = (hi_r - hi_b) - (base_r - base_b)

    mask = (
        (hi_r >= args.highlight_red_min)
        & (delta_r >= args.delta_r_min)
        & (delta_rg >= args.delta_rg_min)
        & (delta_rb >= args.delta_rb_min)
    )
    return mask.astype(np.uint8)


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Metrics:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = int(np.logical_and(~pred_bool, gt_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~gt_bool).sum())
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return Metrics(tp=tp, fp=fp, fn=fn, tn=tn, iou=iou, precision=precision, recall=recall, f1=f1)


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Metrics:
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return Metrics(tp=tp, fp=fp, fn=fn, tn=tn, iou=iou, precision=precision, recall=recall, f1=f1)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def save_gray(path: Path, x: np.ndarray) -> None:
    x = np.clip(x, 0.0, 1.0)
    Image.fromarray((x * 255).astype(np.uint8)).save(path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def save_overlay(path: Path, image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> None:
    out = image.copy()
    gt_only = np.logical_and(gt.astype(bool), ~pred.astype(bool))
    pred_only = np.logical_and(pred.astype(bool), ~gt.astype(bool))
    both = np.logical_and(gt.astype(bool), pred.astype(bool))
    out[gt_only] = (0.20 * out[gt_only] + 0.80 * np.array([0, 0, 255])).astype(np.uint8)
    out[pred_only] = (0.20 * out[pred_only] + 0.80 * np.array([255, 0, 0])).astype(np.uint8)
    out[both] = (0.20 * out[both] + 0.80 * np.array([0, 255, 0])).astype(np.uint8)
    Image.fromarray(out).save(path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    args.damage_label_id = infer_damage_label_id(args.gt_root, args.damage_label_id)
    means, quats, scales, opacities, colors, sh_degree = load_ply_as_gsplat_tensors(args.gs_ply, device)
    column_prob = torch.from_numpy(np.load(args.column_prob).astype(np.float32)).to(device=device)
    damage_prob = torch.from_numpy(np.load(args.damage_prob).astype(np.float32)).to(device=device)
    if column_prob.shape[0] != means.shape[0]:
        raise ValueError(f"column-prob length {column_prob.shape[0]} does not match gaussians {means.shape[0]}")
    if damage_prob.shape[0] != means.shape[0]:
        raise ValueError(f"damage-prob length {damage_prob.shape[0]} does not match gaussians {means.shape[0]}")

    column_mask = column_prob >= args.column_thresh
    gated_damage = damage_prob * column_mask.float()
    damage_score = ((gated_damage - args.damage_thresh) / (1.0 - args.damage_thresh + 1e-6)).clamp(0.0, 1.0)
    if not torch.any(damage_score > 0):
        k = min(max(args.min_damage_count, 1), int(gated_damage.numel()))
        topk_idx = torch.topk(gated_damage, k=k).indices
        damage_score = torch.zeros_like(gated_damage)
        damage_score[topk_idx] = 1.0
        print(f"[warn] no damage above threshold; fallback to top-{k}")

    hi_colors = make_highlight_colors(
        colors,
        damage_score,
        (1.0, 0.10, 0.05),
        args.highlight_alpha,
        args.highlight_min_alpha,
    )
    hi_opacity = make_highlight_opacity(
        opacities,
        damage_score,
        args.highlight_alpha,
        args.highlight_min_alpha,
        args.highlight_transparency,
        args.highlight_opacity_boost,
    )

    rows: list[tuple[str, Metrics]] = []
    for dataset_dir in sorted(args.gt_root.glob("*_dataset")):
        view_name = dataset_dir.name.removesuffix("_dataset")
        camera_path = args.view_root / f"{view_name}_camera.json"
        gt_path = dataset_dir / "label.png"
        img_path = dataset_dir / "img.png"
        if not camera_path.exists() or not gt_path.exists() or not img_path.exists():
            print(f"[warn] skipping {view_name}: missing files")
            continue

        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        K = torch.from_numpy(np.asarray(camera["intrinsics"], dtype=np.float32)).to(device=device)
        w2c = torch.from_numpy(np.asarray(camera["w2c_extrinsics"], dtype=np.float32)).to(device=device)
        gt_img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        gt_label = np.asarray(Image.open(gt_path).convert("L"), dtype=np.uint8)
        h, w = gt_label.shape
        gt_damage = (gt_label == args.damage_label_id).astype(np.uint8)

        base_rgb = render_rgb(means, quats, scales, opacities, colors, sh_degree, w2c, K, h, w)
        hi_rgb = render_rgb(means, quats, scales, hi_opacity, hi_colors, sh_degree, w2c, K, h, w)
        pred_damage = extract_added_highlight_mask(base_rgb, hi_rgb, args)
        m = compute_metrics(pred_damage, gt_damage)
        rows.append((view_name, m))

        delta_r = np.clip(hi_rgb[..., 0] - base_rgb[..., 0], 0.0, 1.0)
        delta_rg = np.clip((hi_rgb[..., 0] - hi_rgb[..., 1]) - (base_rgb[..., 0] - base_rgb[..., 1]), 0.0, 1.0)

        save_rgb(args.out_dir / f"{view_name}_base.png", base_rgb)
        save_rgb(args.out_dir / f"{view_name}_highlight.png", hi_rgb)
        save_gray(args.out_dir / f"{view_name}_delta_r.png", delta_r)
        save_gray(args.out_dir / f"{view_name}_delta_rg.png", delta_rg)
        save_mask(args.out_dir / f"{view_name}_pred_damage.png", pred_damage)
        save_overlay(args.out_dir / f"{view_name}_overlay_damage.png", gt_img, gt_damage, pred_damage)

        print(
            f"[{view_name}] damage IoU={m.iou:.4f} P={m.precision:.4f} R={m.recall:.4f} F1={m.f1:.4f}"
        )

    if not rows:
        raise RuntimeError("No valid test views found.")

    overall = metrics_from_counts(
        sum(m.tp for _, m in rows),
        sum(m.fp for _, m in rows),
        sum(m.fn for _, m in rows),
        sum(m.tn for _, m in rows),
    )

    summary = {
        "config": {
            "gs_ply": str(args.gs_ply.resolve()),
            "column_prob": str(args.column_prob.resolve()),
            "damage_prob": str(args.damage_prob.resolve()),
            "view_root": str(args.view_root.resolve()),
            "gt_root": str(args.gt_root.resolve()),
            "column_thresh": args.column_thresh,
            "damage_thresh": args.damage_thresh,
            "min_damage_count": args.min_damage_count,
            "delta_r_min": args.delta_r_min,
            "delta_rg_min": args.delta_rg_min,
            "delta_rb_min": args.delta_rb_min,
            "highlight_red_min": args.highlight_red_min,
            "damage_label_id": args.damage_label_id,
        },
        "damage": {
            "overall": overall.__dict__,
            "per_view": {name: m.__dict__ for name, m in rows},
        },
    }
    (args.out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "[overall] "
        f"damage IoU={overall.iou:.4f} "
        f"P={overall.precision:.4f} "
        f"R={overall.recall:.4f} "
        f"F1={overall.f1:.4f}"
    )
    print(f"[done] saved viewer-delta eval to {args.out_dir}")


if __name__ == "__main__":
    main()
