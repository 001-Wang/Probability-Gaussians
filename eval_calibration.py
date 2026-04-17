#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from gsplat.rendering import rasterization

from eval_test_view_metrics_viewer import (
    infer_damage_label_id,
    load_ply_as_gsplat_tensors,
    save_gray,
)
from view_trained_gs import (
    column_axis_core_mask,
    largest_voxel_component_mask,
    load_ply_arrays,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate pixel-space calibration of rendered GS damage probabilities.")
    p.add_argument("--gs-ply", type=Path, default=Path("output/column/gs_ply_mv_c2f/0000_damage_opt_mv_c2f.ply"))
    p.add_argument("--column-prob", type=Path, default=Path("output/column/gs_ply_mv_c2f/0000_column_prob_mv_c2f.npy"))
    p.add_argument("--damage-prob", type=Path, default=Path("output/column/gs_ply_mv_c2f/0000_damage_prob_mv_c2f.npy"))
    p.add_argument("--view-root", type=Path, default=Path("output/column/real_gs_saved_test"))
    p.add_argument("--gt-root", type=Path, default=Path("assets/examples/column/real_gs_saved_test_gt"))
    p.add_argument("--damage-label-id", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-dir", type=Path, default=Path("output/column/test_calibration_mv_c2f"))

    p.add_argument("--column-thresh", type=float, default=0.995)
    p.add_argument("--damage-thresh", type=float, default=0.95)
    p.add_argument("--column-opacity-min", type=float, default=0.08)
    p.add_argument("--damage-opacity-min", type=float, default=0.08)
    p.add_argument("--gate-damage-by-column", action="store_true", default=True)
    p.add_argument("--no-gate-damage-by-column", action="store_false", dest="gate_damage_by_column")
    p.add_argument("--keep-largest-column-component", action="store_true", default=True)
    p.add_argument("--no-keep-largest-column-component", action="store_false", dest="keep_largest_column_component")
    p.add_argument("--column-voxel-size", type=float, default=0.12)
    p.add_argument("--keep-column-axis-core", action="store_true", default=True)
    p.add_argument("--no-keep-column-axis-core", action="store_false", dest="keep_column_axis_core")
    p.add_argument("--column-radial-quantile", type=float, default=0.97)
    p.add_argument("--calibration-mode", choices=("raw", "soft_gated", "hard_filtered"), default="soft_gated")
    p.add_argument("--mask-pred-to-column", action="store_true", default=False)
    p.add_argument("--no-mask-pred-to-column", action="store_false", dest="mask_pred_to_column")

    p.add_argument("--num-bins", type=int, default=15)
    p.add_argument("--mask-to-column", action="store_true", default=False)
    p.add_argument("--save-prob-maps", action="store_true", default=True)
    p.add_argument("--no-save-prob-maps", action="store_false", dest="save_prob_maps")
    return p.parse_args()


def resolve_gt_root(gt_root: Path) -> Path:
    if gt_root.exists():
        return gt_root
    fallback = Path("output/column/real_gs_saved_test_gt")
    if gt_root.name == fallback.name and fallback.exists():
        print(f"[warn] gt-root {gt_root} does not exist; using fallback {fallback}")
        return fallback
    raise FileNotFoundError(
        f"GT root not found: {gt_root}. Expected a directory containing <view_name>_dataset folders, "
        f"for example {fallback}."
    )


def render_probability_map(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh_degree: int,
    scalar_prob: torch.Tensor,
    w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    h: int,
    w: int,
) -> np.ndarray:
    del sh_degree  # Probability rendering should not use SH interpretation.
    prob_rgb = scalar_prob[:, None].repeat(1, 3)
    render_colors, _, _ = rasterization(
        means,
        quats,
        scales,
        torch.sigmoid(opacities),
        prob_rgb,
        w2c[None],
        intrinsics[None],
        w,
        h,
        sh_degree=None,
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
    return render_colors[0, ..., 0].clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)


def compute_ece(prob: np.ndarray, label: np.ndarray, num_bins: int) -> tuple[float, list[dict[str, float | int | list[float]]]]:
    prob_flat = np.clip(prob.reshape(-1).astype(np.float64), 0.0, 1.0)
    label_flat = label.reshape(-1).astype(np.float64)
    n = int(prob_flat.size)
    if n == 0:
        raise ValueError("ECE requires at least one sample")

    edges = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)
    ece = 0.0
    bins: list[dict[str, float | int | list[float]]] = []
    for idx in range(num_bins):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        if idx == num_bins - 1:
            in_bin = (prob_flat >= lo) & (prob_flat <= hi)
        else:
            in_bin = (prob_flat >= lo) & (prob_flat < hi)
        count = int(in_bin.sum())
        if count == 0:
            bins.append(
                {
                    "bin_index": idx,
                    "range": [lo, hi],
                    "count": 0,
                    "weight": 0.0,
                    "accuracy": 0.0,
                    "confidence": 0.0,
                    "gap": 0.0,
                }
            )
            continue

        acc = float(label_flat[in_bin].mean())
        conf = float(prob_flat[in_bin].mean())
        weight = count / n
        gap = abs(acc - conf)
        ece += weight * gap
        bins.append(
            {
                "bin_index": idx,
                "range": [lo, hi],
                "count": count,
                "weight": weight,
                "accuracy": acc,
                "confidence": conf,
                "gap": gap,
            }
        )
    return float(ece), bins


def compute_brier(prob: np.ndarray, label: np.ndarray) -> float:
    prob_flat = np.clip(prob.reshape(-1).astype(np.float64), 0.0, 1.0)
    label_flat = label.reshape(-1).astype(np.float64)
    if prob_flat.size == 0:
        raise ValueError("Brier score requires at least one sample")
    return float(np.mean((prob_flat - label_flat) ** 2))


def maybe_mask_to_column(prob: np.ndarray, gt_damage: np.ndarray, gt_column: np.ndarray, use_mask: bool) -> tuple[np.ndarray, np.ndarray]:
    if not use_mask:
        return prob, gt_damage
    valid = gt_column.astype(bool)
    return prob[valid], gt_damage[valid]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.gt_root = resolve_gt_root(args.gt_root)

    args.damage_label_id = infer_damage_label_id(args.gt_root, args.damage_label_id)
    means, quats, scales, opacities, colors, sh_degree = load_ply_as_gsplat_tensors(args.gs_ply, device)
    raw_column_prob = np.load(args.column_prob).astype(np.float32)
    raw_damage_prob = np.load(args.damage_prob).astype(np.float32)
    ply_means, opacity_prob_np = load_ply_arrays(args.gs_ply)

    if not (raw_column_prob.shape[0] == raw_damage_prob.shape[0] == ply_means.shape[0] == means.shape[0]):
        raise ValueError("Probability arrays and PLY gaussian count do not match")

    column_keep = (raw_column_prob >= args.column_thresh) & (opacity_prob_np >= args.column_opacity_min)
    if args.keep_largest_column_component:
        submask = largest_voxel_component_mask(ply_means[column_keep], args.column_voxel_size)
        keep_idx = np.flatnonzero(column_keep)
        refined = np.zeros_like(column_keep)
        refined[keep_idx[submask]] = True
        column_keep = refined
    if args.keep_column_axis_core and column_keep.any():
        submask = column_axis_core_mask(ply_means[column_keep], args.column_radial_quantile)
        keep_idx = np.flatnonzero(column_keep)
        refined = np.zeros_like(column_keep)
        refined[keep_idx[submask]] = True
        column_keep = refined

    raw_column_prob_t = torch.from_numpy(raw_column_prob.astype(np.float32)).to(device=device)
    raw_damage_prob_t = torch.from_numpy(raw_damage_prob.astype(np.float32)).to(device=device)
    column_keep_t = torch.from_numpy(column_keep.astype(np.float32)).to(device=device)

    if args.calibration_mode == "raw":
        column_prob_t = raw_column_prob_t
        damage_prob_t = raw_damage_prob_t
        pred_damage_prob_t = damage_prob_t
    elif args.calibration_mode == "soft_gated":
        column_prob_t = raw_column_prob_t
        if args.mask_pred_to_column:
            column_prob_t = column_prob_t * column_keep_t
        damage_prob_t = raw_damage_prob_t
        pred_damage_prob_t = damage_prob_t * column_prob_t if args.gate_damage_by_column else damage_prob_t
    else:
        damage_keep = (raw_damage_prob >= args.damage_thresh) & (opacity_prob_np >= args.damage_opacity_min)
        if args.gate_damage_by_column:
            damage_keep &= column_keep
        column_prob_np = np.where(column_keep, raw_column_prob, 0.0).astype(np.float32)
        damage_prob_np = np.where(damage_keep, raw_damage_prob, 0.0).astype(np.float32)
        column_prob_t = torch.from_numpy(column_prob_np).to(device=device)
        damage_prob_t = torch.from_numpy(damage_prob_np).to(device=device)
        pred_damage_prob_t = damage_prob_t * (column_prob_t >= args.column_thresh).float()

    per_view: dict[str, dict[str, object]] = {}
    overall_prob: list[np.ndarray] = []
    overall_gt: list[np.ndarray] = []

    dataset_dirs = sorted(args.gt_root.glob("*_dataset"))
    if not dataset_dirs:
        raise RuntimeError(
            f"No <view_name>_dataset folders found under {args.gt_root}. "
            "This evaluator expects GT folders like view_0000_dataset/label.png."
        )

    for dataset_dir in dataset_dirs:
        view_name = dataset_dir.name.removesuffix("_dataset")
        camera_path = args.view_root / f"{view_name}_camera.json"
        gt_path = dataset_dir / "label.png"
        if not camera_path.exists() or not gt_path.exists():
            print(f"[warn] skipping {view_name}: missing files")
            continue

        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        intrinsics = torch.from_numpy(np.asarray(camera["intrinsics"], dtype=np.float32)).to(device=device)
        w2c = torch.from_numpy(np.asarray(camera["w2c_extrinsics"], dtype=np.float32)).to(device=device)
        gt_label = np.asarray(Image.open(gt_path).convert("L"), dtype=np.uint8)
        h, w = gt_label.shape

        gt_damage = (gt_label == args.damage_label_id).astype(np.uint8)
        gt_column = (gt_label > 0).astype(np.uint8)
        damage_prob_map = render_probability_map(
            means,
            quats,
            scales,
            opacities,
            sh_degree,
            pred_damage_prob_t,
            w2c,
            intrinsics,
            h,
            w,
        )
        eval_prob, eval_gt = maybe_mask_to_column(damage_prob_map, gt_damage, gt_column, args.mask_to_column)
        view_ece, bins = compute_ece(eval_prob, eval_gt, args.num_bins)
        view_brier = compute_brier(eval_prob, eval_gt)

        overall_prob.append(eval_prob.reshape(-1))
        overall_gt.append(eval_gt.reshape(-1))
        per_view[view_name] = {
            "ece": view_ece,
            "brier": view_brier,
            "num_pixels": int(eval_prob.size),
            "mean_confidence": float(np.mean(eval_prob)),
            "positive_rate": float(np.mean(eval_gt)),
            "bins": bins,
        }

        if args.save_prob_maps:
            save_gray(args.out_dir / f"{view_name}_damage_prob.png", damage_prob_map)
        print(
            f"[{view_name}] ECE={view_ece:.6f} "
            f"Brier={view_brier:.6f} "
            f"mean_conf={float(np.mean(eval_prob)):.6f} "
            f"positive_rate={float(np.mean(eval_gt)):.6f}"
        )

    if not overall_prob:
        raise RuntimeError(
            f"No valid test views found after scanning {args.gt_root}. "
            "Check that each GT folder contains label.png and each view has a matching *_camera.json in view-root."
        )

    overall_prob_flat = np.concatenate(overall_prob, axis=0)
    overall_gt_flat = np.concatenate(overall_gt, axis=0)
    overall_ece, overall_bins = compute_ece(overall_prob_flat, overall_gt_flat, args.num_bins)
    overall_brier = compute_brier(overall_prob_flat, overall_gt_flat)

    summary = {
        "config": {
            "gs_ply": str(args.gs_ply.resolve()),
            "column_prob": str(args.column_prob.resolve()),
            "damage_prob": str(args.damage_prob.resolve()),
            "view_root": str(args.view_root.resolve()),
            "gt_root": str(args.gt_root.resolve()),
            "column_thresh": args.column_thresh,
            "damage_thresh": args.damage_thresh,
            "column_opacity_min": args.column_opacity_min,
            "damage_opacity_min": args.damage_opacity_min,
            "column_voxel_size": args.column_voxel_size,
            "column_radial_quantile": args.column_radial_quantile,
            "calibration_mode": args.calibration_mode,
            "gate_damage_by_column": args.gate_damage_by_column,
            "mask_pred_to_column": args.mask_pred_to_column,
            "mask_to_column": args.mask_to_column,
            "num_bins": args.num_bins,
            "damage_label_id": args.damage_label_id,
        },
        "damage": {
            "overall": {
                "ece": overall_ece,
                "brier": overall_brier,
                "num_pixels": int(overall_prob_flat.size),
                "mean_confidence": float(np.mean(overall_prob_flat)),
                "positive_rate": float(np.mean(overall_gt_flat)),
                "bins": overall_bins,
            },
            "per_view": per_view,
        },
    }
    (args.out_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "[overall] "
        f"ECE={overall_ece:.6f} "
        f"Brier={overall_brier:.6f} "
        f"mean_conf={float(np.mean(overall_prob_flat)):.6f} "
        f"positive_rate={float(np.mean(overall_gt_flat)):.6f}"
    )
    print(f"[done] saved calibration eval to {args.out_dir}")


if __name__ == "__main__":
    main()
