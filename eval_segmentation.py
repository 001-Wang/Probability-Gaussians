#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval_test_view_metrics_viewer import (
    Metrics,
    compute_metrics,
    extract_added_highlight_mask,
    infer_damage_label_id,
    load_ply_as_gsplat_tensors,
    make_highlight_colors,
    make_highlight_opacity,
    metrics_from_counts,
    render_rgb,
    save_gray,
    save_mask,
    save_overlay,
    save_rgb,
)
from view_trained_gs import (
    column_axis_core_mask,
    largest_voxel_component_mask,
    load_ply_arrays,
)


def extract_added_column_mask(base_rgb: np.ndarray, hi_rgb: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    base_r = base_rgb[..., 0]
    base_g = base_rgb[..., 1]
    base_b = base_rgb[..., 2]
    hi_r = hi_rgb[..., 0]
    hi_g = hi_rgb[..., 1]
    hi_b = hi_rgb[..., 2]

    delta_g = hi_g - base_g
    delta_gr = (hi_g - hi_r) - (base_g - base_r)
    delta_gb = (hi_g - hi_b) - (base_g - base_b)

    mask = (
        (hi_g >= args.highlight_green_min)
        & (delta_g >= args.delta_g_min)
        & (delta_gr >= args.delta_gr_min)
        & (delta_gb >= args.delta_gb_min)
    )
    return mask.astype(np.uint8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate viewer-style strict filtered highlights against GT masks.")
    p.add_argument("--gs-ply", type=Path, default=Path("output/column/gs_ply_mv_c2f/0000_damage_opt_mv_c2f.ply"))
    p.add_argument("--column-prob", type=Path, default=Path("output/column/gs_ply_mv_c2f/0000_column_prob_mv_c2f.npy"))
    p.add_argument("--damage-prob", type=Path, default=Path("output/column/gs_ply_mv_c2f/0000_damage_prob_mv_c2f.npy"))
    p.add_argument("--view-root", type=Path, default=Path("output/column/real_gs_saved_test"))
    p.add_argument("--gt-root", type=Path, default=Path("assets/examples/column/real_gs_saved_test_converted"))
    p.add_argument("--damage-label-id", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-dir", type=Path, default=Path("output/column/test_metrics_viewer_strict_mv_c2f"))

    p.add_argument("--column-thresh", type=float, default=0.995)
    p.add_argument("--damage-thresh", type=float, default=0.95)
    p.add_argument("--min-damage-count", type=int, default=5000)
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

    p.add_argument("--highlight-alpha", type=float, default=1.0)
    p.add_argument("--highlight-min-alpha", type=float, default=0.8)
    p.add_argument("--highlight-transparency", type=float, default=0.7)
    p.add_argument("--highlight-opacity-boost", type=float, default=0.35)

    p.add_argument("--delta-r-min", type=float, default=0.20)
    p.add_argument("--delta-rg-min", type=float, default=0.15)
    p.add_argument("--delta-rb-min", type=float, default=0.15)
    p.add_argument("--highlight-red-min", type=float, default=0.35)
    p.add_argument("--delta-g-min", type=float, default=0.20)
    p.add_argument("--delta-gr-min", type=float, default=0.15)
    p.add_argument("--delta-gb-min", type=float, default=0.15)
    p.add_argument("--highlight-green-min", type=float, default=0.35)
    return p.parse_args()


def accuracy_from_metrics(m: Metrics) -> float:
    total = m.tp + m.fp + m.fn + m.tn
    return (m.tp + m.tn) / max(total, 1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

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

    damage_keep = (raw_damage_prob >= args.damage_thresh) & (opacity_prob_np >= args.damage_opacity_min)
    if args.gate_damage_by_column:
        damage_keep &= column_keep

    column_prob = torch.from_numpy(np.where(column_keep, raw_column_prob, 0.0).astype(np.float32)).to(device=device)
    damage_prob = torch.from_numpy(np.where(damage_keep, raw_damage_prob, 0.0).astype(np.float32)).to(device=device)

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

    column_score = ((column_prob - args.column_thresh) / (1.0 - args.column_thresh + 1e-6)).clamp(0.0, 1.0)
    hi_colors_column = make_highlight_colors(
        colors,
        column_score,
        (0.05, 1.0, 0.10),
        args.highlight_alpha,
        args.highlight_min_alpha,
    )
    hi_opacity_column = make_highlight_opacity(
        opacities,
        column_score,
        args.highlight_alpha,
        args.highlight_min_alpha,
        args.highlight_transparency,
        args.highlight_opacity_boost,
    )

    damage_rows: list[tuple[str, Metrics]] = []
    column_rows: list[tuple[str, Metrics]] = []
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
        gt_column = (gt_label > 0).astype(np.uint8)

        base_rgb = render_rgb(means, quats, scales, opacities, colors, sh_degree, w2c, K, h, w)
        hi_rgb_damage = render_rgb(means, quats, scales, hi_opacity, hi_colors, sh_degree, w2c, K, h, w)
        hi_rgb_column = render_rgb(means, quats, scales, hi_opacity_column, hi_colors_column, sh_degree, w2c, K, h, w)
        pred_damage = extract_added_highlight_mask(base_rgb, hi_rgb_damage, args)
        pred_column = extract_added_column_mask(base_rgb, hi_rgb_column, args)
        damage_m = compute_metrics(pred_damage, gt_damage)
        column_m = compute_metrics(pred_column, gt_column)
        damage_rows.append((view_name, damage_m))
        column_rows.append((view_name, column_m))

        delta_r = np.clip(hi_rgb_damage[..., 0] - base_rgb[..., 0], 0.0, 1.0)
        delta_rg = np.clip((hi_rgb_damage[..., 0] - hi_rgb_damage[..., 1]) - (base_rgb[..., 0] - base_rgb[..., 1]), 0.0, 1.0)
        delta_g = np.clip(hi_rgb_column[..., 1] - base_rgb[..., 1], 0.0, 1.0)
        delta_gr = np.clip((hi_rgb_column[..., 1] - hi_rgb_column[..., 0]) - (base_rgb[..., 1] - base_rgb[..., 0]), 0.0, 1.0)

        save_rgb(args.out_dir / f"{view_name}_base.png", base_rgb)
        save_rgb(args.out_dir / f"{view_name}_highlight_damage.png", hi_rgb_damage)
        save_rgb(args.out_dir / f"{view_name}_highlight_column.png", hi_rgb_column)
        save_gray(args.out_dir / f"{view_name}_delta_r.png", delta_r)
        save_gray(args.out_dir / f"{view_name}_delta_rg.png", delta_rg)
        save_gray(args.out_dir / f"{view_name}_delta_g.png", delta_g)
        save_gray(args.out_dir / f"{view_name}_delta_gr.png", delta_gr)
        save_mask(args.out_dir / f"{view_name}_pred_damage.png", pred_damage)
        save_mask(args.out_dir / f"{view_name}_pred_column.png", pred_column)
        save_overlay(args.out_dir / f"{view_name}_overlay_damage.png", gt_img, gt_damage, pred_damage)
        save_overlay(args.out_dir / f"{view_name}_overlay_column.png", gt_img, gt_column, pred_column)

        damage_acc = accuracy_from_metrics(damage_m)
        column_acc = accuracy_from_metrics(column_m)
        print(
            f"[{view_name}] damage IoU={damage_m.iou:.4f} P={damage_m.precision:.4f} "
            f"R={damage_m.recall:.4f} F1={damage_m.f1:.4f} Acc={damage_acc:.4f}"
        )
        print(
            f"[{view_name}] column IoU={column_m.iou:.4f} P={column_m.precision:.4f} "
            f"R={column_m.recall:.4f} F1={column_m.f1:.4f} Acc={column_acc:.4f}"
        )

    if not damage_rows:
        raise RuntimeError("No valid test views found.")

    damage_overall = metrics_from_counts(
        sum(m.tp for _, m in damage_rows),
        sum(m.fp for _, m in damage_rows),
        sum(m.fn for _, m in damage_rows),
        sum(m.tn for _, m in damage_rows),
    )
    column_overall = metrics_from_counts(
        sum(m.tp for _, m in column_rows),
        sum(m.fp for _, m in column_rows),
        sum(m.fn for _, m in column_rows),
        sum(m.tn for _, m in column_rows),
    )
    damage_overall_accuracy = accuracy_from_metrics(damage_overall)
    column_overall_accuracy = accuracy_from_metrics(column_overall)

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
            "min_damage_count": args.min_damage_count,
            "delta_r_min": args.delta_r_min,
            "delta_rg_min": args.delta_rg_min,
            "delta_rb_min": args.delta_rb_min,
            "highlight_red_min": args.highlight_red_min,
            "delta_g_min": args.delta_g_min,
            "delta_gr_min": args.delta_gr_min,
            "delta_gb_min": args.delta_gb_min,
            "highlight_green_min": args.highlight_green_min,
            "damage_label_id": args.damage_label_id,
        },
        "damage": {
            "overall": {**damage_overall.__dict__, "accuracy": damage_overall_accuracy},
            "per_view": {
                name: {**m.__dict__, "accuracy": accuracy_from_metrics(m)}
                for name, m in damage_rows
            },
        },
        "column": {
            "overall": {**column_overall.__dict__, "accuracy": column_overall_accuracy},
            "per_view": {
                name: {**m.__dict__, "accuracy": accuracy_from_metrics(m)}
                for name, m in column_rows
            },
        },
    }
    (args.out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "[overall] "
        f"damage IoU={damage_overall.iou:.4f} "
        f"P={damage_overall.precision:.4f} "
        f"R={damage_overall.recall:.4f} "
        f"F1={damage_overall.f1:.4f} "
        f"Acc={damage_overall_accuracy:.4f}"
    )
    print(
        "[overall] "
        f"column IoU={column_overall.iou:.4f} "
        f"P={column_overall.precision:.4f} "
        f"R={column_overall.recall:.4f} "
        f"F1={column_overall.f1:.4f} "
        f"Acc={column_overall_accuracy:.4f}"
    )
    print(f"[done] saved strict viewer-delta eval to {args.out_dir}")


if __name__ == "__main__":
    main()
