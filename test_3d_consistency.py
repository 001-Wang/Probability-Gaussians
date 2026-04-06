#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate 3D consistency metrics for mv_c2f Gaussian predictions.")
    p.add_argument("--gs-ply", type=Path, default=Path("output/column/gs_ply_mv_c2f/0000_damage_opt_mv_c2f.ply"))
    p.add_argument("--damage-prob", type=Path, default=Path("output/column/gs_ply_mv_c2f/0000_damage_prob_mv_c2f.npy"))
    p.add_argument("--label-root", type=Path, default=Path("assets/examples/column/real_gs_saved_test_converted"))
    p.add_argument("--view-root", type=Path, default=Path("output/column/real_gs_saved_test_gt"))
    p.add_argument("--damage-label-id", type=int, default=2)
    p.add_argument("--spall-only-supervision", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--depth-tau-rel", type=float, default=0.08)
    p.add_argument("--depth-tau-abs", type=float, default=0.20)
    p.add_argument("--label-radius", type=int, default=2)
    p.add_argument("--no-infer-damage-label-id", action="store_true")
    p.add_argument("--out-json", type=Path, default=Path("output/column/gs_ply_mv_c2f/metrics_3d_consistency.json"))
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


def load_means(path: Path, device: torch.device) -> torch.Tensor:
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    means = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    return torch.tensor(means, device=device)


def load_supervision(label_root: Path, view_root: Path, damage_label_id: int, spall_only_supervision: bool):
    dataset_dirs = sorted(label_root.glob("*_dataset"))
    if not dataset_dirs:
        raise RuntimeError(f"No label datasets found under: {label_root}")

    view_names = []
    depths = []
    intrinsics = []
    extrinsics = []
    labels = []
    raw_labels = []

    for dataset_dir in dataset_dirs:
        view_name = dataset_dir.name.removesuffix("_dataset")
        label_path = dataset_dir / "label.png"
        cam_path = view_root / f"{view_name}_camera.json"
        depth_path = view_root / f"{view_name}_depth.npy"
        missing = [str(p) for p in (label_path, cam_path, depth_path) if not p.exists()]
        if missing:
            print(f"[warn] skipping {view_name}, missing: {', '.join(missing)}")
            continue

        label = np.asarray(Image.open(label_path).convert("L"), dtype=np.uint8)
        depth = np.load(depth_path).astype(np.float32)
        camera = json.loads(cam_path.read_text(encoding="utf-8"))
        intr = np.asarray(camera["intrinsics"], dtype=np.float32)
        extr = np.asarray(camera["w2c_extrinsics"], dtype=np.float32)
        if label.shape != depth.shape:
            raise ValueError(f"shape mismatch for {view_name}: {label.shape} vs {depth.shape}")

        if spall_only_supervision:
            obs_label = (label == damage_label_id).astype(np.float32)
        else:
            obs_label = (label > 0).astype(np.float32)

        view_names.append(view_name)
        depths.append(depth)
        intrinsics.append(intr)
        extrinsics.append(extr)
        labels.append(obs_label)
        raw_labels.append(label.astype(np.uint8))

    if not view_names:
        raise RuntimeError("No valid labeled views found")

    target_h = min(x.shape[0] for x in depths)
    target_w = min(x.shape[1] for x in depths)
    mixed_res = any(x.shape != (target_h, target_w) for x in depths)
    if mixed_res:
        print(f"[warn] mixed resolutions found, resizing all views to {target_h}x{target_w}")

    out_depths = []
    out_intrinsics = []
    out_labels = []
    out_raw_labels = []
    for depth, intr, obs_label, raw_label in zip(depths, intrinsics, labels, raw_labels):
        src_h, src_w = depth.shape
        sx = target_w / src_w
        sy = target_h / src_h
        if (src_h, src_w) != (target_h, target_w):
            depth = np.asarray(
                Image.fromarray(depth.astype(np.float32), mode="F").resize((target_w, target_h), Image.BILINEAR),
                dtype=np.float32,
            )
            obs_label = np.asarray(
                Image.fromarray((obs_label * 255).astype(np.uint8), mode="L").resize((target_w, target_h), Image.NEAREST),
                dtype=np.uint8,
            )
            obs_label = (obs_label > 127).astype(np.float32)
            raw_label = np.asarray(
                Image.fromarray(raw_label, mode="L").resize((target_w, target_h), Image.NEAREST),
                dtype=np.uint8,
            )
        intr = intr.copy()
        intr[0, 0] *= sx
        intr[0, 2] *= sx
        intr[1, 1] *= sy
        intr[1, 2] *= sy
        out_depths.append(depth)
        out_intrinsics.append(intr)
        out_labels.append(obs_label)
        out_raw_labels.append(raw_label)

    return {
        "view_names": view_names,
        "depths": torch.from_numpy(np.stack(out_depths, axis=0)),
        "intrinsics": torch.from_numpy(np.stack(out_intrinsics, axis=0)),
        "extrinsics": torch.from_numpy(np.stack(extrinsics, axis=0)),
        "labels": torch.from_numpy(np.stack(out_labels, axis=0)),
        "raw_labels": torch.from_numpy(np.stack(out_raw_labels, axis=0)),
    }


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
    return ui, vi, z, in_img


def sample_label_neighborhood(label_map: torch.Tensor, ui: torch.Tensor, vi: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return label_map[vi, ui]
    h, w = label_map.shape
    pooled = torch.zeros_like(ui, dtype=torch.float32, device=label_map.device)
    for dy in range(-radius, radius + 1):
        yy = (vi + dy).clamp(0, h - 1)
        for dx in range(-radius, radius + 1):
            xx = (ui + dx).clamp(0, w - 1)
            pooled = torch.maximum(pooled, label_map[yy, xx])
    return pooled


def collect_view_observations(
    means: torch.Tensor,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    labels: torch.Tensor,
    depth_tau_rel: float,
    depth_tau_abs: float,
    label_radius: int,
):
    n_view = depths.shape[0]
    n_gauss = means.shape[0]
    obs = torch.zeros((n_view, n_gauss), dtype=torch.float32, device=means.device)
    visible = torch.zeros((n_view, n_gauss), dtype=torch.bool, device=means.device)
    depth_supported = torch.zeros((n_view, n_gauss), dtype=torch.bool, device=means.device)
    depth_abs_err = torch.full((n_view, n_gauss), float("nan"), dtype=torch.float32, device=means.device)

    for i in range(n_view):
        h, w = depths[i].shape
        ui, vi, z, in_img = project_gaussians(means, extrinsics[i], intrinsics[i], h, w)
        gt_depth = depths[i, vi, ui]
        depth_ok = gt_depth > 0

        vis = in_img & depth_ok
        err = (z - gt_depth).abs()
        tau = torch.maximum(depth_tau_rel * gt_depth.abs(), torch.full_like(gt_depth, depth_tau_abs))
        supported = vis & (err < tau)

        visible[i] = vis
        depth_supported[i] = supported
        obs[i] = sample_label_neighborhood(labels[i], ui, vi, label_radius)
        depth_abs_err[i] = torch.where(vis, err, torch.full_like(err, float("nan")))

    return obs, visible, depth_supported, depth_abs_err


def compute_label_consistency(obs: torch.Tensor, visible: torch.Tensor, view_names: list[str]):
    pair_details = []
    pair_num = torch.tensor(0.0, device=obs.device)
    pair_den = torch.tensor(0.0, device=obs.device)
    n_view = len(view_names)
    for i in range(n_view):
        for j in range(i + 1, n_view):
            overlap = visible[i] & visible[j]
            overlap_count = int(overlap.sum().item())
            if overlap_count == 0:
                continue
            disagreement = (obs[i, overlap] - obs[j, overlap]).abs()
            pair_sum = disagreement.sum()
            pair_mean = disagreement.mean()
            pair_num += pair_sum
            pair_den += float(overlap_count)
            pair_details.append(
                {
                    "view_a": view_names[i],
                    "view_b": view_names[j],
                    "overlap_gaussians": overlap_count,
                    "label_disagreement_mean": float(pair_mean.item()),
                    "label_disagreement_sum": float(pair_sum.item()),
                }
            )
    value = float((pair_num / pair_den.clamp_min(1.0)).item()) if pair_den.item() > 0 else 0.0
    return value, pair_details


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    if args.no_infer_damage_label_id:
        print(f"[info] using user-specified damage label id={args.damage_label_id}")
    else:
        args.damage_label_id = infer_damage_label_id(args.label_root, args.damage_label_id)
    supervision = load_supervision(args.label_root, args.view_root, args.damage_label_id, args.spall_only_supervision)

    means = load_means(args.gs_ply, device)
    pred_prob = torch.from_numpy(np.load(args.damage_prob).astype(np.float32)).to(device)
    if pred_prob.shape[0] != means.shape[0]:
        raise ValueError("Damage probability array length does not match number of gaussians")

    depths = supervision["depths"].to(device)
    intrinsics = supervision["intrinsics"].to(device)
    extrinsics = supervision["extrinsics"].to(device)
    labels = supervision["labels"].to(device)
    raw_labels = supervision["raw_labels"].to(device)
    view_names = supervision["view_names"]

    column_labels = (raw_labels > 0).float()
    spalling_labels = (raw_labels == args.damage_label_id).float()

    obs, visible, depth_supported, depth_abs_err = collect_view_observations(
        means,
        depths,
        intrinsics,
        extrinsics,
        labels,
        args.depth_tau_rel,
        args.depth_tau_abs,
        args.label_radius,
    )

    obs_column, visible_column, _, _ = collect_view_observations(
        means,
        depths,
        intrinsics,
        extrinsics,
        column_labels,
        args.depth_tau_rel,
        args.depth_tau_abs,
        args.label_radius,
    )
    obs_spalling, visible_spalling, _, _ = collect_view_observations(
        means,
        depths,
        intrinsics,
        extrinsics,
        spalling_labels,
        args.depth_tau_rel,
        args.depth_tau_abs,
        args.label_radius,
    )

    e_label_cons, pair_details = compute_label_consistency(obs, visible, view_names)
    e_label_cons_column, pair_details_column = compute_label_consistency(obs_column, visible_column, view_names)
    e_label_cons_spalling, pair_details_spalling = compute_label_consistency(obs_spalling, visible_spalling, view_names)

    # E_pred
    visible_count = visible.sum(dim=0)
    observed_mask = visible_count > 0
    obs_mean = (obs * visible.float()).sum(dim=0) / visible_count.clamp_min(1).float()
    e_pred = (
        float((pred_prob[observed_mask] - obs_mean[observed_mask]).abs().mean().item())
        if observed_mask.any()
        else 0.0
    )

    # R_depth
    total_visible = int(visible.sum().item())
    total_supported = int(depth_supported.sum().item())
    r_depth = float(total_supported / total_visible) if total_visible > 0 else 0.0

    per_view = {}
    for i, view_name in enumerate(view_names):
        view_visible = visible[i]
        view_supported = depth_supported[i]
        view_total = int(view_visible.sum().item())
        view_good = int(view_supported.sum().item())
        mean_err = float(torch.nanmean(depth_abs_err[i]).item()) if view_total > 0 else 0.0
        per_view[view_name] = {
            "visible_gaussians": view_total,
            "depth_supported_gaussians": view_good,
            "depth_support_ratio": float(view_good / view_total) if view_total > 0 else 0.0,
            "mean_depth_abs_error": mean_err,
            "observed_positive_ratio": float(obs[i, view_visible].mean().item()) if view_total > 0 else 0.0,
        }

    result = {
        "config": {
            "gs_ply": str(args.gs_ply),
            "damage_prob": str(args.damage_prob),
            "label_root": str(args.label_root),
            "view_root": str(args.view_root),
            "damage_label_id": args.damage_label_id,
            "spall_only_supervision": args.spall_only_supervision,
            "depth_tau_rel": args.depth_tau_rel,
            "depth_tau_abs": args.depth_tau_abs,
            "label_radius": args.label_radius,
        },
        "summary": {
            "num_views": len(view_names),
            "num_gaussians": int(means.shape[0]),
            "num_visible_gaussians_any_view": int(observed_mask.sum().item()),
            "total_visible_observations": total_visible,
            "total_depth_supported_observations": total_supported,
            "E_label_cons": e_label_cons,
            "E_label_cons_column": e_label_cons_column,
            "E_label_cons_spalling": e_label_cons_spalling,
            "E_pred": e_pred,
            "R_depth": r_depth,
        },
        "per_view": per_view,
        "pairwise_label_consistency": pair_details,
        "pairwise_label_consistency_column": pair_details_column,
        "pairwise_label_consistency_spalling": pair_details_spalling,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"[done] saved 3D consistency metrics to {args.out_json}")


if __name__ == "__main__":
    main()
