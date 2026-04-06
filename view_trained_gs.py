#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import viser
from PIL import Image
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[2]

from gsplat.rendering import rasterization  # noqa: E402
from local_gsplat_viewer import GsplatViewer, GsplatRenderTabState  # noqa: E402
from nerfview import CameraState, RenderTabState, apply_float_colormap  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("View trained Gaussian splats with column/damage highlight toggles.")
    parser.add_argument("--gs-ply", type=Path, default=Path("output/column/gs_ply_mv_c2f/0000_damage_opt_mv_c2f.ply"))
    parser.add_argument("--column-prob", type=Path, default=Path("output/column/gs_ply_mv_c2f/0000_column_prob_mv_c2f.npy"))
    parser.add_argument("--damage-prob", type=Path, default=Path("output/column/gs_ply_mv_c2f/0000_damage_prob_mv_c2f.npy"))
    parser.add_argument("--port", type=int, default=8893)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("output/column/trained_gs_viewer"))
    parser.add_argument("--background", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--with-ut", action="store_true")
    parser.add_argument("--with-eval3d", action="store_true")
    parser.add_argument("--column-thresh-export", type=float, default=0.97)
    parser.add_argument("--damage-thresh-export", type=float, default=0.95)
    parser.add_argument("--min-damage-count", type=int, default=5000)
    parser.add_argument("--highlight-alpha", type=float, default=1.0)
    parser.add_argument("--highlight-floor", type=float, default=0.8)
    parser.add_argument("--highlight-transparency", type=float, default=0.7)
    parser.add_argument("--highlight-opacity-boost", type=float, default=0.35)
    parser.add_argument("--strict-filter", action="store_true", default=True)
    parser.add_argument("--no-strict-filter", action="store_false", dest="strict_filter")
    parser.add_argument("--cache-dir", type=Path, default=Path("output/column/gs_ply_mv_c2f/strict_view_cache"))
    parser.add_argument("--column-opacity-min", type=float, default=0.08)
    parser.add_argument("--damage-opacity-min", type=float, default=0.08)
    parser.add_argument("--gate-damage-by-column", action="store_true", default=True)
    parser.add_argument("--no-gate-damage-by-column", action="store_false", dest="gate_damage_by_column")
    parser.add_argument("--keep-largest-column-component", action="store_true", default=True)
    parser.add_argument("--no-keep-largest-column-component", action="store_false", dest="keep_largest_column_component")
    parser.add_argument("--column-voxel-size", type=float, default=0.12)
    parser.add_argument("--keep-column-axis-core", action="store_true", default=True)
    parser.add_argument("--no-keep-column-axis-core", action="store_false", dest="keep_column_axis_core")
    parser.add_argument("--column-radial-quantile", type=float, default=0.985)
    return parser.parse_args()


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


def load_ply_arrays(gs_ply: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(gs_ply))
    v = ply["vertex"]
    means = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    opacity_logit = np.asarray(v["opacity"], dtype=np.float32)
    opacity = 1.0 / (1.0 + np.exp(-opacity_logit))
    return means, opacity


def largest_voxel_component_mask(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    coords = np.floor(points / max(voxel_size, 1e-6)).astype(np.int32)
    uniq, inv = np.unique(coords, axis=0, return_inverse=True)
    voxel_to_points: dict[int, list[int]] = {}
    for pi, vi in enumerate(inv):
        voxel_to_points.setdefault(int(vi), []).append(pi)

    lookup = {tuple(v.tolist()): i for i, v in enumerate(uniq)}
    visited = np.zeros((uniq.shape[0],), dtype=bool)
    best_component: list[int] = []
    neighbors = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    for start in range(uniq.shape[0]):
        if visited[start]:
            continue
        q: deque[int] = deque([start])
        visited[start] = True
        component_points: list[int] = []
        while q:
            cur = q.popleft()
            component_points.extend(voxel_to_points.get(cur, []))
            base = uniq[cur]
            for dx, dy, dz in neighbors:
                nxt = lookup.get((int(base[0] + dx), int(base[1] + dy), int(base[2] + dz)))
                if nxt is None or visited[nxt]:
                    continue
                visited[nxt] = True
                q.append(nxt)
        if len(component_points) > len(best_component):
            best_component = component_points

    keep = np.zeros((points.shape[0],), dtype=bool)
    if best_component:
        keep[np.asarray(best_component, dtype=np.int64)] = True
    return keep


def column_axis_core_mask(points: np.ndarray, radial_quantile: float) -> np.ndarray:
    if points.shape[0] <= 8:
        return np.ones((points.shape[0],), dtype=bool)
    center = points.mean(axis=0, keepdims=True)
    centered = points - center
    cov = centered.T @ centered / max(points.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, np.argmax(evals)]
    axis = axis / max(np.linalg.norm(axis), 1e-8)
    axial = centered @ axis
    radial_vec = centered - np.outer(axial, axis)
    radial = np.linalg.norm(radial_vec, axis=1)
    q = float(np.clip(radial_quantile, 0.5, 0.999))
    thr = float(np.quantile(radial, q))
    return radial <= thr


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
    tgt_rgb = torch.tensor(highlight_rgb, device=base_colors.device, dtype=base_colors.dtype).unsqueeze(0)
    blended_rgb = (1.0 - alpha) * base_rgb + alpha * tgt_rgb

    out = base_colors.clone()
    out[:, 0, :] = torch.logit(blended_rgb.clamp(1e-4, 1.0 - 1e-4))
    if out.shape[1] > 1:
        out[:, 1:, :] = torch.where(
            active[:, None, None],
            torch.zeros_like(out[:, 1:, :]),
            out[:, 1:, :],
        )
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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA device is required for the trained gsplat viewer.")

    means, quats, scales, opacities, colors, sh_degree = load_ply_as_gsplat_tensors(args.gs_ply, device)
    raw_column_prob_np = np.load(args.column_prob).astype(np.float32)
    raw_damage_prob_np = np.load(args.damage_prob).astype(np.float32)
    if raw_column_prob_np.shape[0] != means.shape[0]:
        raise ValueError(
            f"column-prob length {raw_column_prob_np.shape[0]} does not match gaussians {means.shape[0]}. "
            f"Use the PLY aligned with this NPY, likely 0000_damage_opt.ply."
        )
    if raw_damage_prob_np.shape[0] != means.shape[0]:
        raise ValueError(
            f"damage-prob length {raw_damage_prob_np.shape[0]} does not match gaussians {means.shape[0]}. "
            f"Use the PLY aligned with this NPY, likely 0000_damage_opt.ply."
        )
    column_prob_np = raw_column_prob_np
    damage_prob_np = raw_damage_prob_np
    strict_stats = None
    if args.strict_filter:
        ply_means, opacity_prob_np = load_ply_arrays(args.gs_ply)
        column_keep = (raw_column_prob_np >= args.column_thresh_export) & (opacity_prob_np >= args.column_opacity_min)
        raw_column_keep = int(column_keep.sum())
        if args.keep_largest_column_component:
            submask = largest_voxel_component_mask(ply_means[column_keep], args.column_voxel_size)
            keep_idx = np.flatnonzero(column_keep)
            refined = np.zeros_like(column_keep)
            refined[keep_idx[submask]] = True
            column_keep = refined
        after_component_keep = int(column_keep.sum())
        if args.keep_column_axis_core and after_component_keep > 0:
            submask = column_axis_core_mask(ply_means[column_keep], args.column_radial_quantile)
            keep_idx = np.flatnonzero(column_keep)
            refined = np.zeros_like(column_keep)
            refined[keep_idx[submask]] = True
            column_keep = refined
        damage_keep = (raw_damage_prob_np >= args.damage_thresh_export) & (opacity_prob_np >= args.damage_opacity_min)
        if args.gate_damage_by_column:
            damage_keep &= column_keep

        column_prob_np = np.where(column_keep, raw_column_prob_np, 0.0).astype(np.float32)
        damage_prob_np = np.where(damage_keep, raw_damage_prob_np, 0.0).astype(np.float32)
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.cache_dir / "column_prob_strict.npy", column_prob_np)
        np.save(args.cache_dir / "damage_prob_strict.npy", damage_prob_np)
        strict_stats = {
            "raw_column_kept": raw_column_keep,
            "after_component_keep": after_component_keep,
            "final_column_kept": int(column_keep.sum()),
            "final_damage_kept": int(damage_keep.sum()),
        }

    column_prob = torch.from_numpy(column_prob_np).to(device=device)
    damage_prob = torch.from_numpy(damage_prob_np).to(device=device)

    means_np = means.detach().cpu().numpy()
    lo = np.percentile(means_np, 5.0, axis=0)
    hi = np.percentile(means_np, 95.0, axis=0)
    robust_mask = np.all((means_np >= lo) & (means_np <= hi), axis=1)
    robust_points = means_np[robust_mask] if np.any(robust_mask) else means_np
    robust_min = robust_points.min(axis=0)
    robust_max = robust_points.max(axis=0)
    robust_center = robust_points.mean(axis=0)
    robust_radius = max(float(np.linalg.norm(robust_max - robust_min)) * 0.5, 1e-3)

    print(f"[info] loaded {means.shape[0]} gaussians from {args.gs_ply}")
    print(f"[info] column prob: {args.column_prob}")
    print(f"[info] damage prob: {args.damage_prob}")
    if args.strict_filter:
        print(f"[info] strict filter enabled; cache dir: {args.cache_dir}")
        print(f"[info] strict stats: {json.dumps(strict_stats)}")
    print(f"[info] sh_degree={sh_degree}")
    print(f"[info] device={device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    state = {"mode": "base"}

    @torch.no_grad()
    def tensors_for_current_mode():
        column_thresh = float(column_thresh_slider.value)
        damage_thresh = float(damage_thresh_slider.value)
        alpha_scale = float(alpha_slider.value)
        alpha_floor = float(alpha_floor_slider.value)
        boost = float(opacity_boost_slider.value)
        transparency = float(transparency_slider.value)

        if state["mode"] == "base":
            return colors, opacities

        if state["mode"] == "column":
            denom = max(1.0 - column_thresh, 1e-6)
            score = ((column_prob - column_thresh) / denom).clamp(0.0, 1.0)
            return (
                make_highlight_colors(colors, score, (0.05, 1.0, 0.10), alpha_scale, alpha_floor),
                make_highlight_opacity(opacities, score, alpha_scale, alpha_floor, transparency, boost),
            )

        if state["mode"] == "damage":
            gated_damage = damage_prob * (column_prob >= column_thresh).float()
            denom = max(1.0 - damage_thresh, 1e-6)
            score = ((gated_damage - damage_thresh) / denom).clamp(0.0, 1.0)
            if not torch.any(score > 0):
                k = min(max(int(min_damage_count_slider.value), 1), int(gated_damage.numel()))
                topk_idx = torch.topk(gated_damage, k=k).indices
                score = torch.zeros_like(gated_damage)
                score[topk_idx] = 1.0
            return (
                make_highlight_colors(colors, score, (1.0, 0.10, 0.05), alpha_scale, alpha_floor),
                make_highlight_opacity(opacities, score, alpha_scale, alpha_floor, transparency, boost),
            )

        return colors, opacities

    @torch.no_grad()
    def render_raw(camera_state: CameraState, width: int, height: int):
        dyn_colors, dyn_opacities = tensors_for_current_mode()
        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        K = torch.from_numpy(camera_state.get_K((width, height))).float().to(device)
        viewmat = c2w.inverse()
        render_colors, render_alphas, info = rasterization(
            means,
            quats,
            scales,
            torch.sigmoid(dyn_opacities),
            dyn_colors,
            viewmat[None],
            K[None],
            width,
            height,
            sh_degree=sh_degree,
            near_plane=1e-2,
            far_plane=1e2,
            radius_clip=0.0,
            eps2d=0.3,
            backgrounds=torch.zeros((1, 3), device=device),
            render_mode="RGB+D",
            rasterize_mode="classic",
            camera_model="pinhole",
            packed=False,
            with_ut=args.with_ut,
            with_eval3d=args.with_eval3d,
        )
        return render_colors, render_alphas, info, K.cpu().numpy(), viewmat.cpu().numpy(), c2w.cpu().numpy()

    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        width = render_tab_state.render_width if render_tab_state.preview_render else render_tab_state.viewer_width
        height = render_tab_state.render_height if render_tab_state.preview_render else render_tab_state.viewer_height

        dyn_colors, dyn_opacities = tensors_for_current_mode()
        render_mode_map = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }
        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        K = torch.from_numpy(camera_state.get_K((width, height))).float().to(device)
        viewmat = c2w.inverse()
        render_colors, render_alphas, info = rasterization(
            means,
            quats,
            scales,
            torch.sigmoid(dyn_opacities),
            dyn_colors,
            viewmat[None],
            K[None],
            width,
            height,
            sh_degree=min(render_tab_state.max_sh_degree, sh_degree) if sh_degree is not None else None,
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=device) / 255.0,
            render_mode=render_mode_map[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            packed=False,
            with_ut=args.with_ut,
            with_eval3d=args.with_eval3d,
        )
        render_tab_state.total_gs_count = len(means)
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            return render_colors[0, ..., 0:3].clamp(0, 1).cpu().numpy()
        if render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            return apply_float_colormap(depth_norm, render_tab_state.colormap).cpu().numpy()
        alpha = render_alphas[0, ..., 0:1]
        return apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()

    server = viser.ViserServer(port=args.port, verbose=False)
    viewer = GsplatViewer(server=server, render_fn=viewer_render_fn, output_dir=args.output_dir, mode="rendering")
    viewer.render_tab_state.backgrounds = tuple(float(v) * 255.0 for v in args.background)

    def rerender_note() -> None:
        column_thresh = float(column_thresh_slider.value)
        damage_thresh = float(damage_thresh_slider.value)
        raw_damage_count = int((damage_prob >= damage_thresh).sum().item())
        gated_damage = damage_prob * (column_prob >= column_thresh).float()
        gated_damage_count = int((gated_damage >= damage_thresh).sum().item())
        status.content = (
            f"Mode: `{state['mode']}`\n\n"
            f"- strict filter: `{args.strict_filter}`\n"
            f"- column >= `{column_thresh:.3f}`: `{int((column_prob >= column_thresh).sum().item())}`\n"
            f"- raw damage >= `{damage_thresh:.3f}`: `{raw_damage_count}`\n"
            f"- gated damage >= `{damage_thresh:.3f}`: `{gated_damage_count}`\n"
            f"- min damage fallback: `{int(min_damage_count_slider.value)}`"
        )

    def request_rerender(event: viser.GuiEvent | None = None) -> None:
        rerender_note()
        viewer.rerender(event)

    def set_client_view(client: viser.ClientHandle, eye: np.ndarray, look_at: np.ndarray) -> None:
        with client.atomic():
            client.camera.look_at = look_at
            client.camera.position = eye
            client.camera.up_direction = np.array([0.0, -1.0, 0.0])

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        eye = robust_center + np.array([0.0, 0.0, 2.2 * robust_radius], dtype=np.float64)
        set_client_view(client, eye=eye, look_at=robust_center.astype(np.float64))

    base_btn = server.gui.add_button("Show Base", color="gray")
    column_btn = server.gui.add_button("Highlight Column", color="green")
    damage_btn = server.gui.add_button("Highlight Damage", color="red")
    focus_btn = server.gui.add_button("Focus Robust Center", color="gray")

    column_thresh_slider = server.gui.add_slider(
        "Column Thresh",
        min=0.0,
        max=1.0,
        step=0.01,
        initial_value=float(args.column_thresh_export),
    )
    damage_thresh_slider = server.gui.add_slider(
        "Damage Thresh",
        min=0.0,
        max=1.0,
        step=0.001,
        initial_value=float(args.damage_thresh_export),
    )
    min_damage_count_slider = server.gui.add_slider(
        "Min Damage Count",
        min=100,
        max=20000,
        step=100,
        initial_value=int(args.min_damage_count),
    )
    alpha_slider = server.gui.add_slider("Highlight Alpha", min=0.0, max=1.0, step=0.01, initial_value=float(args.highlight_alpha))
    alpha_floor_slider = server.gui.add_slider("Highlight Floor", min=0.0, max=1.0, step=0.01, initial_value=float(args.highlight_floor))
    transparency_slider = server.gui.add_slider("Transparency", min=0.0, max=1.0, step=0.01, initial_value=float(args.highlight_transparency))
    opacity_boost_slider = server.gui.add_slider("Opacity Boost", min=0.0, max=1.0, step=0.01, initial_value=float(args.highlight_opacity_boost))

    save_dir_text = server.gui.add_text("Save Dir", initial_value=str((ROOT / "output/column/real_gs_saved").resolve()))
    save_name_text = server.gui.add_text("Save Name", initial_value="view_0000")
    save_button = server.gui.add_button("Save Current View", color="orange")
    status = server.gui.add_markdown("")
    rerender_note()

    def next_name(current: str) -> str:
        stem = current.strip() or "view"
        if "_" in stem and stem.rsplit("_", 1)[1].isdigit():
            base, idx = stem.rsplit("_", 1)
            return f"{base}_{int(idx) + 1:04d}"
        return f"{stem}_0001"

    def depth_preview(depth: np.ndarray) -> np.ndarray:
        valid = np.isfinite(depth) & (depth > 0)
        out = np.zeros(depth.shape, dtype=np.uint8)
        if not np.any(valid):
            return out
        vals = depth[valid]
        lo = float(np.percentile(vals, 2))
        hi = float(np.percentile(vals, 98))
        hi = max(hi, lo + 1e-6)
        norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        return (norm * 255.0).astype(np.uint8)

    @base_btn.on_click
    def _(event: viser.GuiEvent) -> None:
        state["mode"] = "base"
        request_rerender(event)

    @column_btn.on_click
    def _(event: viser.GuiEvent) -> None:
        state["mode"] = "column"
        request_rerender(event)

    @damage_btn.on_click
    def _(event: viser.GuiEvent) -> None:
        state["mode"] = "damage"
        request_rerender(event)

    @focus_btn.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None
        eye = robust_center + np.array([0.0, 0.0, 2.2 * robust_radius], dtype=np.float64)
        set_client_view(event.client, eye=eye, look_at=robust_center.astype(np.float64))

    @column_thresh_slider.on_update
    @damage_thresh_slider.on_update
    @min_damage_count_slider.on_update
    @alpha_slider.on_update
    @alpha_floor_slider.on_update
    @transparency_slider.on_update
    @opacity_boost_slider.on_update
    def _(event: viser.GuiEvent) -> None:
        request_rerender(event)

    @save_button.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None
        out_dir = Path(save_dir_text.value).expanduser()
        if not out_dir.is_absolute():
            out_dir = (ROOT / out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = save_name_text.value.strip() or "view_0000"
        camera_state = viewer.get_camera_state(event.client)
        width = viewer.render_tab_state.viewer_width
        height = viewer.render_tab_state.viewer_height
        render_colors, _, _, K_np, w2c_np, c2w_np = render_raw(camera_state, width, height)
        rgb = render_colors[0, ..., :3].clamp(0, 1).cpu().numpy()
        depth = render_colors[0, ..., 3].cpu().numpy().astype(np.float32)

        suffix = state["mode"]
        rgb_path = out_dir / f"{stem}_{suffix}.png"
        depth_png_path = out_dir / f"{stem}_{suffix}_depth.png"
        depth_npy_path = out_dir / f"{stem}_{suffix}_depth.npy"
        meta_path = out_dir / f"{stem}_{suffix}_camera.json"

        Image.fromarray((rgb * 255.0).astype(np.uint8)).save(rgb_path)
        Image.fromarray(depth_preview(depth)).save(depth_png_path)
        np.save(depth_npy_path, depth)

        meta = {
            "name": stem,
            "mode": suffix,
            "gs_ply": str(args.gs_ply.resolve()),
            "column_prob": str(args.column_prob.resolve()),
            "damage_prob": str(args.damage_prob.resolve()),
            "image_size": {"width": int(width), "height": int(height)},
            "fov_rad": float(camera_state.fov),
            "intrinsics": K_np.tolist(),
            "w2c_extrinsics": w2c_np.tolist(),
            "c2w": c2w_np.tolist(),
            "camera_center_xyz": c2w_np[:3, 3].tolist(),
            "thresholds": {
                "column": float(column_thresh_slider.value),
                "damage": float(damage_thresh_slider.value),
                "min_damage_count": int(min_damage_count_slider.value),
            },
            "files": {
                "rendered_rgb": str(rgb_path.resolve()),
                "rendered_depth_npy": str(depth_npy_path.resolve()),
                "rendered_depth_preview": str(depth_png_path.resolve()),
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        status.content = (
            f"Saved `{stem}` in mode `{suffix}` to `{out_dir}`\n\n"
            f"- RGB: `{rgb_path.name}`\n"
            f"- Depth: `{depth_npy_path.name}`\n"
            f"- Camera: `{meta_path.name}`"
        )
        save_name_text.value = next_name(stem)

    print(f"[done] trained viewer running at http://127.0.0.1:{args.port}")
    print("Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    main()
