#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
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
    parser = argparse.ArgumentParser("View a real Gaussian splat PLY in the gsplat browser viewer.")
    parser.add_argument("--gs-ply", type=Path, default=Path("output/column/gs_ply/0000.ply"))
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("output/column/real_gs_viewer"))
    parser.add_argument("--background", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--with-ut", action="store_true")
    parser.add_argument("--with-eval3d", action="store_true")
    return parser.parse_args()


def load_ply_as_gsplat_tensors(path: Path, device: torch.device):
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    names = set(v.data.dtype.names or [])

    means = torch.tensor(
        np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32),
        device=device,
    )
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

    f_rest_names = sorted([name for name in names if name.startswith("f_rest_")], key=lambda x: int(x.split("_")[-1]))
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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA device is required for the real gsplat viewer.")

    means, quats, scales, opacities, colors, sh_degree = load_ply_as_gsplat_tensors(args.gs_ply, device)
    means_np = means.detach().cpu().numpy()
    scene_min = means_np.min(axis=0)
    scene_max = means_np.max(axis=0)
    scene_center = 0.5 * (scene_min + scene_max)
    scene_radius = max(float(np.linalg.norm(scene_max - scene_min)) * 0.5, 1e-3)
    lo = np.percentile(means_np, 5.0, axis=0)
    hi = np.percentile(means_np, 95.0, axis=0)
    robust_mask = np.all((means_np >= lo) & (means_np <= hi), axis=1)
    robust_points = means_np[robust_mask] if np.any(robust_mask) else means_np
    robust_min = robust_points.min(axis=0)
    robust_max = robust_points.max(axis=0)
    robust_center = robust_points.mean(axis=0)
    robust_radius = max(float(np.linalg.norm(robust_max - robust_min)) * 0.5, 1e-3)
    print(f"[info] loaded {means.shape[0]} gaussians from {args.gs_ply}")
    print(f"[info] sh_degree={sh_degree}")
    print(f"[info] device={device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def render_raw(camera_state: CameraState, width: int, height: int):
        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        K = torch.from_numpy(camera_state.get_K((width, height))).float().to(device)
        viewmat = c2w.inverse()
        render_colors, render_alphas, info = rasterization(
            means,
            quats,
            scales,
            torch.sigmoid(opacities),
            colors,
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
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height

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
            torch.sigmoid(opacities),
            colors,
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
    viewer = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=args.output_dir,
        mode="rendering",
    )
    viewer.render_tab_state.backgrounds = tuple(float(v) * 255.0 for v in args.background)

    def set_client_view(client: viser.ClientHandle, eye: np.ndarray, look_at: np.ndarray) -> None:
        with client.atomic():
            client.camera.look_at = look_at
            client.camera.position = eye
            client.camera.up_direction = np.array([0.0, -1.0, 0.0])

    def refocus_client(client: viser.ClientHandle, target: np.ndarray) -> None:
        current_pos = np.asarray(client.camera.position, dtype=np.float64)
        current_look = np.asarray(client.camera.look_at, dtype=np.float64)
        offset = current_pos - current_look
        if np.linalg.norm(offset) < 1e-6:
            offset = np.array([0.0, 0.0, 2.2 * robust_radius], dtype=np.float64)
        set_client_view(client, eye=target.astype(np.float64) + offset, look_at=target.astype(np.float64))

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        eye = robust_center + np.array([0.0, 0.0, 2.2 * robust_radius], dtype=np.float64)
        set_client_view(client, eye=eye, look_at=robust_center.astype(np.float64))

    focus_center_btn = server.gui.add_button("Focus Robust Center", color="gray")
    front_view_btn = server.gui.add_button("Front View", color="gray")
    side_view_btn = server.gui.add_button("Side View", color="gray")
    top_view_btn = server.gui.add_button("Top View", color="gray")

    save_dir_text = server.gui.add_text("Save Dir", initial_value=str((ROOT / "output/column/real_gs_saved").resolve()))
    save_name_text = server.gui.add_text("Save Name", initial_value="view_0000")
    save_button = server.gui.add_button("Save Current View", color="orange")
    save_status = server.gui.add_markdown("Ready.")

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

    @focus_center_btn.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None
        refocus_client(event.client, robust_center)

    @front_view_btn.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None
        eye = robust_center + np.array([0.0, 0.0, 2.2 * robust_radius], dtype=np.float64)
        set_client_view(event.client, eye=eye, look_at=robust_center.astype(np.float64))

    @side_view_btn.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None
        eye = robust_center + np.array([2.2 * robust_radius, 0.0, 0.0], dtype=np.float64)
        set_client_view(event.client, eye=eye, look_at=robust_center.astype(np.float64))

    @top_view_btn.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None
        eye = robust_center + np.array([0.0, -2.2 * robust_radius, 0.0], dtype=np.float64)
        set_client_view(event.client, eye=eye, look_at=robust_center.astype(np.float64))

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

        rgb_path = out_dir / f"{stem}.png"
        depth_png_path = out_dir / f"{stem}_depth.png"
        depth_npy_path = out_dir / f"{stem}_depth.npy"
        meta_path = out_dir / f"{stem}_camera.json"

        Image.fromarray((rgb * 255.0).astype(np.uint8)).save(rgb_path)
        Image.fromarray(depth_preview(depth)).save(depth_png_path)
        np.save(depth_npy_path, depth)

        meta = {
            "name": stem,
            "gs_ply": str(args.gs_ply.resolve()),
            "image_size": {"width": int(width), "height": int(height)},
            "fov_rad": float(camera_state.fov),
            "intrinsics": K_np.tolist(),
            "w2c_extrinsics": w2c_np.tolist(),
            "c2w": c2w_np.tolist(),
            "camera_center_xyz": c2w_np[:3, 3].tolist(),
            "files": {
                "rendered_rgb": str(rgb_path.resolve()),
                "rendered_depth_npy": str(depth_npy_path.resolve()),
                "rendered_depth_preview": str(depth_png_path.resolve()),
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        save_status.content = (
            f"Saved `{stem}` to `{out_dir}`\n\n"
            f"- RGB: `{rgb_path.name}`\n"
            f"- Depth: `{depth_npy_path.name}`\n"
            f"- Camera: `{meta_path.name}`"
        )
        save_name_text.value = next_name(stem)

    print(f"[done] viewer running at http://127.0.0.1:{args.port}")
    print("Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    main()
