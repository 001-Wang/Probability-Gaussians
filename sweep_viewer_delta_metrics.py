#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def parse_floats(spec: str) -> list[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Fine-tune viewer delta extraction thresholds around a good baseline.")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument(
        "--eval-script",
        type=Path,
        default=Path("output/column/eval_test_view_metrics_viewer.py"),
    )
    p.add_argument("--gs-ply", type=Path, default=Path("output/column/gs_ply_clean/0000_damage_opt_clean.ply"))
    p.add_argument("--column-prob", type=Path, default=Path("output/column/gs_ply_clean/0000_column_prob_clean.npy"))
    p.add_argument("--damage-prob", type=Path, default=Path("output/column/gs_ply_clean/0000_damage_prob_clean.npy"))
    p.add_argument("--view-root", type=Path, default=Path("output/column/real_gs_saved_test"))
    p.add_argument("--gt-root", type=Path, default=Path("assets/examples/column/real_gs_saved_test_converted"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--column-thresh", type=float, default=0.90)
    p.add_argument("--damage-thresh", type=float, default=0.15)
    p.add_argument("--min-damage-count", type=int, default=3000)

    p.add_argument("--delta-r-min-list", type=str, default="0.16,0.18,0.20,0.22,0.24")
    p.add_argument("--delta-rg-min-list", type=str, default="0.11,0.13,0.15,0.17,0.19")
    p.add_argument("--delta-rb-min-list", type=str, default="0.11,0.13,0.15,0.17,0.19")
    p.add_argument("--highlight-red-min-list", type=str, default="0.30,0.33,0.35,0.37,0.40")

    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=Path("output/column/test_metrics_viewer_sweep"))
    return p.parse_args()


def score_run(summary: dict) -> tuple[float, float, float]:
    dmg = summary["damage"]["overall"]
    return (float(dmg["iou"]), float(dmg["f1"]), float(dmg["precision"]))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    delta_r_list = parse_floats(args.delta_r_min_list)
    delta_rg_list = parse_floats(args.delta_rg_min_list)
    delta_rb_list = parse_floats(args.delta_rb_min_list)
    highlight_red_list = parse_floats(args.highlight_red_min_list)

    rows: list[dict] = []
    run_idx = 0

    for delta_r, delta_rg, delta_rb, red_min in itertools.product(
        delta_r_list,
        delta_rg_list,
        delta_rb_list,
        highlight_red_list,
    ):
        run_dir = args.out_dir / (
            f"run_{run_idx:04d}"
            f"_dr{delta_r:.3f}"
            f"_drg{delta_rg:.3f}"
            f"_drb{delta_rb:.3f}"
            f"_rmin{red_min:.3f}"
        )
        cmd = [
            args.python,
            str(args.eval_script),
            "--gs-ply", str(args.gs_ply),
            "--column-prob", str(args.column_prob),
            "--damage-prob", str(args.damage_prob),
            "--view-root", str(args.view_root),
            "--gt-root", str(args.gt_root),
            "--column-thresh", str(args.column_thresh),
            "--damage-thresh", str(args.damage_thresh),
            "--min-damage-count", str(args.min_damage_count),
            "--delta-r-min", str(delta_r),
            "--delta-rg-min", str(delta_rg),
            "--delta-rb-min", str(delta_rb),
            "--highlight-red-min", str(red_min),
            "--device", args.device,
            "--out-dir", str(run_dir),
        ]

        print(
            f"[run {run_idx + 1}] "
            f"delta_r={delta_r:.3f} delta_rg={delta_rg:.3f} "
            f"delta_rb={delta_rb:.3f} red_min={red_min:.3f}"
        )
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            raise RuntimeError(
                f"sweep failed for delta_r={delta_r}, delta_rg={delta_rg}, "
                f"delta_rb={delta_rb}, red_min={red_min}"
            )

        summary = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
        overall = summary["damage"]["overall"]
        row = {
            "run_dir": str(run_dir),
            "delta_r_min": delta_r,
            "delta_rg_min": delta_rg,
            "delta_rb_min": delta_rb,
            "highlight_red_min": red_min,
            "damage_overall": overall,
        }
        rows.append(row)
        run_idx += 1

    rows_sorted = sorted(rows, key=lambda x: score_run({"damage": {"overall": x["damage_overall"]}}), reverse=True)
    best_rows = rows_sorted[: max(args.top_k, 1)]

    summary = {
        "config": {
            "column_thresh": args.column_thresh,
            "damage_thresh": args.damage_thresh,
            "min_damage_count": args.min_damage_count,
            "delta_r_min_list": delta_r_list,
            "delta_rg_min_list": delta_rg_list,
            "delta_rb_min_list": delta_rb_list,
            "highlight_red_min_list": highlight_red_list,
            "top_k": args.top_k,
        },
        "best_runs": best_rows,
        "all_runs": rows_sorted,
    }
    out_json = args.out_dir / "sweep_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[best runs]")
    for idx, row in enumerate(best_rows, start=1):
        dmg = row["damage_overall"]
        print(
            f"{idx}. "
            f"delta_r={row['delta_r_min']:.3f} "
            f"delta_rg={row['delta_rg_min']:.3f} "
            f"delta_rb={row['delta_rb_min']:.3f} "
            f"red_min={row['highlight_red_min']:.3f} | "
            f"IoU={dmg['iou']:.4f} P={dmg['precision']:.4f} "
            f"R={dmg['recall']:.4f} F1={dmg['f1']:.4f}"
        )
    print(f"[done] saved sweep summary to {out_json}")


if __name__ == "__main__":
    main()
