#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""生成 SA-UNet 分割和导航路径可视化图。"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "outputs" / "_matplotlib_cache"))

import matplotlib.pyplot as plt
import pandas as pd

from common import drive_case_paths, ensure_dir, estimate_fov_mask, load_binary, load_config, load_rgb, normalize_for_display, overlay_mask


def save_mask_overlay(rgb, mask, path: Path, title: str) -> None:
    """保存 mask overlay。"""
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=180)
    ax.imshow(overlay_mask(rgb, mask))
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_path_overlay(rgb, mask, path_df: pd.DataFrame, path: Path, title: str) -> None:
    """保存导航路径 overlay。"""
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=180)
    ax.imshow(overlay_mask(rgb, mask, alpha=0.18))
    if not path_df.empty:
        ax.plot(path_df["x_px"], path_df["y_px"], color="red", linewidth=2.0)
        ax.scatter(path_df["x_px"].iloc[0], path_df["y_px"].iloc[0], s=60, c="lime", edgecolors="black")
        ax.scatter(path_df["x_px"].iloc[-1], path_df["y_px"].iloc[-1], s=60, c="yellow", edgecolors="black")
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    out_dir = ensure_dir(Path(cfg["output_dir"]) / "05_visualization")
    overlay_dir = ensure_dir(out_dir / "overlays")
    mask_dir = Path(cfg["output_dir"]) / "03_clean_evaluate" / "masks_cleaned"
    path_dir = Path(cfg["output_dir"]) / "04_graph_navigation" / "paths"
    rows = []

    for case_id in map(str, cfg["case_ids"]):
        image_path, _, fov_path = drive_case_paths(cfg["drive_root"], cfg["split"], case_id)
        rgb = load_rgb(image_path)
        fov = load_binary(fov_path) if fov_path is not None else estimate_fov_mask(rgb)
        rgb_disp = normalize_for_display(rgb, fov)
        mask_path = mask_dir / f"{case_id}_sa_unet_cleaned_mask.png"
        if not mask_path.exists():
            continue
        mask = load_binary(mask_path)
        mask_overlay = overlay_dir / f"{case_id}_sa_unet_mask_overlay.png"
        save_mask_overlay(rgb_disp, mask, mask_overlay, f"{case_id} SA-UNet mask")
        rows.append({"case_id": case_id, "type": "mask_overlay", "output_path": str(mask_overlay)})
        path_csv = path_dir / f"planned_path_sa_unet_{case_id}.csv"
        if path_csv.exists():
            path_df = pd.read_csv(path_csv)
            nav_overlay = overlay_dir / f"{case_id}_sa_unet_navigation_overlay.png"
            save_path_overlay(rgb_disp, mask, path_df, nav_overlay, f"{case_id} SA-UNet navigation")
            rows.append({"case_id": case_id, "type": "navigation_overlay", "output_path": str(nav_overlay)})

    summary_path = out_dir / "visualization_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"可视化完成: {summary_path}")


if __name__ == "__main__":
    main()
