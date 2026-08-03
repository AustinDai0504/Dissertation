#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""把导航路径可视化在 graph 上。

中文说明：
已有的 `05_visualize.py` 是把导航路径叠加在视网膜原图/mask 上。
本脚本专门生成“graph 视角”的导航图：
1. 灰色点：血管骨架 skeleton；
2. 黄色线：branch graph edges；
3. 蓝色点：junction nodes；
4. 绿色点：endpoint nodes；
5. 红色粗线：planned path，也就是导航路径。

输出：
  outputs/06_graph_path_visualization/graph_path_overlays/<case>_sa_unet_path_on_graph.png
  outputs/06_graph_path_visualization/graph_path_visualization_summary.csv
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "outputs" / "_matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ensure_dir, load_binary, load_config


def save_path_on_graph(
    skeleton,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    path_df: pd.DataFrame,
    save_path: Path,
    title: str,
) -> None:
    """保存“导航路径叠加在 graph 上”的可视化图。"""
    ensure_dir(save_path.parent)
    h, w = skeleton.shape

    fig, ax = plt.subplots(figsize=(6, 6), dpi=220)
    ax.set_facecolor("#0b0f14")

    # 1. skeleton：用细小灰白点显示完整血管中心线。
    sy, sx = np.where(skeleton)
    if len(sx) > 0:
        step = max(1, len(sx) // 7000)
        ax.scatter(sx[::step], sy[::step], s=0.8, c="#d8dde6", alpha=0.55, linewidths=0)

    # 2. branch graph edges：用节点之间的直线表示抽象 graph 边。
    node_lookup = {
        int(row.node_id): (float(row.x_px), float(row.y_px), str(row.type))
        for row in nodes_df.itertuples(index=False)
    } if not nodes_df.empty else {}

    if not edges_df.empty and node_lookup:
        for row in edges_df.itertuples(index=False):
            source = node_lookup.get(int(row.source))
            target = node_lookup.get(int(row.target))
            if source is None or target is None:
                continue
            ax.plot(
                [source[0], target[0]],
                [source[1], target[1]],
                color="#f6c343",
                linewidth=0.65,
                alpha=0.58,
                zorder=2,
            )

    # 3. planned path：使用逐像素路径，红色粗线高亮导航路线。
    if not path_df.empty:
        ax.plot(
            path_df["x_px"],
            path_df["y_px"],
            color="#ff2f2f",
            linewidth=2.4,
            alpha=0.96,
            zorder=5,
        )
        # 沿路径放少量方向点，帮助看出 path 不是普通 graph 边。
        idx = np.linspace(0, len(path_df) - 1, min(14, len(path_df))).astype(int)
        ax.scatter(
            path_df["x_px"].iloc[idx],
            path_df["y_px"].iloc[idx],
            s=14,
            c="#ffef5a",
            edgecolors="#111111",
            linewidths=0.35,
            zorder=6,
        )
        ax.scatter(path_df["x_px"].iloc[0], path_df["y_px"].iloc[0], s=80, c="#35e36f", edgecolors="black", zorder=7)
        ax.scatter(path_df["x_px"].iloc[-1], path_df["y_px"].iloc[-1], s=80, c="#ffdf3d", edgecolors="black", zorder=7)
        ax.text(path_df["x_px"].iloc[0] + 5, path_df["y_px"].iloc[0] + 5, "S", color="#35e36f", fontsize=10, weight="bold")
        ax.text(path_df["x_px"].iloc[-1] + 5, path_df["y_px"].iloc[-1] + 5, "T", color="#ffdf3d", fontsize=10, weight="bold")

    # 4. graph nodes：junction 和 endpoint 分色显示。
    if not nodes_df.empty:
        endpoints = nodes_df[nodes_df["type"] == "endpoint"]
        junctions = nodes_df[nodes_df["type"] == "junction"]
        if not junctions.empty:
            ax.scatter(junctions["x_px"], junctions["y_px"], s=18, c="#4d8dff", edgecolors="white", linewidths=0.25, zorder=4)
        if not endpoints.empty:
            ax.scatter(endpoints["x_px"], endpoints["y_px"], s=16, c="#42f57b", edgecolors="black", linewidths=0.25, zorder=4)

    ax.set_title(title, fontsize=10, color="#f2f4f8")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    out_dir = ensure_dir(Path(cfg["output_dir"]) / "06_graph_path_visualization")
    overlay_dir = ensure_dir(out_dir / "graph_path_overlays")
    graph_dir = Path(cfg["output_dir"]) / "04_graph_navigation" / "graphs"
    path_dir = Path(cfg["output_dir"]) / "04_graph_navigation" / "paths"
    skeleton_dir = Path(cfg["output_dir"]) / "04_graph_navigation" / "skeletons"
    rows = []

    for case_id in map(str, cfg["case_ids"]):
        skeleton_path = skeleton_dir / f"{case_id}_sa_unet_skeleton.png"
        nodes_path = graph_dir / f"graph_nodes_sa_unet_{case_id}.csv"
        edges_path = graph_dir / f"graph_edges_sa_unet_{case_id}.csv"
        path_csv = path_dir / f"planned_path_sa_unet_{case_id}.csv"

        if not (skeleton_path.exists() and nodes_path.exists() and edges_path.exists() and path_csv.exists()):
            rows.append({
                "case_id": case_id,
                "status": "missing_input",
                "output_path": "",
            })
            continue

        skeleton = load_binary(skeleton_path)
        nodes_df = pd.read_csv(nodes_path)
        edges_df = pd.read_csv(edges_path)
        path_df = pd.read_csv(path_csv)
        save_path = overlay_dir / f"{case_id}_sa_unet_path_on_graph.png"
        save_path_on_graph(
            skeleton=skeleton,
            nodes_df=nodes_df,
            edges_df=edges_df,
            path_df=path_df,
            save_path=save_path,
            title=f"{case_id} SA-UNet planned path on vascular graph",
        )
        rows.append({
            "case_id": case_id,
            "status": "saved",
            "node_count": int(len(nodes_df)),
            "edge_count": int(len(edges_df)),
            "path_pixel_count": int(len(path_df)),
            "output_path": str(save_path),
        })

    summary_path = out_dir / "graph_path_visualization_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"graph 上的路径可视化完成: {summary_path}")


if __name__ == "__main__":
    main()
