#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""从 SA-UNet cleaned mask 生成 graph 并规划导航路径。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import ensure_dir, graph_path_metrics, load_binary, load_config, save_binary


def main() -> None:
    cfg = load_config()
    out_dir = ensure_dir(Path(cfg["output_dir"]) / "04_graph_navigation")
    graph_dir = ensure_dir(out_dir / "graphs")
    path_dir = ensure_dir(out_dir / "paths")
    skel_dir = ensure_dir(out_dir / "skeletons")
    mask_dir = Path(cfg["output_dir"]) / "03_clean_evaluate" / "masks_cleaned"
    min_obj = int(cfg.get("graph", {}).get("min_object_size", 30))
    rows = []

    for case_id in map(str, cfg["case_ids"]):
        mask_path = mask_dir / f"{case_id}_sa_unet_cleaned_mask.png"
        if not mask_path.exists():
            rows.append({"case_id": case_id, "status": "missing_cleaned_mask"})
            continue
        mask = load_binary(mask_path)
        metrics, skel, nodes, edges, path_pixels = graph_path_metrics(mask, min_object_size=min_obj)
        pd.DataFrame(nodes).to_csv(graph_dir / f"graph_nodes_sa_unet_{case_id}.csv", index=False)
        pd.DataFrame(edges).to_csv(graph_dir / f"graph_edges_sa_unet_{case_id}.csv", index=False)
        pd.DataFrame([
            {"order": i, "x_px": int(p[1]), "y_px": int(p[0])}
            for i, p in enumerate(path_pixels)
        ], columns=["order", "x_px", "y_px"]).to_csv(path_dir / f"planned_path_sa_unet_{case_id}.csv", index=False)
        save_binary(skel, skel_dir / f"{case_id}_sa_unet_skeleton.png")
        rows.append({"case_id": case_id, "status": "built", **metrics})

    summary_path = out_dir / "graph_navigation_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"图和导航路径完成: {summary_path}")


if __name__ == "__main__":
    main()
