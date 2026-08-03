#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""共享工具：路径、图像、指标、骨架、建图和路径规划。"""

from __future__ import annotations

from pathlib import Path
import random

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk, remove_small_objects, skeletonize


ROOT = Path(__file__).resolve().parent


def resolve_path(path: str | Path) -> Path:
    """把配置中的相对路径解析到当前 workflow 文件夹。"""
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_config(path: str | Path = "config.yaml") -> dict:
    """读取 YAML 配置，并转换关键路径。"""
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["drive_root"] = resolve_path(cfg["drive_root"])
    cfg["output_dir"] = resolve_path(cfg["output_dir"])
    cfg["sa_unet"]["weights_path"] = resolve_path(cfg["sa_unet"]["weights_path"])
    return cfg


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 42) -> None:
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)


def drive_case_paths(drive_root: str | Path, split: str, case_id: str):
    """根据 DRIVE 命名规则找到原图、人工 mask 和 FOV mask。"""
    root = Path(drive_root)
    split = split.lower().strip()
    suffix = "training" if split == "training" else "test"
    image_path = root / split / "images" / f"{case_id}_{suffix}.tif"
    manual_path = root / split / "1st_manual" / f"{case_id}_manual1.gif"
    fov_path = root / split / "mask" / f"{case_id}_{suffix}_mask.gif"
    if not image_path.exists():
        raise FileNotFoundError(f"找不到 DRIVE 图像: {image_path}")
    return image_path, manual_path if manual_path.exists() else None, fov_path if fov_path.exists() else None


def load_rgb(path: str | Path) -> np.ndarray:
    """读取 RGB 图像。"""
    return np.asarray(Image.open(path).convert("RGB"))


def load_gray(path: str | Path) -> np.ndarray:
    """读取灰度图像。"""
    return np.asarray(Image.open(path).convert("L"))


def load_binary(path: str | Path, threshold: int = 0) -> np.ndarray:
    """读取二值 mask。"""
    return load_gray(path) > threshold


def save_binary(mask: np.ndarray, path: str | Path) -> None:
    """保存二值 mask。"""
    ensure_dir(Path(path).parent)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def save_probability(prob: np.ndarray, path: str | Path) -> None:
    """保存概率图，便于检查阈值是否合适。"""
    ensure_dir(Path(path).parent)
    Image.fromarray(np.clip(prob * 255, 0, 255).astype(np.uint8)).save(path)


def center_pad_to_square(arr: np.ndarray, size: int, fill_value: float = 0.0):
    """按原 SA-UNet 训练脚本方式，四周居中补黑到 592x592。"""
    h, w = arr.shape[:2]
    delta_h = size - h
    delta_w = size - w
    if delta_h < 0 or delta_w < 0:
        raise ValueError(f"输入尺寸 {arr.shape[:2]} 大于目标尺寸 {size}")
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left
    if arr.ndim == 2:
        padded = np.pad(arr, ((top, bottom), (left, right)), mode="constant", constant_values=fill_value)
    else:
        padded = np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode="constant", constant_values=fill_value)
    return padded, (top, bottom, left, right)


def remove_center_padding(arr: np.ndarray, pads: tuple[int, int, int, int], original_shape: tuple[int, int]) -> np.ndarray:
    """把 592x592 输出裁回原图尺寸。"""
    top, bottom, left, right = pads
    h, w = original_shape
    return arr[top:top + h, left:left + w]


def estimate_fov_mask(rgb: np.ndarray) -> np.ndarray:
    """没有 FOV 文件时，从非黑区域估计 FOV。"""
    gray = rgb2gray(rgb)
    mask = gray > 0.04
    mask = ndi.binary_fill_holes(mask)
    mask = binary_closing(mask, disk(8))
    lab = label(mask.astype(np.uint8), connectivity=2)
    if lab.max() == 0:
        return np.ones(gray.shape, dtype=bool)
    largest = max(regionprops(lab), key=lambda p: p.area)
    return lab == largest.label


def normalize_for_display(rgb: np.ndarray, fov_mask: np.ndarray | None = None) -> np.ndarray:
    """显示用归一化，不参与模型推理。"""
    out = rgb.astype(np.float32) / 255.0
    if fov_mask is None:
        fov_mask = np.ones(out.shape[:2], dtype=bool)
    for c in range(3):
        ch = out[..., c]
        vals = ch[fov_mask]
        if vals.size:
            lo, hi = np.percentile(vals, [1, 99])
            if hi > lo:
                ch = np.clip(ch, lo, hi)
                ch = (ch - lo) / (hi - lo)
        out[..., c] = ch
    out[~fov_mask] = 0
    return (out * 255).astype(np.uint8)


def clean_mask(mask: np.ndarray, fov_mask: np.ndarray | None, min_object_size: int = 30) -> np.ndarray:
    """清理预测 mask。"""
    out = mask.astype(bool)
    if fov_mask is not None:
        out &= fov_mask.astype(bool)
    out = remove_small_objects(out, min_size=min_object_size)
    out = binary_closing(out, disk(1))
    if fov_mask is not None:
        out &= fov_mask.astype(bool)
    return out


def segmentation_metrics(pred: np.ndarray, gt: np.ndarray, fov_mask: np.ndarray | None = None) -> dict:
    """计算 Dice、IoU、Precision、Recall 等指标。"""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if fov_mask is not None:
        pred &= fov_mask.astype(bool)
        gt &= fov_mask.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "dice": round(float(2 * tp / (2 * tp + fp + fn + 1e-8)), 6),
        "iou": round(float(tp / (tp + fp + fn + 1e-8)), 6),
        "precision": round(float(tp / (tp + fp + 1e-8)), 6),
        "recall": round(float(tp / (tp + fn + 1e-8)), 6),
        "specificity": round(float(tn / (tn + fp + 1e-8)), 6),
        "pred_pixels": int(pred.sum()),
        "gt_pixels": int(gt.sum()),
    }


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.35) -> np.ndarray:
    """把 mask 半透明叠加到原图。"""
    out = rgb.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    out[mask.astype(bool)] = (1 - alpha) * out[mask.astype(bool)] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def skeleton_from_mask(mask: np.ndarray, min_object_size: int = 30) -> np.ndarray:
    """血管 mask 骨架化。"""
    return skeletonize(remove_small_objects(mask.astype(bool), min_size=min_object_size)).astype(bool)


def make_pixel_graph(skel: np.ndarray) -> nx.Graph:
    """把骨架像素转成 8 邻域图。"""
    graph = nx.Graph()
    ys, xs = np.where(skel)
    pixels = set(zip(ys.tolist(), xs.tolist()))
    for y, x in pixels:
        graph.add_node((int(y), int(x)))
    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for y, x in pixels:
        for dy, dx in neigh:
            nb = (y + dy, x + dx)
            if nb in pixels:
                weight = float(np.sqrt(2.0)) if abs(dy) + abs(dx) == 2 else 1.0
                graph.add_edge((int(y), int(x)), (int(nb[0]), int(nb[1])), weight=weight)
    return graph


def neighbour_count(skel: np.ndarray) -> np.ndarray:
    """统计每个骨架像素的邻居数。"""
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    return ndi.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)


def extract_branch_graph(skel: np.ndarray):
    """从骨架提取端点、交叉点和血管段。"""
    skel = skel.astype(bool)
    deg = neighbour_count(skel)
    node_mask = (skel & (deg <= 1)) | (skel & (deg >= 3))
    labels = label(node_mask.astype(np.uint8), connectivity=2)
    nodes = []
    pixel_to_node = {}
    for region in regionprops(labels):
        coords = region.coords
        node_id = len(nodes)
        local_deg = deg[coords[:, 0], coords[:, 1]]
        node_type = "junction" if np.any(local_deg >= 3) else "endpoint"
        nodes.append({
            "node_id": node_id,
            "x_px": int(round(float(coords[:, 1].mean()))),
            "y_px": int(round(float(coords[:, 0].mean()))),
            "type": node_type,
            "pixel_count": int(region.area),
            "mean_degree": round(float(local_deg.mean()), 3),
        })
        for y, x in coords:
            pixel_to_node[(int(y), int(x))] = node_id
    chains = label((skel & ~node_mask).astype(np.uint8), connectivity=2)
    edges = []
    used = set()
    for region in regionprops(chains):
        coords = region.coords
        touching = set()
        for y, x in coords:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    node_id = pixel_to_node.get((int(y + dy), int(x + dx)))
                    if node_id is not None:
                        touching.add(node_id)
        touching = sorted(touching)
        for i in range(len(touching)):
            for j in range(i + 1, len(touching)):
                pair = tuple(sorted((touching[i], touching[j])))
                if pair in used:
                    continue
                used.add(pair)
                edges.append({
                    "edge_id": len(edges),
                    "source": int(pair[0]),
                    "target": int(pair[1]),
                    "length_px": round(float(len(coords)), 3),
                    "n_pixels": int(len(coords)),
                    "mean_x_px": round(float(coords[:, 1].mean()), 3),
                    "mean_y_px": round(float(coords[:, 0].mean()), 3),
                })
    return nodes, edges


def choose_representative_path(graph: nx.Graph):
    """选择一条代表性较长路径。"""
    if graph.number_of_nodes() < 2:
        return (0, 0), (0, 0), [], 0.0, False
    endpoints = [n for n, d in graph.degree() if d == 1]
    candidates = endpoints if len(endpoints) >= 2 else list(graph.nodes())
    try:
        dist0 = nx.single_source_dijkstra_path_length(graph, candidates[0], weight="weight")
        u = max(candidates, key=lambda n: dist0.get(n, -1))
        dist1 = nx.single_source_dijkstra_path_length(graph, u, weight="weight")
        v = max(candidates, key=lambda n: dist1.get(n, -1))
        path = nx.shortest_path(graph, u, v, weight="weight")
        return u, v, path, float(nx.path_weight(graph, path, weight="weight")), True
    except Exception:
        return (0, 0), (0, 0), [], 0.0, False


def graph_path_metrics(mask: np.ndarray, min_object_size: int = 30):
    """从 mask 输出骨架、图、路径和统计。"""
    skel = skeleton_from_mask(mask, min_object_size)
    pixel_graph = make_pixel_graph(skel)
    nodes, edges = extract_branch_graph(skel)
    start, target, path, path_len, success = choose_representative_path(pixel_graph)
    metrics = {
        "skeleton_pixels": int(skel.sum()),
        "pixel_graph_nodes": int(pixel_graph.number_of_nodes()),
        "pixel_graph_edges": int(pixel_graph.number_of_edges()),
        "branch_graph_nodes": int(len(nodes)),
        "branch_graph_edges": int(len(edges)),
        "endpoint_nodes": int(sum(1 for r in nodes if r["type"] == "endpoint")),
        "junction_nodes": int(sum(1 for r in nodes if r["type"] == "junction")),
        "path_success": bool(success),
        "path_length_px": round(float(path_len), 3),
        "path_pixel_count": int(len(path)),
        "start_x_px": int(start[1]) if success else "",
        "start_y_px": int(start[0]) if success else "",
        "target_x_px": int(target[1]) if success else "",
        "target_y_px": int(target[0]) if success else "",
    }
    return metrics, skel, nodes, edges, path


def main() -> None:
    """单独运行时输出环境检查。"""
    cfg = load_config()
    out_dir = ensure_dir(Path(cfg["output_dir"]) / "00_environment")
    pd.DataFrame([
        {"item": "drive_root", "value": str(cfg["drive_root"])},
        {"item": "weights_path", "value": str(cfg["sa_unet"]["weights_path"])},
        {"item": "weights_exists", "value": str(Path(cfg["sa_unet"]["weights_path"]).exists())},
        {"item": "case_ids", "value": ",".join(map(str, cfg["case_ids"]))},
    ]).to_csv(out_dir / "environment_summary.csv", index=False)
    print(f"环境检查完成: {out_dir / 'environment_summary.csv'}")


if __name__ == "__main__":
    main()
