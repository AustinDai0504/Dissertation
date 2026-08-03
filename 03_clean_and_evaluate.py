#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""清理 SA-UNet 预测结果并计算分割指标。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import clean_mask, drive_case_paths, ensure_dir, estimate_fov_mask, load_binary, load_config, load_rgb, save_binary, segmentation_metrics


def main() -> None:
    cfg = load_config()
    out_dir = ensure_dir(Path(cfg["output_dir"]) / "03_clean_evaluate")
    clean_dir = ensure_dir(out_dir / "masks_cleaned")
    raw_dir = Path(cfg["output_dir"]) / "02_predict" / "masks"
    min_obj = int(cfg.get("preprocessing", {}).get("min_object_size", 30))
    rows = []

    for case_id in map(str, cfg["case_ids"]):
        image_path, manual_path, fov_path = drive_case_paths(cfg["drive_root"], cfg["split"], case_id)
        raw_path = raw_dir / f"{case_id}_sa_unet_mask.png"
        if not raw_path.exists():
            rows.append({"case_id": case_id, "status": "missing_prediction"})
            continue
        rgb = load_rgb(image_path)
        fov = load_binary(fov_path) if fov_path is not None else estimate_fov_mask(rgb)
        raw = load_binary(raw_path)
        cleaned = clean_mask(raw, fov, min_object_size=min_obj)
        clean_path = clean_dir / f"{case_id}_sa_unet_cleaned_mask.png"
        save_binary(cleaned, clean_path)
        row = {
            "case_id": case_id,
            "status": "cleaned",
            "raw_pixels": int(raw.sum()),
            "cleaned_pixels": int(cleaned.sum()),
            "cleaned_path": str(clean_path),
        }
        if manual_path is not None:
            row.update(segmentation_metrics(cleaned, load_binary(manual_path), fov))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "segmentation_metrics.csv", index=False)
    df.to_excel(out_dir / "segmentation_metrics.xlsx", index=False)
    if "dice" in df.columns:
        df[["dice", "iou", "precision", "recall", "specificity", "pred_pixels"]].mean(numeric_only=True).to_frame("mean").to_csv(out_dir / "segmentation_mean.csv")
    print(f"清理和评估完成: {out_dir / 'segmentation_metrics.csv'}")


if __name__ == "__main__":
    main()
