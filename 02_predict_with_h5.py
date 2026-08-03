#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""使用已有 h5 权重预测 DRIVE 血管 mask。

输出：
  outputs/02_predict/probability_maps/<case>_sa_unet_prob.png
  outputs/02_predict/masks/<case>_sa_unet_mask.png
  outputs/02_predict/prediction_summary.csv
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    center_pad_to_square,
    drive_case_paths,
    ensure_dir,
    load_config,
    load_rgb,
    remove_center_padding,
    save_binary,
    save_probability,
    set_seed,
)

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "outputs" / "_matplotlib_cache"))


def load_model_helpers():
    """动态加载模型构建函数。"""
    script = Path(__file__).resolve().parent / "01_sa_unet_model.py"
    spec = importlib.util.spec_from_file_location("sa_unet_h5_model", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    cfg = load_config()
    set_seed(42)
    mcfg = cfg["sa_unet"]
    out_dir = ensure_dir(Path(cfg["output_dir"]) / "02_predict")
    prob_dir = ensure_dir(out_dir / "probability_maps")
    mask_dir = ensure_dir(out_dir / "masks")
    rows = []

    try:
        helpers = load_model_helpers()
        model = helpers.build_sa_unet(
            input_size=(int(mcfg["image_size"]), int(mcfg["image_size"]), 3),
            start_neurons=int(mcfg["start_neurons"]),
        )
        helpers.load_drive_weights(model, mcfg["weights_path"])
    except Exception as exc:
        rows.append({"case_id": "", "status": "model_load_failed", "error": repr(exc)})
        pd.DataFrame(rows).to_csv(out_dir / "prediction_summary.csv", index=False)
        print(f"预测失败，模型加载错误已写入: {out_dir / 'prediction_summary.csv'}")
        return

    for case_id in map(str, cfg["case_ids"]):
        image_path, _, _ = drive_case_paths(cfg["drive_root"], cfg["split"], case_id)
        rgb = load_rgb(image_path)
        original_shape = rgb.shape[:2]
        x = rgb.astype(np.float32) / 255.0
        padded, pads = center_pad_to_square(x, int(mcfg["image_size"]), fill_value=0.0)
        pred_pad = model.predict(padded[None, ...], batch_size=int(mcfg.get("batch_size", 1)), verbose=0)[0, :, :, 0]
        prob = remove_center_padding(pred_pad, pads, original_shape)
        mask = prob >= float(mcfg["threshold"])
        prob_path = prob_dir / f"{case_id}_sa_unet_prob.png"
        mask_path = mask_dir / f"{case_id}_sa_unet_mask.png"
        save_probability(prob, prob_path)
        save_binary(mask, mask_path)
        rows.append({
            "case_id": case_id,
            "status": "saved",
            "image_path": str(image_path),
            "prob_path": str(prob_path),
            "mask_path": str(mask_path),
            "threshold": float(mcfg["threshold"]),
            "prob_min": float(prob.min()),
            "prob_mean": float(prob.mean()),
            "prob_max": float(prob.max()),
            "mask_pixels": int(mask.sum()),
        })

    summary_path = out_dir / "prediction_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"预测完成: {summary_path}")


if __name__ == "__main__":
    main()
