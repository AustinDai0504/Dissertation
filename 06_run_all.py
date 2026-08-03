#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""一键运行：h5 推理 -> graph -> navigation。"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from common import ensure_dir, load_config


STEPS = [
    "common.py",
    "01_sa_unet_model.py",
    "02_predict_with_h5.py",
    "03_clean_and_evaluate.py",
    "04_graph_and_navigation.py",
    "05_visualize.py",
    "06_visualize_path_on_graph.py",
]


def main() -> None:
    cfg = load_config()
    script_dir = Path(__file__).resolve().parent
    out_dir = ensure_dir(Path(cfg["output_dir"]) / "06_run_all")
    rows = []
    for step in STEPS:
        started = datetime.now()
        result = subprocess.run([sys.executable, str(script_dir / step)], cwd=script_dir, text=True, capture_output=True)
        rows.append({
            "step": step,
            "returncode": result.returncode,
            "started_at": started.isoformat(),
            "finished_at": datetime.now().isoformat(),
            "stdout_tail": result.stdout[-1500:],
            "stderr_tail": result.stderr[-1500:],
        })
        if result.returncode != 0:
            break
    log_path = out_dir / "run_all_log.csv"
    pd.DataFrame(rows).to_csv(log_path, index=False)
    print(f"总运行日志完成: {log_path}")


if __name__ == "__main__":
    main()
