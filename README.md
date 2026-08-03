# SA-UNet h5 图像到 Graph 到导航流程

这套 workflow **不训练模型**，只使用已有权重：

```text
../SA-UNet-master/Model/DRIVE/SA_UNet.h5
```

流程：

```text
DRIVE image
→ load SA_UNet.h5 weights
→ vessel probability map
→ binary vessel mask
→ cleaned mask
→ skeleton
→ graph nodes / graph edges
→ representative navigation path
→ overlay figures and CSV tables
```

## 运行

```bash
cd sa_unet_h5_navigation_workflow
python 06_run_all.py
```

## 输出

- `outputs/02_predict/probability_maps/`
- `outputs/02_predict/masks/`
- `outputs/03_clean_evaluate/segmentation_metrics.csv`
- `outputs/04_graph_navigation/graphs/`
- `outputs/04_graph_navigation/paths/`
- `outputs/05_visualization/overlays/`
- `outputs/06_graph_path_visualization/graph_path_overlays/`
- `outputs/06_run_all/run_all_log.csv`

## 重要说明

原 SA-UNet 项目使用老版本 Keras/TensorFlow，当前脚本用 `tf.keras` 重建同结构模型，并加载 h5 权重。若你要完全复现原作者环境，应使用它的 `requirements.txt` 中的 TensorFlow 1.14 / Keras 2.3；但当前 workflow 的目标是直接使用已有 h5 权重完成图像到导航任务。
