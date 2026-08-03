# The wokflow from vessels images to graph map to path planning



```text
../SA-UNet-master/Model/DRIVE/SA_UNet.h5
```

code flow：

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

## run

```bash
cd sa_unet_h5_navigation_workflow
python 06_run_all.py
```

## output

- `outputs/02_predict/probability_maps/`
- `outputs/02_predict/masks/`
- `outputs/03_clean_evaluate/segmentation_metrics.csv`
- `outputs/04_graph_navigation/graphs/`
- `outputs/04_graph_navigation/paths/`
- `outputs/05_visualization/overlays/`
- `outputs/06_graph_path_visualization/graph_path_overlays/`
- `outputs/06_run_all/run_all_log.csv`
