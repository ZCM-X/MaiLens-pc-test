# PC detector models

`frame-geometry-yolo11n-v3.onnx` is the model trained from the MaiMoller fish-eye photo set. It exposes `outer_buttons` and `inner_screen`. The PC lock uses only those two classes: `outer_buttons` supplies the cabinet anchor and `inner_screen` supplies the crop scale. The older `frame-geometry-yolo11n-v2.onnx` remains available for comparison.

To train another model from a dataset made by `tools/annotate_geometry.py`:

```powershell
.\.venv\Scripts\python.exe tools\train_detector.py `
  --dataset datasets\maimoller-geometry\dataset.yaml `
  --weights ..\..\yolo11n.pt --device 0 `
  --project runs --name geometry-yolo11n `
  --export onnx
```

A separate gameplay model may expose `button`/`inner_screen` or `slide`/`inner_screen`; its detections are intentionally not used by the cabinet lock.
