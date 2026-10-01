# PC detector models

Copy the trained geometry ONNX model into this directory or pass any compatible Ultralytics model with `--model`. The geometry model should expose `outer_buttons` and `inner_screen` classes. The PC lock uses only those two classes: `outer_buttons` supplies the cabinet anchor and `inner_screen` supplies the crop scale. A separate gameplay model may expose `button`/`inner_screen` or `slide`/`inner_screen`; its detections are intentionally not used by the cabinet lock.
