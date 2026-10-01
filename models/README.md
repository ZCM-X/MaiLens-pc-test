# PC detector models

Copy the trained `frame-geometry-yolo11n-v2.onnx` into this directory or pass any compatible Ultralytics model with `--model`. The model should expose `outer_frame` and `inner_screen` classes. The eight chart judgement markers are intentionally not used by the processor.
