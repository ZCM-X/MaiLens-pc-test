# iOS resources

`MachineDetector.mlpackage` lands here during the Codemagic build:
`Training/export_coreml.py` converts `Training/machine-detector.pt` with
Ultralytics and coremltools, which only run on macOS or Linux.

A local build without that step simply ships without the detector; the app
guards the load and falls back to the plain preview.
