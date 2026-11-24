# A-nanTa V8 Native EA

Native MQL5 input, pipeline and output headers. No trading policy yet.

From repository root, install torch, numpy, pandas, onnx and onnxruntime, then run:

```sh
python EA/export_mt5_scalers.py
python EA/export_v8_onnx.py
```

Generated files: `EA/AnantaScalers.mqh`, `EA/ananta_v8_btc_dollar.onnx`, `EA/onnx_manifest.json`. The exporter reads the existing frozen checkpoint and architecture from the parent project. Original paths retained for compatibility; EA is the new development workspace. Generated binaries and MetaEditor compilation are not yet verified. Quote proxy calibration and numerical parity remain required.
