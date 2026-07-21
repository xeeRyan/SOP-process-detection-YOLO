# C++ three-format model deployment

One training run can retain the Ultralytics checkpoint and export three C++ deployment artifacts:

| Artifact | Purpose | C++ backend |
|---|---|---|
| `best.pt` | Python training, resume, and Ultralytics inference | Not loaded directly by C++ |
| `best.torchscript` | LibTorch inference | `SopAid_InitPt` |
| `best.onnx` | ONNX Runtime inference | `SopAid_InitOnnx` |
| `best.engine` | TensorRT inference | `SopAid_InitEngine` |

`SopAid_InitPt` is retained for API compatibility, but its input must contain a
TorchScript archive. A regular Ultralytics training checkpoint cannot be loaded
with `torch::jit::load`.

## Training request

```json
{
  "export_torchscript": true,
  "torchscript_output_path": "models/best_yolo26n.torchscript",
  "torchscript_optimize": false,
  "export_onnx": true,
  "onnx_output_path": "models/best_yolo26n.onnx",
  "export_engine": true,
  "engine_output_path": "models/best_yolo26n.engine",
  "trtexec_path": "E:\\TensorRT-11.1.0.106\\bin\\trtexec.exe",
  "export_strict": true
}
```

The response reports each artifact independently:

```json
{
  "exported_models": {
    "requested": {
      "torchscript": true,
      "onnx": true,
      "engine": true
    },
    "torchscript": "models/best_yolo26n.torchscript",
    "onnx": "models/best_yolo26n.onnx",
    "engine": "models/best_yolo26n.engine",
    "errors": []
  }
}
```

## C++ selection

Keep `model_format` as `Auto`; the DLL selects the backend from the extension:

```cpp
SopAidInitConfig config{};
config.model_path = "models/best_yolo26n.torchscript";
config.model_format = SopAidModelFormat::Auto;
```

Supported extensions:

- `.torchscript` or a TorchScript archive named `.pt`: LibTorch
- `.onnx`: ONNX Runtime
- `.engine` or `.plan`: TensorRT

TensorRT engines are tied to the TensorRT version and target GPU. Build the
engine on the deployment machine whenever possible.
