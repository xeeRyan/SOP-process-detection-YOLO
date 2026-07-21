# TensorRT 转换说明

本文档说明如何将当前 ONNX 模型转换为 TensorRT engine，并进行 ONNX/TensorRT 输出对齐。

## 当前状态

项目已有 ONNX 模型：

```text
models/best_yolo26s.onnx
```

本机曾检测到 NVIDIA GPU：

```text
NVIDIA GeForce RTX 2080
Driver 565.90
CUDA 12.7
```

如果当前环境未检测到 `trtexec` 或 TensorRT Python 包，则还不能直接生成 `.engine` 文件，需要先安装 NVIDIA TensorRT，并将 TensorRT 的 `bin` 目录加入 `PATH`。

## 1. 检查 trtexec

```powershell
trtexec --version
```

如果提示找不到命令，需要安装 TensorRT，或使用完整路径：

```powershell
python deploy/export_tensorrt.py --trtexec "C:\TensorRT\bin\trtexec.exe"
```

## 2. 转换 engine

默认 FP16 转换：

```powershell
python deploy/export_tensorrt.py
```

等价 trtexec 命令：

```powershell
trtexec --onnx=models/best_yolo26s.onnx --saveEngine=models/best_yolo26s.engine --fp16
```

如果需要 FP32：

```powershell
python deploy/export_tensorrt.py --fp32
```

输出文件：

```text
models/best_yolo26s.engine
```

## 3. ONNX/TensorRT 对齐测试

安装 Python 运行依赖后可执行：

```powershell
python deploy/compare_onnx_tensorrt.py
```

默认输出：

```text
outputs/onnx_tensorrt_compare.json
```

对齐结果重点关注：

```text
same_shape
max_abs_diff
mean_abs_diff
onnx_elapsed_ms
tensorrt_elapsed_ms
```

## 4. 注意事项

- TensorRT engine 与 GPU 架构、TensorRT 版本强相关，不建议跨机器复用。
- RTX 2080 支持 FP16，可优先使用 `--fp16`。
- 如果后续部署机器不同，应在目标机器上重新转换 engine。
- 当前主流程仍默认使用 `.pt` 或 `.onnx` 权重；TensorRT engine 先作为加速验证方向，不直接替换主流程。
