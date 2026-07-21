# SOP 工序检测部署计划

当前项目采用参考 `DEEPAIY/DEEPLEARN_PYD.py` 的封装形式，但不修改 `DEEPAIY` 文件夹。

## 当前封装形式

```text
WPF/外部软件
  -> TCP 或命令行
  -> SOP_PYD.py / SOP_PYD.exe
  -> scripts 运行时算法核心
```

`SOP_PYD.py` 支持：

- `tcp [port]`：启动 TCP 服务。
- `detect health`：命令行健康检查。
- `detect "{json}"`：命令行传 JSON 字符串检测。
- `detect_file path.json`：命令行传 JSON 文件检测。
- `train "{json}"`：命令行传 JSON 字符串训练。
- `train_file path.json`：命令行传 JSON 文件训练。
- TCP 消息 `model_file:path.json`：读取 JSON 文件并执行任务。
- TCP 消息 `close`：关闭服务。

## TCP 通信约定

```text
客户端发送 UTF-8 字符串
服务端返回 UTF-8 JSON 字符串
```

发送 JSON 文件路径：

```text
model_file:D:\Python\SOPAID\config\detect_request.json
```

发送 JSON 内容：

```json
{
  "command": "detect",
  "params": {
    "video_path": "videos/sk.mp4",
    "output_dir": "outputs/yolo26s"
  }
}
```

响应：

```json
{
  "status": "ok",
  "message": "",
  "data": {}
}
```

## EXE 交付阶段

推荐将 `SOP_PYD.py` 作为 PyInstaller 打包入口。

最终交付结构：

```text
SOP_PYD/
├── SOP_PYD.exe
├── config/
│   └── app_config.json
├── models/
│   ├── best_yolo26s.pt 或 best_yolo26s.onnx
│   └── hand_landmarker.task
├── outputs/
│   ├── yolo26s/
│   └── logs/
└── docs/
```

模型、配置、输出目录建议外置，便于现场替换模型、调整 ROI 和排查日志。

## ONNX 阶段

导出命令：

```powershell
python deploy/export_onnx.py --overwrite
```

默认输出：

```text
models/best_yolo26s.onnx
```

PT/ONNX 对齐验证：

```powershell
python deploy/compare_pt_onnx.py
```

## TensorRT 阶段

PC 端如果使用 NVIDIA 显卡，建议走 TensorRT 加速：

```text
best_yolo26s.pt -> best_yolo26s.onnx -> best_yolo26s.engine
```

当前项目已提供：

```text
deploy/export_tensorrt.py
deploy/compare_onnx_tensorrt.py
docs/TENSORRT_DEPLOY.md
```

## 推荐里程碑

1. 固定 `SOP_PYD.py` 的 TCP/命令行调用格式。
2. WPF 完成 `model_file:path.json` 调用联调。
3. 固定输出 JSON 字段和日志字段。
4. 打包 `SOP_PYD.py` 为 exe。
5. 导出 ONNX 并完成 PT/ONNX 对齐。
6. 根据硬件验证 TensorRT 或 RKNN 加速版本。


## ??????

???????`SOP_PYD.py/SOP_PYD.exe -> task_dispatcher.py -> tools/training.py ? scripts/main_video.py`?
