# SOPAID Python TCP 接口说明

本文档说明当前 SOPAID 项目中 Python 算法端与软件端/WPF 端的 TCP + JSON 对接方式。

## 1. 模块定位

`SOP_PYD.exe` 是当前项目打包后的 Python 算法入口，负责 TCP 通信、JSON 解析、任务分发、训练启动和视频检测启动。

当前调用链如下：

```text
软件端/WPF
  -> TCP 发送 JSON 请求
  -> SOP_PYD.exe
  -> task_dispatcher.py
  -> tools/training.py 或 scripts/main_video.py
  -> 返回 JSON 结果
```

当前 Python 端支持三类任务：

- `health`：检查算法服务是否启动。
- `train`：按软件端传入的训练参数启动 YOLO 模型训练。
- `detect`：按软件端传入的视频、模型和检测参数执行 SOP 视频检测。

## 2. 启动方式

使用默认配置启动 TCP 服务：

```powershell
.\dist\SOP_PYD\SOP_PYD.exe
```

指定端口启动 TCP 服务：

```powershell
.\dist\SOP_PYD\SOP_PYD.exe tcp 9000
```

默认 TCP 地址从 `config/app_config.json` 读取：

```json
{
  "tcp": {
    "host": "0.0.0.0",
    "port": 9000
  }
}
```

如果软件端和算法端在同一台电脑，软件端可连接：

```text
127.0.0.1:9000
```

如果软件端和算法端在不同电脑，软件端应连接算法电脑的局域网 IP：

```text
算法电脑IP:9000
```

例如：

```text
192.168.1.20:9000
```

## 3. TCP 协议格式

编码格式：`UTF-8`。

一次 TCP 请求发送一个任务，算法端处理完成后返回一个 JSON 响应。

支持三种请求形式。

直接发送 JSON：

```json
{"command":"health"}
```

发送 JSON 文件路径：

```text
json_file:D:\Python\SOPAID\config\train_request_ui_v1.json
```

兼容旧项目风格的文件路径：

```text
model_file:D:\Python\SOPAID\config\detect_request.json
```

`model_file:` 只是为了兼容参考项目写法，新对接建议统一使用 `json_file:` 或直接发送 JSON。

关闭 TCP 服务：

```text
close
```

## 4. 统一返回格式

成功返回：

```json
{
  "status": "ok",
  "message": "",
  "data": {}
}
```

失败返回：

```json
{
  "status": "error",
  "message": "错误信息",
  "data": {}
}
```

软件端应优先判断 `status`，当 `status == "ok"` 时再读取 `data`。

## 5. 参数优先级

当前项目的有效参数优先级为：

```text
软件端请求 params > config/app_config.json > scripts/config.py 默认常量
```

也就是说，软件端不需要每次传完整参数，只传需要覆盖的字段即可。未传字段会走项目默认配置。

## 6. health 接口

请求：

```json
{
  "command": "health"
}
```

返回 `data` 示例：

```json
{
  "service": "SOP_PYD",
  "status": "ok",
  "commands": ["health", "detect", "train"]
}
```

## 7. train 训练接口

请求示例：

```json
{
  "command": "train",
  "params": {
    "model": "models/yolo26s.pt",
    "data": "datasets/sop/sop.yaml",
    "project_dir": "runs",
    "run_name": "sop_yolo26s_ui_exp",
    "epochs": 100,
    "batch": 8,
    "imgsz": 640,
    "workers": 0,
    "device": null,
    "optimizer": "auto",
    "amp": true,
    "val_ratio": 0.2,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "copy_best_to": "models/best_yolo26s_ui_exp.pt",
    "export_onnx": true,
    "export_engine": false,
    "onnx_output_path": "models/best_yolo26s_ui_exp.onnx",
    "engine_output_path": "models/best_yolo26s_ui_exp.engine",
    "export_imgsz": 640,
    "export_opset": 12,
    "export_overwrite": true,
    "trtexec_path": "trtexec"
  }
}
```

训练参数说明：

| 字段 | 含义 |
| --- | --- |
| `model` | 初始 YOLO 模型路径，例如 `models/yolo26s.pt`。 |
| `data` | 数据集配置文件，例如 `datasets/sop/sop.yaml`。 |
| `project_dir` | 训练输出根目录。 |
| `run_name` | 本次训练任务名称。 |
| `epochs` | 训练轮数。 |
| `batch` | 批大小。 |
| `imgsz` | 训练输入图像尺寸。 |
| `workers` | 数据加载线程数，打包环境建议先用 `0`。 |
| `device` | 训练设备，`null` 表示自动，常见值为 `0` 或 `cpu`。 |
| `optimizer` | 优化器，例如 `auto`、`SGD`、`Adam`、`AdamW`。 |
| `amp` | 是否启用混合精度训练。 |
| `val_ratio` | 自动划分数据集时的验证集比例。 |
| `images_dir` | 可选，原始图片目录。 |
| `labels_dir` | 可选，YOLO 标签目录。 |
| `dataset_output_dir` | 可选，整理后的数据集输出目录。 |
| `copy_best_to` | 将训练得到的 best 模型复制到指定路径，供软件端后续选择。 |
| `hsv_h` | 色度增强。 |
| `hsv_s` | 饱和度增强。 |
| `hsv_v` | 亮度增强。 |
| `degrees` | 随机旋转角度。 |
| `translate` | 随机平移比例。 |
| `scale` | 随机缩放比例。 |
| `shear` | 随机剪切角度。 |
| `fliplr` | 左右翻转概率。 |
| `flipud` | 上下翻转概率。 |
| `mosaic` | Mosaic 增强概率。 |
| `mixup` | MixUp 增强概率。 |
| `copy_paste` | Copy-Paste 增强概率。 |

训练返回 `data` 主要字段：

| 字段 | 含义 |
| --- | --- |
| `run_dir` | 本次训练输出目录。 |
| `best_model` | YOLO 输出的 best 模型路径。 |
| `last_model` | YOLO 输出的 last 模型路径。 |
| `results_csv` | 训练指标 CSV 路径。 |
| `args_yaml` | YOLO 本次训练参数记录。 |
| `copied_best_model` | 复制后的最终模型路径。 |
| `exported_models` | ????? ONNX / TensorRT engine ?????????? |
| `train_params` | 本次实际生效的训练参数。 |
| `log_path` | 本次训练日志路径。 |

## 8. detect 视频检测接口

请求示例：

```json
{
  "command": "detect",
  "params": {
    "video_path": "videos/sk.mp4",
    "model_path": "models/best_yolo26s.pt",
    "output_dir": "outputs/yolo26s",
    "confidence_threshold": 0.25,
    "nms_threshold": 0.7,
    "target_classes": ["bearing", "cover", "tool"],
    "work_roi": [1115, 1132, 1648, 1522],
    "screw_bin_roi": [1092, 505, 1410, 932],
    "tool_home_roi": [782, 508, 1108, 930],
    "enable_yolo": true,
    "enable_hand_pose": true,
    "hand_pose_model_path": "models/hand_landmarker.task",
    "output_video": true,
    "output_json": true,
    "realtime_display": false
  }
}
```

检测参数说明：

| 字段 | 含义 |
| --- | --- |
| `video_path` | 输入视频路径。 |
| `model_path` | 检测使用的 YOLO `.pt` 模型路径。 |
| `output_dir` | 检测结果输出目录。 |
| `confidence_threshold` | 置信度阈值。 |
| `nms_threshold` | NMS IoU 阈值。 |
| `target_classes` | 保留的检测类别，当前 SOP 类别为 `bearing`、`cover`、`tool`。 |
| `work_roi` | 主工作区域 ROI，格式 `[x1, y1, x2, y2]`。 |
| `screw_bin_roi` | 螺丝/物料区域 ROI。 |
| `tool_home_roi` | 工具初始/归位区域 ROI。 |
| `enable_yolo` | 是否启用 YOLO 检测。 |
| `enable_hand_pose` | 是否启用手部骨骼检测。 |
| `hand_pose_model_path` | MediaPipe 手部骨骼模型路径。 |
| `sop_step_enabled` | 可选，各 SOP 步骤是否启用。 |
| `sop_trigger_sources` | 可选，各 SOP 步骤触发来源。 |
| `sop_step_timeouts_sec` | 可选，各 SOP 步骤超时时间。 |
| `output_video` | 是否保存结果视频。 |
| `output_json` | 是否保存结果 JSON。 |
| `realtime_display` | 是否实时弹出 OpenCV 显示窗口，软件对接通常设为 `false`。 |

检测返回 `data` 包含 SOP 判断结果、输出视频路径、结果 JSON 路径、日志路径、检测管线参数和手部骨骼统计信息。开启 `output_json` 后，完整检测结果会写入 `output_dir`。

## 9. C# 调用示例

```csharp
using System.Net.Sockets;
using System.Text;

var host = "127.0.0.1";
var port = 9000;
var json = "{\"command\":\"health\"}";

using var client = new TcpClient(host, port);
using var stream = client.GetStream();

var requestBytes = Encoding.UTF8.GetBytes(json);
stream.Write(requestBytes, 0, requestBytes.Length);

var buffer = new byte[1024 * 1024];
var count = stream.Read(buffer, 0, buffer.Length);
var responseJson = Encoding.UTF8.GetString(buffer, 0, count);
```

## 10. 对接注意事项

- 软件端和算法端不在同一工作目录时，建议传绝对路径。
- 正式对接时建议 `realtime_display=false`，避免算法端弹窗阻塞。
- 当前训练接口是同步任务，训练结束后才返回 TCP 响应。
- 当前不会在每次训练后自动导出 ONNX/TensorRT，需要后续单独调用格式转换脚本或新增独立导出命令。
