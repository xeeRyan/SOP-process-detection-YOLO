# C++/CLR API 参数字典

本表记录 `SOPAID_wrapper.dll` 当前公开参数的中文含义、默认值与硬校验。

## 1. ModelFormat

| 值 | 名称 | 含义 |
|---:|---|---|
| 0 | `Auto` | 自动选择 |
| 1 | `Pt` | TorchScript，不是训练检查点 |
| 2 | `Onnx` | ONNX Runtime |
| 3 | `Engine` | TensorRT engine |

## 2. InferenceConfig

| 属性 | 中文名 | 默认值 | 允许值/硬约束 | 建议 |
|---|---|---:|---|---|
| `ModelPath` | 模型路径 | 空 | 必填，文件存在 | 绝对路径 |
| `Format` | 模型格式 | Auto | 0～3 | 生产显式指定 |
| `InputWidth` | 输入宽 | 640 | `>0` | 与导出一致 |
| `InputHeight` | 输入高 | 640 | `>0` | 与导出一致 |
| `ConfidenceThreshold` | 置信度阈值 | 0.25 | `[0,1]` | 常用 0.2～0.6 |
| `NmsThreshold` | NMS IoU 阈值 | 0.70 | `[0,1]` | 常用 0.4～0.7 |
| `ClassNamesCsv` | 类别英文名顺序 | `bearing,cover,tool` | 逗号分隔，与类别 ID 一致 | 优先项目模式 |
| `UseCuda` | 使用 CUDA | false | 布尔 | 按后端选择 |
| `DeviceId` | GPU 编号 | 0 | `>=0` | 单卡为 0 |

## 3. ProjectInferenceConfig

| 属性 | 中文名 | 默认值 | 允许值/硬约束 |
|---|---|---:|---|
| `ProjectDirectory` | 项目目录 | 空 | 必填，含 `project.json` |
| `PreferredFormat` | 首选格式 | Auto | 0～3；Auto 为 ONNX→TorchScript→Engine |
| `InputWidth/Height` | 输入尺寸 | 640/640 | 均 `>0` |
| `ConfidenceThreshold` | 置信度阈值 | 0.25 | `[0,1]` |
| `NmsThreshold` | NMS 阈值 | 0.70 | `[0,1]` |
| `UseCuda` | 使用 CUDA | false | 布尔 |
| `DeviceId` | GPU 编号 | 0 | `>=0` |

## 4. Evaluate

| 参数 | 中文名 | 硬约束 |
|---|---|---|
| `imageData` | 像素缓冲区 | 非空；长度至少 `stride*height` |
| `width/height` | 原图尺寸 | 均 `>0` |
| `channels` | 通道/格式 | 1=Gray、3=BGR、4=BGRA |
| `stride` | 每行字节数 | `>= width*channels` |
| `results` | 结果列表 | 非 null；调用开始清空 |

返回 `false` 后读取 `LastError`。

## 5. DetectionResult

| 属性 | 中文名 | 说明 |
|---|---|---|
| `ClassId` | 类别 ID | 从 0 开始 |
| `ClassName` | 类别名 | 项目或 CSV |
| `Confidence` | 置信度 | 通常 `[0,1]` |
| `X1/Y1` | 左上角 | 原图像素 |
| `X2/Y2` | 右下角 | 原图像素 |
| `Width/Height` | 框宽高 | 端点差值 |

## 6. HandPoseConfig

| 属性 | 中文名 | 默认值 | 约束/建议 |
|---|---|---:|---|
| `ModelPath` | 单模型路径 | 空 | 与双模型路径二选一 |
| `PalmModelPath` | 掌心模型 | 空 | 与关键点模型同时提供 |
| `HandPoseModelPath` | 21 点模型 | 空 | 与掌心模型同时提供 |
| `MaxHands` | 最大手数 | 2 | UI 建议 `>=1`，常用 1～2 |
| `PalmScoreThreshold` | 掌心阈值 | 0.5 | UI 限制 `[0,1]` |
| `HandScoreThreshold` | 手存在/跟踪阈值 | 0.5 | UI 限制 `[0,1]` |
| `UseGpu` | 使用 GPU | false | 先用 CPU 验证 |
| `DeviceId` | GPU 编号 | 0 | UI 限制 `>=0` |

包装层只硬校验模型路径组合；MaxHands 和手部阈值未完整硬校验，正式 UI 必须按表限制。

## 7. 返回对象

`HandPoseResult`：`HandId` 不保证跨帧稳定；`Confidence` 为置信度；`Handedness` 为左右手；`Landmarks` 正常 21 点。点的 X/Y 为原图像素，Z 为相对深度。

`ModelInfo`：包含 `ProjectId`、`ModelId`、`ModelVersion`、实际 `ModelPath`、`Backend`、输入尺寸和类别数。初始化后应记录一次，确认 Auto 实际后端。
