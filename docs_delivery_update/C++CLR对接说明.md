# C++/CLR 前端对接说明

适用场景：C# / WPF 对摄像头帧或图片进行低延迟 YOLO 与手部关键点推理。

## 1. 接口定位

C# 入口程序集是 `SOPAID_wrapper.dll`，命名空间 `SOPAIDwrapper`：

- `InferenceEvaluator`：YOLO 目标检测。
- `HandPoseEvaluator`：掌心检测与每只手 21 个关键点。

输入是解码后的单帧字节数组。前端负责摄像头/视频解码、抽帧、线程、显示和绘制。

## 2. 工程配置与部署

1. C# 使用 .NET Framework 4.7.2 或兼容 Windows .NET Framework 工程。
2. 平台目标固定 `x64`，关闭“首选 32 位”。
3. 引用 `C++推理SDK\frontend_bin\SOPAID_wrapper.dll`。
4. 将 `frontend_bin` 所有文件复制到前端 EXE 同目录。
5. 目标机安装 `vc_redist.x64.exe`。

| 异常 | 常见原因 |
|---|---|
| `BadImageFormatException` | 前端是 x86，SDK 是 x64 |
| 包装器找不到 | 引用或复制路径错误 |
| 提示找不到模块 | `SOPAID.dll`、OpenCV 或后端依赖缺失 |
| `BackendError` | 模型、GPU 运行时、驱动或后端不兼容 |

## 3. 推荐：项目目录初始化

项目模式从 `project.json` 读取类别，并按 `active_model_manifest` 选择模型，可避免类别顺序错误。

```csharp
using System;
using System.Collections.Generic;
using SOPAIDwrapper;

var config = new ProjectInferenceConfig
{
    ProjectDirectory = @"D:\SOPAID_Deliever11\model_projects\keyboard",
    PreferredFormat = ModelFormat.Onnx,
    InputWidth = 640,
    InputHeight = 640,
    ConfidenceThreshold = 0.25f,
    NmsThreshold = 0.70f,
    UseCuda = false,
    DeviceId = 0
};

using (var detector = new InferenceEvaluator(config))
{
    if (!detector.IsInitialized)
        throw new InvalidOperationException(
            detector.LastError.Status + ": " + detector.LastError.Message);

    ModelInfo info = detector.GetModelInfo();
    // 保留并复用 detector，不要每帧 new。
}
```

`Auto` 实际优先级：ONNX → TorchScript → TensorRT engine。生产建议显式指定。

项目约束：

- 必须含合法 `project.json`。
- `classes` 非空，ID 从 0 开始、连续、不重复。
- 有 `active_model_manifest` 时读取其 `artifacts`。
- 无 manifest 时，只在 `active_model` 相邻位置找同名 `.onnx/.torchscript/.engine`。

## 4. 单模型初始化

```csharp
var config = new InferenceConfig
{
    ModelPath = @"D:\models\best.onnx",
    Format = ModelFormat.Onnx,
    InputWidth = 640,
    InputHeight = 640,
    ConfidenceThreshold = 0.25f,
    NmsThreshold = 0.70f,
    ClassNamesCsv = "bearing,cover,tool",
    UseCuda = false,
    DeviceId = 0
};
```

`ModelFormat.Pt` 实际表示 LibTorch TorchScript，推荐 `.torchscript`。普通 Ultralytics `best.pt` 不能直接使用。

## 5. 单帧 YOLO 推理

```csharp
var detections = new List<DetectionResult>();
bool ok = detector.Evaluate(
    frameBytes, frameWidth, frameHeight,
    3, frameStride, detections);

if (!ok)
    throw new InvalidOperationException(
        detector.LastError.Status + ": " + detector.LastError.Message);

foreach (var item in detections)
    DrawBox(item.X1, item.Y1, item.X2, item.Y2,
            item.ClassName, item.Confidence);
```

输入硬约束：

- `imageData` 非空；`width/height > 0`。
- `channels` 只能是 1=Gray、3=BGR、4=BGRA。
- `stride >= width * channels`，单位字节。
- 数组长度至少为 `stride * height`。
- 无 stride 重载按连续的 `width * channels` 处理。

每次调用会清空 `results`。框坐标已映射回原图像素，无需按 640 再缩放。

## 6. 摄像头处理链

```text
前端打开摄像头 → 获取帧 → 转 BGR/BGRA/Gray byte[]
→ 后台串行 Evaluate → 切回 UI 线程绘制
```

- 不要在 UI 线程同步推理。
- 不要无限堆积帧；通常丢弃旧帧、只保留最新帧。
- WPF `Pbgra32` 是预乘 Alpha，不应直接当普通 BGRA；优先 Bgr24。
- RGB/RGBA 必须先交换为 BGR/BGRA。

## 7. 手部关键点

```csharp
var config = new HandPoseConfig
{
    PalmModelPath = @"D:\SOPAID_Deliever11\models\hand_pose\palm_detection_mediapipe_2023feb.onnx",
    HandPoseModelPath = @"D:\SOPAID_Deliever11\models\hand_pose\handpose_estimation_mediapipe_2023feb.onnx",
    MaxHands = 2,
    PalmScoreThreshold = 0.5f,
    HandScoreThreshold = 0.5f,
    UseGpu = false,
    DeviceId = 0
};

using (var hands = new HandPoseEvaluator(config))
{
    if (!hands.IsInitialized)
        throw new InvalidOperationException(hands.LastError.Message);

    var results = new List<HandPoseResult>();
    bool ok = hands.Evaluate(frameBytes, width, height, channels, stride, results);
}
```

每只手正常为 21 点；`X/Y` 是原图像素，`Z` 是相对深度，`Visibility` 是可见度。`HandId` 不保证跨帧稳定。

## 8. 生命周期、线程和状态码

- 初始化一次、重复 Evaluate、结束 Dispose/Release。
- 一个 evaluator 串行调用；无并发线程安全承诺。
- 窗口关闭先停摄像头/循环，再释放 evaluator。

| 值 | 枚举 | 含义 | 前端处理 |
|---:|---|---|---|
| 0 | `Ok` | 成功 | 读取结果 |
| 1 | `InvalidArgument` | 参数或调用状态错误 | 修正参数 |
| 2 | `FileNotFound` | 文件不存在 | 检查部署路径 |
| 3 | `UnsupportedModel` | 格式不支持/无部署模型 | 调整或重新导出 |
| 4 | `BackendError` | 后端环境错误 | 检查 DLL/GPU 并记录 Message |
| 5 | `InferenceError` | 推理失败 | 记录输入和 Message |

## 9. 与 Python 组合

实时预览可用 CLR 对摄像头逐帧叠框；正式 SOP 可保存视频后调用 Python `detect`。Python 结果 JSON 才包含步骤、事件和最终 SOP 判定，不能把 CLR 单帧框等同于 SOP 结果。
