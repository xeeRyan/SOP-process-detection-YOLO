# SOPAID C++ 推理 DLL

本工程的核心交付是 `SOPAID.dll`：封装 C++ 推理接口，支持 `.pt`、`.onnx`、`.engine` 三种模型格式。`SOPAIDExe` 是测试程序，用于读取视频、逐帧调用 DLL、绘制检测框和手部骨骼，并输出测试视频。

当前工程不包含 ROI 区域判断、SOP 流程状态机或 Python 交付包中的业务算法。它是与
Python推理接口并列的底层推理SDK，不替代Python训练和完整SOP识别链路。

## 多SOP项目约定

每个SOP项目拥有独立类别和模型版本。C++调用方必须：

1. 从`projects/<PROJECT_ID>/project.json`按类别`id`顺序生成`class_names_csv`；
2. 从活动模型的`model_manifest.json`选择`torchscript`、`onnx`或`engine`产物；
3. 将`project_id`、`model_id`和`model_version`传入初始化配置，便于结果追溯；
4. 不要把Python训练检查点`best.pt`当作LibTorch模型，C++的PT入口要求TorchScript。

SDK默认采用与当前Python/Ultralytics导出链路一致的letterbox预处理，并将检测框还原到
原始图像像素坐标。

### 直接从项目目录初始化

新版接口可直接接收 `projects/<PROJECT_ID>`，调用方无需重复解析项目 JSON：

```cpp
SopAidProjectDirectoryConfig config;
config.project_dir = "projects/SK_DEMO";
config.preferred_model_format = SopAidModelFormat::Auto;

sopaid::Inference inference;
SopAidError error;
const auto status = inference.InitProjectDirectory(config, &error);
```

解析规则：

- 类别按 `project.json/classes[].id` 排序，ID 必须从 0 连续递增；
- 存在 `active_model_manifest` 时，从 `artifacts` 选择部署模型；
- 没有模型清单时，在 `active_model` 同目录查找同名 `.onnx`、`.torchscript`、`.engine`；
- `Auto` 默认优先 ONNX，其次 TorchScript，最后 TensorRT；
- 普通 Ultralytics `.pt` 训练检查点不会被误当作 C++ TorchScript 模型。

## 工程入口

```text
SOPAIDC++/SOPAID.sln
```

解决方案包含两个项目：

```text
SOPAID.vcxproj       # DLL 工程，提供推理接口
SOPAIDExe.vcxproj    # 测试 EXE，调用 DLL 输出结果视频
```

## DLL 接口边界

`SOPAID.dll` 的目标是三格式模型推理：

- `.pt`：TorchScript / LibTorch 后端
- `.onnx`：ONNX Runtime 后端
- `.engine`：TensorRT 后端

主要接口：

```cpp
Init / InitPt / InitOnnx / InitEngine
Evaluate(cv::Mat, std::vector<SopAidDetection>&)
Release
```

一个模型对应一个句柄。调用方负责读取图片或视频帧，并把 `cv::Mat` 传入 `Evaluate`。

## 主要文件

```text
include\SopAidInfer.h          # DLL 对外推理接口、参数结构体、检测结果结构体
include\SopAidHandPose.h       # ONNX 手部骨骼接口结构体
src\SopAidInfer.cpp            # Init / Evaluate / Release 主流程
src\OnnxYoloBackend.cpp        # ONNX Runtime 后端
src\TensorRtYoloBackend.cpp    # TensorRT engine 后端
src\TorchScriptYoloBackend.cpp # TorchScript pt 后端
src\SopAidHandPose.cpp         # OpenCV DNN 手部 ONNX 推理
src\YoloPostprocess.cpp        # YOLO 后处理和辅助函数
app\main.cpp                   # SOPAIDExe 测试程序
SopAidInfer.user.props          # 本机依赖路径配置
```

## 推理输入参数

```cpp
struct SopAidInitConfig {
    const char* model_path;
    SopAidModelFormat model_format;
    int32_t input_width;
    int32_t input_height;
    float confidence_threshold;
    float nms_threshold;
    const char* class_names_csv;
    bool use_cuda;
    int32_t device_id;
};

struct SopAidProjectInitConfig {
    uint32_t struct_size;
    uint32_t api_version;
    SopAidInitConfig inference;
    SopAidResizeMode resize_mode;
    const char* project_id;
    const char* model_id;
    const char* model_version;
};
```

常用字段：

- `model_path`：模型路径。
- `model_format`：`Auto / Pt / Onnx / Engine`，`Auto` 会按后缀识别。
- `input_width / input_height`：模型输入尺寸，当前默认 `640 x 640`。
- `confidence_threshold`：置信度阈值。
- `nms_threshold`：NMS 阈值。
- `class_names_csv`：必填；按项目类别id顺序传入，例如`keyboard_front,keyboard_back,battery`。
- `use_cuda / device_id`：后端设备参数。
- `resize_mode`：当前必须为`Letterbox`。
- `project_id / model_id / model_version`：可选追溯元数据，不参与模型计算。

旧项目继续调用`Init(SopAidInitConfig)`；多SOP项目调用
`InitProject(SopAidProjectInitConfig)`。旧结构体布局保持不变，新旧调用端可以逐步迁移。
初始化后可调用`GetModelInfo`读取实际后端、输入尺寸、类别数量及项目模型标识。

多项目初始化示例：

```cpp
SopAidProjectInitConfig project_config;
project_config.inference.model_path =
    "projects/keyboard/models/sop_detector/1.0.0/best.onnx";
project_config.inference.model_format = SopAidModelFormat::Onnx;
project_config.inference.class_names_csv =
    "keyboard_front,keyboard_back,battery_compartment_open,battery,battery_cover_closed";
project_config.project_id = "keyboard";
project_config.model_id = "sop_detector";
project_config.model_version = "1.0.0";

sopaid::Inference inference;
SopAidError error;
const auto status = inference.InitProject(project_config, &error);
```

## 检测输出结构体

```cpp
struct SopAidDetection {
    int32_t class_id;
    char class_name[64];
    float confidence;
    float x1;
    float y1;
    float x2;
    float y2;
};
```

坐标格式为左上角和右下角：`x1, y1, x2, y2`。

## 手部骨骼接口

当前手部骨骼使用 OpenCV DNN 加载两个 ONNX 模型：

```text
palm_detection_mediapipe_2023feb.onnx
handpose_estimation_mediapipe_2023feb.onnx
```

接口：

```cpp
HandInit
HandEvaluate(cv::Mat, std::vector<SopAidHandResult>&)
HandRelease
```

输出结构体包含单只手的 21 个关键点：

```cpp
struct SopAidHandResult {
    int32_t hand_id;
    float confidence;
    char handedness[16];
    SopAidHandLandmark landmarks[21];
};
```

说明：MediaPipe 0.10.10 只作为实验环境保留，当前项目测试链路暂不使用 MediaPipe runner。

## SOPAIDExe 测试程序

启动格式：

```text
SOPAID.exe <model_path> <video_path> [output_root] [confidence_threshold] [nms_threshold] [palm_model_path] [handpose_model_path] [sample_interval] [class_names_csv] [project_id] [model_id] [model_version]
```

示例：

```text
SOPAIDExe.exe models/best_yolo26s.engine videos/sk.mp4 outputs 0.25 0.70 models/palm_detection_mediapipe_2023feb.onnx models/handpose_estimation_mediapipe_2023feb.onnx 5
```

`frame_stride` 表示抽帧推理间隔：

- `1`：每帧都推理。
- `5`：每 5 帧推理一次，中间帧复用上一次结果。

## 输出规则

默认输出根目录：

```text
outputs
```

每次运行会新建结果文件夹：

```text
<model_name>_<video_name>_<yyyyMMdd_HHmmss>
```

结果文件夹包含：

```text
params.json          # 本次模型、视频、阈值、手部模型、抽帧参数
result.mp4           # 绘制 YOLO 检测框和 ONNX 手部骨骼的视频
frame_results.jsonl  # 抽帧推理结果，每行包含 detections 和 hands
```

日志单独放入：

```text
outputs/logs
```

## 本机依赖配置

当前依赖路径在 `SopAidInfer.user.props` 中配置：

- OpenCV：通过 `SOPAID_OPENCV_ROOT` 或 `SopAidInfer.user.props` 配置
- ONNX Runtime：通过 `SOPAID_ONNXRUNTIME_ROOT` 或 `SopAidInfer.user.props` 配置
- TensorRT、LibTorch：仅在启用对应后端时通过本机 props 配置
- C++ 标准：C++20

注意：普通 Ultralytics 训练得到的 `.pt` 通常不能直接由 LibTorch 加载；`.pt` 后端要求 TorchScript 模型。
