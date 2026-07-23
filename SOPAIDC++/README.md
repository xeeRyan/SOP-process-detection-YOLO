# SOPAID C++ 推理 DLL

本工程的核心交付是 `SOPAID.dll`：封装 C++ 推理接口，支持 `.pt`、`.onnx`、`.engine` 三种模型格式。`SOPAIDExe` 是测试程序，用于读取视频、逐帧调用 DLL、绘制检测框和手部骨骼，并输出测试视频。

当前工程不包含 ROI 区域判断、SOP 流程状态机或 Python 交付包中的业务算法。若后续需要完整 SOP 判断，需要在 C++ 侧另行实现业务流程。

## 工程入口

```text
D:\C++\SOPAID\SOPAID.sln
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
```

常用字段：

- `model_path`：模型路径。
- `model_format`：`Auto / Pt / Onnx / Engine`，`Auto` 会按后缀识别。
- `input_width / input_height`：模型输入尺寸，当前默认 `640 x 640`。
- `confidence_threshold`：置信度阈值。
- `nms_threshold`：NMS 阈值。
- `class_names_csv`：类别名，例如 `bearing,cover,tool`。
- `use_cuda / device_id`：后端设备参数。

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
SopAidHand_Init
HandEvaluate(cv::Mat, std::vector<SopAidHandResult>&)
SopAidHand_Release
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
SOPAID.exe <model_path> <video_path> [output_root] [confidence_threshold] [palm_model_path] [handpose_model_path] [frame_stride]
```

示例：

```text
D:\C++\SOPAID\x64\Release\SOPAIDExe\SOPAID.exe D:\Python\SOPAID\dist\SOP_PYD\models\sop_yolo26s_v1_best.engine D:\Python\SOPAID\dist\SOP_PYD\_internal\videos\sk.mp4 D:\Python\SOPAID\dist\SOP_PYD\outputs 0.25 D:\Python\SOPAID\dist\SOP_PYD\models\palm_detection_mediapipe_2023feb.onnx D:\Python\SOPAID\dist\SOP_PYD\models\handpose_estimation_mediapipe_2023feb.onnx 5
```

`frame_stride` 表示抽帧推理间隔：

- `1`：每帧都推理。
- `5`：每 5 帧推理一次，中间帧复用上一次结果。

## 输出规则

默认输出根目录：

```text
D:\Python\SOPAID\dist\SOP_PYD\outputs
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
D:\Python\SOPAID\dist\SOP_PYD\outputs\logs
```

## 本机依赖配置

当前依赖路径在 `SopAidInfer.user.props` 中配置：

- OpenCV：`D:\Opencv\10\opencv\build`
- ONNX Runtime：`D:\Opencv\Onnxruntime\onnxruntime-win-x64-1.15.1`
- TensorRT：`E:\TensorRT-11.1.0.106`
- LibTorch：`E:\libtorch`
- C++ 标准：C++20

注意：普通 Ultralytics 训练得到的 `.pt` 通常不能直接由 LibTorch 加载；`.pt` 后端要求 TorchScript 模型。
