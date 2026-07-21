# SOPAID C++ 推理 DLL

本工程是 SOPAID 算法的 C++ DLL 封装工程，使用 Visual Studio 2019 编译，目标平台建议使用 `Release | x64`。

当前推理封装已经合并进原始 DLL 空模板工程。日常开发只需要打开：

```text
D:\C++\SOPAID\SOPAID.sln
```

## 目录说明

```text
D:\C++\SOPAID
├─ SOPAID.sln                     # VS2019 解决方案入口
├─ SOPAID.vcxproj                 # DLL 工程配置
├─ include\SopAidInfer.h          # 对外接口头文件
├─ src\SopAidInfer.cpp            # Init / EvaluateVideo / Release 主实现
├─ src\OnnxYoloBackend.cpp        # ONNX 推理后端，当前可用
├─ src\TensorRtYoloBackend.cpp    # TensorRT engine 后端占位
├─ src\TorchScriptYoloBackend.cpp # PT / TorchScript 后端占位
├─ src\YoloPostprocess.cpp        # YOLO 输出解析和 NMS 后处理
├─ examples\example_main.cpp      # C++ 调用示例
└─ SopAidInfer.user.props.template# OpenCV 路径配置模板
```

## 接口目标

DLL 对外提供三类能力：

1. 初始化模型，返回模型句柄。
2. 输入视频路径，DLL 内部逐帧读取视频并推理。
3. 释放模型句柄。

一个模型对应一个句柄。不同模型格式使用不同的初始化接口：

```cpp
SopAidHandle handle = sopaid::InitPt(config, &err);
SopAidHandle handle = sopaid::InitOnnx(config, &err);
SopAidHandle handle = sopaid::InitEngine(config, &err);
```

视频推理和释放接口统一：

```cpp
sopaid::EvaluateVideo(input, output, &err);
sopaid::Release(handle);
```

## 主要结构体

### 初始化结构体

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

常用字段说明：

- `model_path`：模型路径，例如 `D:/Python/SOPAID/dist/SOP_PYD/models/wpf_train_smoke_best.onnx`。
- `model_format`：模型格式，使用 `InitOnnx / InitPt / InitEngine` 时会自动固定。
- `input_width / input_height`：模型输入尺寸，默认 `640 x 640`。
- `confidence_threshold`：置信度阈值，默认 `0.25`。
- `nms_threshold`：NMS 阈值，默认 `0.70`。
- `class_names_csv`：类别名，当前可用 `bearing,cover,tool`。
- `use_cuda`：ONNX OpenCV DNN 后端是否尝试使用 CUDA。

### 视频推理输入结构体

```cpp
struct SopAidVideoEvaluateInput {
    SopAidHandle handle;
    const char* video_path;
    int32_t frame_step;
    int32_t max_frames;
};
```

字段说明：

- `handle`：模型句柄，由 `InitOnnx / InitPt / InitEngine` 返回。
- `video_path`：待检测视频路径。
- `frame_step`：抽帧间隔。`1` 表示每帧都检测，`5` 表示每 5 帧检测一次。
- `max_frames`：最多检测多少个抽样帧。`0` 表示不限制，直到视频结束。

### 视频推理输出结构体

C++ 推荐使用：

```cpp
struct SopAidVideoEvaluateResult {
    std::vector<SopAidVideoDetection> results;
};
```

如果需要 C 风格缓冲区，也可以使用：

```cpp
struct SopAidVideoEvaluateOutput {
    SopAidVideoDetection* results;
    int32_t result_capacity;
    int32_t result_count;
};
```

### 视频检测结果结构体

```cpp
struct SopAidVideoDetection {
    int64_t frame_index;
    double time_sec;
    SopAidDetection detection;
};
```

字段说明：

- `frame_index`：检测结果来自第几帧，从 `0` 开始。
- `time_sec`：检测结果对应的视频时间，单位秒。
- `detection`：该帧里的单个目标检测框。

### 单个检测框结构体

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

## C++ 视频调用示例

```cpp
#include "SopAidInfer.h"
#include <iostream>

int main() {
    SopAidError err;

    SopAidInitConfig config;
    config.model_path = "D:/Python/SOPAID/dist/SOP_PYD/models/wpf_train_smoke_best.onnx";
    config.input_width = 640;
    config.input_height = 640;
    config.confidence_threshold = 0.25f;
    config.nms_threshold = 0.70f;
    config.class_names_csv = "bearing,cover,tool";

    SopAidHandle handle = sopaid::InitOnnx(config, &err);
    if (!handle) {
        std::cout << "Init failed: " << err.message << std::endl;
        return 1;
    }

    SopAidVideoEvaluateInput input;
    input.handle = handle;
    input.video_path = "D:/Python/SOPAID/dist/SOP_PYD/_internal/videos/smoke_60f.mp4";
    input.frame_step = 1;
    input.max_frames = 0;

    SopAidVideoEvaluateResult output;
    SopAidStatus status = sopaid::EvaluateVideo(input, output, &err);
    if (status != SopAidStatus::Ok) {
        std::cout << "EvaluateVideo failed: " << err.message << std::endl;
        sopaid::Release(handle);
        return 2;
    }

    for (const auto& item : output.results) {
        const auto& det = item.detection;
        std::cout << "frame=" << item.frame_index
                  << " time=" << item.time_sec
                  << " " << det.class_name
                  << " " << det.confidence
                  << " [" << det.x1 << ", " << det.y1
                  << ", " << det.x2 << ", " << det.y2 << "]"
                  << std::endl;
    }

    sopaid::Release(handle);
    return 0;
}
```

## C 导出接口

如果上层不方便直接使用 C++ `std::vector`，可以调用 C 风格接口：

```cpp
SopAidHandle SopAid_InitPt(const SopAidInitConfig* config, SopAidError* error);
SopAidHandle SopAid_InitOnnx(const SopAidInitConfig* config, SopAidError* error);
SopAidHandle SopAid_InitEngine(const SopAidInitConfig* config, SopAidError* error);

SopAidStatus SopAid_EvaluateVideoStruct(
    const SopAidVideoEvaluateInput* input,
    SopAidVideoEvaluateOutput* output,
    SopAidError* error);

void SopAid_Release(SopAidHandle handle);
```

C 风格输出示例：

```cpp
SopAidVideoDetection buffer[4096];

SopAidVideoEvaluateOutput output;
output.results = buffer;
output.result_capacity = 4096;
output.result_count = 0;

SopAid_EvaluateVideoStruct(&input, &output, &err);
```

## 单帧接口说明

工程内部仍保留单帧 `cv::Mat` 推理接口：

```cpp
sopaid::Evaluate(frame_input, frame_output, &err);
```

它主要用于：

- 视频接口内部逐帧复用。
- 后续接相机实时流时复用。
- 调试单张图片或单帧结果。

普通视频检测调用方优先使用 `EvaluateVideo`。

## 编译步骤

1. 安装 VS2019，并确保安装了 C++ 桌面开发组件。
2. 安装或准备 OpenCV x64 版本。
3. 复制 `SopAidInfer.user.props.template`，重命名为 `SopAidInfer.user.props`。
4. 修改 `SopAidInfer.user.props` 中的 OpenCV 路径和版本号。
5. 打开 `D:\C++\SOPAID\SOPAID.sln`。
6. 选择 `Release | x64`。
7. 生成项目。

## 当前模型格式支持状态

- `.onnx`：已实现，使用 OpenCV DNN 推理。
- `.engine`：接口已预留，后续需要接 TensorRT SDK。
- `.pt`：接口已预留，后续需要接 LibTorch / TorchScript。

注意：Ultralytics 训练得到的 `.pt` 通常不是 C++ LibTorch 可直接加载的 TorchScript 文件。实际生产部署建议优先使用 ONNX 或 TensorRT engine。

## 当前推荐路径

当前阶段建议先跑通 ONNX：

```text
D:/Python/SOPAID/dist/SOP_PYD/models/wpf_train_smoke_best.onnx
```

示例视频可以先使用：

```text
D:/Python/SOPAID/dist/SOP_PYD/_internal/videos/smoke_60f.mp4
```

等 ONNX 视频推理接口、输入输出、上层调用流程稳定后，再继续补 TensorRT engine 后端。
