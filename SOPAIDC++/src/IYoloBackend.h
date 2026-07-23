#pragma once

#include "SopAidInfer.h"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

// 统一 YOLO 后端接口。DLL 主流程只依赖这个抽象，具体模型格式由不同后端实现。
class IYoloBackend {
public:
    virtual ~IYoloBackend() = default;
    // 输入单帧 BGR 图像，输出统一的检测结构体列表。
    virtual SopAidStatus evaluate(const cv::Mat& image, std::vector<SopAidDetection>& results, SopAidError* error) = 0;
};

// 三种模型格式分别创建对应后端；未启用的后端会返回 UnsupportedModel。
std::unique_ptr<IYoloBackend> CreateOnnxBackend(const SopAidInitConfig& config, SopAidError* error);
std::unique_ptr<IYoloBackend> CreateTensorRtBackend(const SopAidInitConfig& config, SopAidError* error);
std::unique_ptr<IYoloBackend> CreateTorchScriptBackend(const SopAidInitConfig& config, SopAidError* error);
