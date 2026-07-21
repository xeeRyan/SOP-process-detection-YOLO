#pragma once

#include "SopAidInfer.h"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

class IYoloBackend {
public:
    virtual ~IYoloBackend() = default;
    virtual SopAidStatus evaluate(const cv::Mat& image, std::vector<SopAidDetection>& results, SopAidError* error) = 0;
};

std::unique_ptr<IYoloBackend> CreateOnnxBackend(const SopAidInitConfig& config, SopAidError* error);
std::unique_ptr<IYoloBackend> CreateTensorRtBackend(const SopAidInitConfig& config, SopAidError* error);
std::unique_ptr<IYoloBackend> CreateTorchScriptBackend(const SopAidInitConfig& config, SopAidError* error);
