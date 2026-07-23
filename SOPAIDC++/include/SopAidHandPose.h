#pragma once

#include "SopAidInfer.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

// 手部骨骼接口独立于 YOLO 检测接口。
// 当前实现使用 OpenCV Zoo 的两个 ONNX 模型：palm_detection 负责找手掌，handpose_estimation 负责输出 21 个关键点。
constexpr int32_t SOPAID_HAND_LANDMARK_COUNT = 21;

using SopAidHandHandle = void*;

// 手部模型初始化参数。
// palm_model_path 和 handpose_model_path 为推荐参数；model_path 保留给旧代码兼容，等同于 handpose_model_path。
struct SopAidHandInitConfig {
    const char* model_path = nullptr;
    const char* palm_model_path = nullptr;
    const char* handpose_model_path = nullptr;
    int32_t max_num_hands = 2;
    float min_hand_detection_confidence = 0.5f;
    float min_hand_presence_confidence = 0.5f;
    float min_tracking_confidence = 0.5f;
    bool use_gpu = false;
    int32_t device_id = 0;
};

// 单个手部关键点。x/y 使用原始图像像素坐标，z 保留模型输出的相对深度。
struct SopAidHandLandmark {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float visibility = 0.0f;
};

// 单只手的 21 点骨骼结果。
struct SopAidHandResult {
    int32_t hand_id = -1;
    float confidence = 0.0f;
    char handedness[16] = {};
    SopAidHandLandmark landmarks[SOPAID_HAND_LANDMARK_COUNT] = {};
};

extern "C" {
SOPAID_API SopAidHandHandle SopAidHand_Init(const SopAidHandInitConfig* config, SopAidError* error);
SOPAID_API void SopAidHand_Release(SopAidHandHandle handle);
}

namespace sopaid {

SOPAID_API SopAidHandHandle HandInit(const SopAidHandInitConfig& config, SopAidError* error = nullptr);
SOPAID_API SopAidStatus HandEvaluate(
    SopAidHandHandle handle,
    const cv::Mat& image,
    std::vector<SopAidHandResult>& results,
    SopAidError* error = nullptr);
SOPAID_API void HandRelease(SopAidHandHandle handle);

}  // namespace sopaid
