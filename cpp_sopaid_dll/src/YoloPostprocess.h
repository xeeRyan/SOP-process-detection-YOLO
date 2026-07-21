#pragma once

#include "SopAidInfer.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

std::vector<std::string> ParseClassNames(const char* csv);
void SetError(SopAidError* error, SopAidStatus status, const std::string& message);
SopAidModelFormat DetectModelFormat(const std::string& path);
SopAidDetection MakeDetection(
    int class_id,
    const std::vector<std::string>& class_names,
    float confidence,
    const cv::Rect& box);
void ApplyClasswiseNms(
    std::vector<SopAidDetection>& detections,
    float confidence_threshold,
    float nms_threshold);
