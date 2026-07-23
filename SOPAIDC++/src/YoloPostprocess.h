#pragma once

#include "SopAidInfer.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

// 将 "bearing,cover,tool" 这类逗号分隔配置转换为类别名列表。
std::vector<std::string> ParseClassNames(const char* csv);

// 统一写入错误状态，调用方可选择传 nullptr 忽略错误详情。
void SetError(SopAidError* error, SopAidStatus status, const std::string& message);

// 根据文件后缀识别模型格式，供 Auto 初始化模式使用。
SopAidModelFormat DetectModelFormat(const std::string& path);

// 将后端解析出的类别、置信度和框坐标封装成统一输出结构体。
SopAidDetection MakeDetection(
    int class_id,
    const std::vector<std::string>& class_names,
    float confidence,
    const cv::Rect& box);
