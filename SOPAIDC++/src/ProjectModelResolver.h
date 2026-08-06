#pragma once

#include "SopAidInfer.h"
#include <string>

struct ResolvedProjectModel {
    // 字符串由解析结果持有，用于在构造原生配置时保证 c_str() 生命周期有效。
    std::string project_id;
    std::string model_id;
    std::string model_version;
    std::string model_path;
    std::string class_names_csv;
    SopAidModelFormat model_format = SopAidModelFormat::Auto;
};

// 按 project.json 中的 active_model_manifest/active_model 解析可部署模型。
// 类别名称严格按连续的 class id 排序，确保输出 class_id 与 Python 项目定义一致。
SopAidStatus ResolveProjectModel(
    const SopAidProjectDirectoryConfig& config,
    ResolvedProjectModel& resolved,
    SopAidError* error);
