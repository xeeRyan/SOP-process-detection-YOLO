#include "YoloPostprocess.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <sstream>

namespace {

std::string trim(const std::string& value) {
    const auto begin = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) { return std::isspace(ch); });
    const auto end = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) { return std::isspace(ch); }).base();
    if (begin >= end) {
        return {};
    }
    return std::string(begin, end);
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

}  // namespace

std::vector<std::string> ParseClassNames(const char* csv) {
    // 类别名来自初始化结构体，为空时回退到当前 SOP 模型的默认三类。
    std::vector<std::string> names;
    std::stringstream stream(csv ? csv : "");
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = trim(item);
        if (!item.empty()) {
            names.push_back(item);
        }
    }
    if (names.empty()) {
        names = {"bearing", "cover", "tool"};
    }
    return names;
}

void SetError(SopAidError* error, SopAidStatus status, const std::string& message) {
    if (!error) {
        return;
    }
    error->status = status;
    std::memset(error->message, 0, sizeof(error->message));
    // message 是固定长度 C 字符数组，拷贝时保留最后一个字节给 '\0'。
    const auto count = std::min(message.size(), sizeof(error->message) - 1);
    std::memcpy(error->message, message.data(), count);
}

SopAidModelFormat DetectModelFormat(const std::string& path) {
    const auto ext = lower(std::filesystem::path(path).extension().string());
    if (ext == ".onnx") {
        return SopAidModelFormat::Onnx;
    }
    if (ext == ".engine" || ext == ".plan") {
        return SopAidModelFormat::Engine;
    }
    if (ext == ".pt" || ext == ".torchscript") {
        return SopAidModelFormat::Pt;
    }
    return SopAidModelFormat::Auto;
}

SopAidDetection MakeDetection(
    int class_id,
    const std::vector<std::string>& class_names,
    float confidence,
    const cv::Rect& box) {
    // 坐标统一输出为原始图像坐标，便于 EXE 直接画框或上层继续处理。
    SopAidDetection detection;
    detection.class_id = class_id;
    detection.confidence = confidence;
    detection.x1 = static_cast<float>(box.x);
    detection.y1 = static_cast<float>(box.y);
    detection.x2 = static_cast<float>(box.x + box.width);
    detection.y2 = static_cast<float>(box.y + box.height);

    const std::string name =
        (class_id >= 0 && class_id < static_cast<int>(class_names.size())) ? class_names[class_id] : std::to_string(class_id);
    std::memset(detection.class_name, 0, sizeof(detection.class_name));
    std::memcpy(detection.class_name, name.data(), std::min(name.size(), sizeof(detection.class_name) - 1));
    return detection;
}
