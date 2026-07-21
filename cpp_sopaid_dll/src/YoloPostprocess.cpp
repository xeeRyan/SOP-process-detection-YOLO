#include "YoloPostprocess.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <map>
#include <opencv2/dnn.hpp>
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

void ApplyClasswiseNms(
    std::vector<SopAidDetection>& detections,
    float confidence_threshold,
    float nms_threshold) {
    std::map<int32_t, std::vector<size_t>> grouped;
    for (size_t i = 0; i < detections.size(); ++i) {
        grouped[detections[i].class_id].push_back(i);
    }
    std::vector<SopAidDetection> kept;
    for (const auto& entry : grouped) {
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        for (const size_t index : entry.second) {
            const auto& detection = detections[index];
            boxes.emplace_back(
                static_cast<int>(detection.x1),
                static_cast<int>(detection.y1),
                std::max(0, static_cast<int>(detection.x2 - detection.x1)),
                std::max(0, static_cast<int>(detection.y2 - detection.y1)));
            scores.push_back(detection.confidence);
        }
        std::vector<int> selected;
        cv::dnn::NMSBoxes(boxes, scores, confidence_threshold, nms_threshold, selected);
        for (const int selected_index : selected) {
            kept.push_back(detections[entry.second[static_cast<size_t>(selected_index)]]);
        }
    }
    std::sort(kept.begin(), kept.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.confidence > rhs.confidence;
    });
    detections = std::move(kept);
}
