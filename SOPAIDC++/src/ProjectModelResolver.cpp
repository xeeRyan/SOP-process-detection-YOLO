#include "ProjectModelResolver.h"
#include "YoloPostprocess.h"
#include <opencv2/core.hpp>
#include <filesystem>
#include <map>
#include <sstream>
#include <vector>

namespace {
namespace fs = std::filesystem;

bool ReadJson(const fs::path& path, cv::FileStorage& storage, SopAidError* error) {
    if (!fs::is_regular_file(path)) {
        SetError(error, SopAidStatus::FileNotFound, "JSON file does not exist: " + path.string());
        return false;
    }
    storage.open(path.string(), cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!storage.isOpened()) {
        SetError(error, SopAidStatus::InvalidArgument, "Failed to parse JSON: " + path.string());
        return false;
    }
    return true;
}

fs::path ResolvePath(const fs::path& root, const std::string& value) {
    // 清理 .. 等路径片段，后续文件检查和模型信息始终使用规范化绝对路径。
    fs::path path(value);
    return fs::weakly_canonical(path.is_relative() ? root / path : path);
}

std::string ReadClasses(const cv::FileNode& classes, SopAidError* error) {
    if (classes.empty() || !classes.isSeq()) {
        SetError(error, SopAidStatus::InvalidArgument, "project classes must be a non-empty array.");
        return {};
    }
    // JSON 数组顺序不作为类别顺序；显式按 id 排序以兼容人工编辑的项目文件。
    std::map<int, std::string> ordered;
    for (const auto& item : classes) {
        const auto id_node = item["id"];
        const auto name_node = item["name"];
        if (!id_node.isInt() || !name_node.isString()) {
            SetError(error, SopAidStatus::InvalidArgument, "Each class requires integer id and string name.");
            return {};
        }
        const int id = static_cast<int>(id_node);
        const std::string name = static_cast<std::string>(name_node);
        if (id < 0 || name.empty() || !ordered.emplace(id, name).second) {
            SetError(error, SopAidStatus::InvalidArgument, "Class ids must be unique non-negative integers.");
            return {};
        }
    }
    std::ostringstream csv;
    int expected = 0;
    for (const auto& [id, name] : ordered) {
        if (id != expected++) {
            SetError(error, SopAidStatus::InvalidArgument, "Class ids must be contiguous and start at zero.");
            return {};
        }
        if (id) csv << ',';
        csv << name;
    }
    return csv.str();
}

struct Choice { const char* key; const char* extension; SopAidModelFormat format; };

std::vector<Choice> Choices(SopAidModelFormat format) {
    if (format == SopAidModelFormat::Onnx) return {{"onnx", ".onnx", format}};
    if (format == SopAidModelFormat::Pt) return {{"torchscript", ".torchscript", format}};
    if (format == SopAidModelFormat::Engine) return {{"engine", ".engine", format}};
    return {
        {"onnx", ".onnx", SopAidModelFormat::Onnx},
        {"torchscript", ".torchscript", SopAidModelFormat::Pt},
        {"engine", ".engine", SopAidModelFormat::Engine},
    };
}

bool ResolveManifest(
    const fs::path& root,
    const fs::path& path,
    SopAidModelFormat preferred,
    ResolvedProjectModel& result,
    SopAidError* error) {
    cv::FileStorage manifest;
    if (!ReadJson(path, manifest, error)) return false;
    const auto artifacts = manifest.root()["artifacts"];
    if (!artifacts.isMap()) {
        SetError(error, SopAidStatus::InvalidArgument, "model manifest artifacts must be an object.");
        return false;
    }
    for (const auto& choice : Choices(preferred)) {
        const auto artifact = artifacts[choice.key];
        if (!artifact.isString()) continue;
        const fs::path candidate = ResolvePath(root, static_cast<std::string>(artifact));
        if (!fs::is_regular_file(candidate)) continue;
        result.model_path = candidate.string();
        result.model_format = choice.format;
        const auto id = manifest.root()["model_id"];
        const auto version = manifest.root()["model_version"];
        if (id.isString()) result.model_id = static_cast<std::string>(id);
        if (version.isString()) result.model_version = static_cast<std::string>(version);
        return true;
    }
    SetError(error, SopAidStatus::UnsupportedModel, "No requested C++ artifact exists in model manifest.");
    return false;
}

// active_model 可能指向 Python 训练检查点 best.pt；这里只取同名的 C++ 部署产物，
// 避免把普通 Ultralytics 检查点误交给 LibTorch。
bool ResolveAdjacent(const fs::path& active_model, SopAidModelFormat preferred, ResolvedProjectModel& result) {
    for (const auto& choice : Choices(preferred)) {
        fs::path candidate = active_model;
        candidate.replace_extension(choice.extension);
        if (!fs::is_regular_file(candidate)) continue;
        result.model_path = fs::weakly_canonical(candidate).string();
        result.model_format = choice.format;
        result.model_id = candidate.stem().string();
        return true;
    }
    return false;
}

}  // namespace

SopAidStatus ResolveProjectModel(
    const SopAidProjectDirectoryConfig& config,
    ResolvedProjectModel& result,
    SopAidError* error) {
    result = {};
    if (config.struct_size < sizeof(SopAidProjectDirectoryConfig) || config.api_version != 2) {
        SetError(error, SopAidStatus::InvalidArgument, "Invalid project directory config size or api_version.");
        return SopAidStatus::InvalidArgument;
    }
    if (!config.project_dir || std::string(config.project_dir).empty()) {
        SetError(error, SopAidStatus::InvalidArgument, "project_dir is required.");
        return SopAidStatus::InvalidArgument;
    }
    const fs::path root = fs::weakly_canonical(config.project_dir);
    cv::FileStorage project;
    if (!ReadJson(root / "project.json", project, error)) {
        return error ? error->status : SopAidStatus::InvalidArgument;
    }
    const auto project_id = project.root()["project_id"];
    if (!project_id.isString()) {
        SetError(error, SopAidStatus::InvalidArgument, "project_id is required and must be a string.");
        return SopAidStatus::InvalidArgument;
    }
    result.project_id = static_cast<std::string>(project_id);
    result.class_names_csv = ReadClasses(project.root()["classes"], error);
    if (result.class_names_csv.empty()) return SopAidStatus::InvalidArgument;

    const auto manifest = project.root()["active_model_manifest"];
    if (manifest.isString()) {
        if (ResolveManifest(root, ResolvePath(root, static_cast<std::string>(manifest)),
                config.preferred_model_format, result, error)) {
            SetError(error, SopAidStatus::Ok, "");
            return SopAidStatus::Ok;
        }
        return error ? error->status : SopAidStatus::UnsupportedModel;
    }
    const auto active_model = project.root()["active_model"];
    if (!active_model.isString()) {
        SetError(error, SopAidStatus::FileNotFound, "Project has no active model.");
        return SopAidStatus::FileNotFound;
    }
    if (ResolveAdjacent(ResolvePath(root, static_cast<std::string>(active_model)),
            config.preferred_model_format, result)) {
        SetError(error, SopAidStatus::Ok, "");
        return SopAidStatus::Ok;
    }
    SetError(error, SopAidStatus::UnsupportedModel,
        "No C++ artifact was found. Export ONNX, TorchScript, or TensorRT first.");
    return SopAidStatus::UnsupportedModel;
}
