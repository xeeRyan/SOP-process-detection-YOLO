#include "SopAidInfer.h"

#include "IYoloBackend.h"
#include "ProjectModelResolver.h"
#include "YoloPostprocess.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <cstring>
#include <memory>
#include <mutex>
#include <utility>

namespace {

struct ModelContext {
    ModelContext(
        std::unique_ptr<IYoloBackend> backend_in,
        const SopAidInitConfig& config_in,
        SopAidResizeMode resize_mode_in,
        const char* project_id_in,
        const char* model_id_in,
        const char* model_version_in)
        : backend(std::move(backend_in)),
          model_path(config_in.model_path ? config_in.model_path : ""),
          project_id(project_id_in ? project_id_in : ""),
          model_id(model_id_in ? model_id_in : ""),
          model_version(model_version_in ? model_version_in : ""),
          input_width(config_in.input_width),
          input_height(config_in.input_height),
          resize_mode(resize_mode_in) {
        class_count = static_cast<int32_t>(ParseClassNames(config_in.class_names_csv).size());
    }

    // 后端对象持有模型状态；一个句柄只绑定一个后端，并用锁串行化推理调用。
    std::unique_ptr<IYoloBackend> backend;
    std::mutex mutex;
    std::string model_path;
    std::string project_id;
    std::string model_id;
    std::string model_version;
    int32_t input_width = 0;
    int32_t input_height = 0;
    int32_t class_count = 0;
    SopAidResizeMode resize_mode = SopAidResizeMode::Letterbox;
};

// 初始化前统一校验参数，避免各后端重复处理空路径、文件不存在等基础错误。
SopAidStatus validateConfig(const SopAidInitConfig& config, SopAidError* error) {
    if (!config.model_path || std::string(config.model_path).empty()) {
        SetError(error, SopAidStatus::InvalidArgument, "model_path is empty.");
        return SopAidStatus::InvalidArgument;
    }
    if (!std::filesystem::exists(config.model_path)) {
        SetError(error, SopAidStatus::FileNotFound, std::string("Model file does not exist: ") + config.model_path);
        return SopAidStatus::FileNotFound;
    }
    if (config.input_width <= 0 || config.input_height <= 0) {
        SetError(error, SopAidStatus::InvalidArgument, "input_width and input_height must be positive.");
        return SopAidStatus::InvalidArgument;
    }
    if (!config.class_names_csv || ParseClassNames(config.class_names_csv).empty()) {
        SetError(
            error,
            SopAidStatus::InvalidArgument,
            "class_names_csv is required and must follow project.json class id order.");
        return SopAidStatus::InvalidArgument;
    }
    return SopAidStatus::Ok;
}

// 支持调用方显式指定格式，也支持根据文件后缀自动识别。
SopAidModelFormat resolveFormat(const SopAidInitConfig& config) {
    return config.model_format == SopAidModelFormat::Auto ? DetectModelFormat(config.model_path) : config.model_format;
}

std::unique_ptr<IYoloBackend> createBackend(const SopAidInitConfig& config, SopAidError* error) {
    const SopAidStatus status = validateConfig(config, error);
    if (status != SopAidStatus::Ok) {
        return nullptr;
    }

    // 三种模型只在后端创建阶段不同；对外的句柄和 Evaluate 调用保持一致。
    switch (resolveFormat(config)) {
    case SopAidModelFormat::Onnx:
        return CreateOnnxBackend(config, error);
    case SopAidModelFormat::Engine:
        return CreateTensorRtBackend(config, error);
    case SopAidModelFormat::Pt:
        return CreateTorchScriptBackend(config, error);
    default:
        SetError(error, SopAidStatus::UnsupportedModel, "Unsupported model extension or model_format.");
        return nullptr;
    }
}

// Init 的核心实现：创建后端并包装成内部模型上下文。
ModelContext* initWithConfig(
    const SopAidInitConfig& config,
    SopAidResizeMode resize_mode,
    const char* project_id,
    const char* model_id,
    const char* model_version,
    SopAidError* error) {
    auto backend = createBackend(config, error);
    if (!backend) {
        return nullptr;
    }

    auto* context = new ModelContext(
        std::move(backend), config, resize_mode, project_id, model_id, model_version);
    SetError(error, SopAidStatus::Ok, "");
    return context;
}

// Evaluate 的核心实现：使用模型上下文对单帧 cv::Mat 做推理。
SopAidStatus runEvaluate(
    ModelContext* context,
    const cv::Mat& image,
    std::vector<SopAidDetection>& detections,
    SopAidError* error) {
    detections.clear();
    if (!context) {
        SetError(error, SopAidStatus::InvalidArgument, "Invalid Evaluate input.");
        return SopAidStatus::InvalidArgument;
    }

    // 同一个推理对象内部串行化，避免多个线程同时访问同一后端上下文。
    std::lock_guard<std::mutex> lock(context->mutex);
    const SopAidStatus status = context->backend->evaluate(image, detections, error);
    if (status != SopAidStatus::Ok) return status;

    // 与 Python target_classes 行为一致：项目未声明的模型类别不进入跟踪和 SOP 状态机。
    detections.erase(
        std::remove_if(
            detections.begin(),
            detections.end(),
            [context](const SopAidDetection& detection) {
                return detection.class_id < 0 || detection.class_id >= context->class_count;
            }),
        detections.end());
    return SopAidStatus::Ok;
}

template <size_t N>
void copyText(char (&target)[N], const std::string& value) {
    strncpy_s(target, N, value.c_str(), _TRUNCATE);
}

}  // namespace

namespace sopaid {

Inference::~Inference() {
    Release();
}

SopAidStatus Inference::Init(const SopAidInitConfig& config, SopAidError* error) {
    Release();

    SopAidError local_error;
    SopAidError* actual_error = error ? error : &local_error;
    context_ = initWithConfig(
        config, SopAidResizeMode::Letterbox, nullptr, nullptr, nullptr, actual_error);
    return context_ ? SopAidStatus::Ok : actual_error->status;
}

SopAidStatus Inference::InitProject(
    const SopAidProjectInitConfig& config,
    SopAidError* error) {
    Release();
    if (config.struct_size < sizeof(SopAidProjectInitConfig) || config.api_version != 2) {
        SetError(error, SopAidStatus::InvalidArgument, "Invalid project config size or api_version.");
        return SopAidStatus::InvalidArgument;
    }
    if (config.resize_mode != SopAidResizeMode::Letterbox) {
        SetError(error, SopAidStatus::InvalidArgument, "Only Letterbox resize_mode is currently supported.");
        return SopAidStatus::InvalidArgument;
    }
    if (!config.project_id || std::string(config.project_id).empty()) {
        SetError(error, SopAidStatus::InvalidArgument, "project_id is required for InitProject.");
        return SopAidStatus::InvalidArgument;
    }

    SopAidError local_error;
    SopAidError* actual_error = error ? error : &local_error;
    context_ = initWithConfig(
        config.inference,
        config.resize_mode,
        config.project_id,
        config.model_id,
        config.model_version,
        actual_error);
    return context_ ? SopAidStatus::Ok : actual_error->status;
}

SopAidStatus Inference::InitProjectDirectory(
    const SopAidProjectDirectoryConfig& config,
    SopAidError* error) {
    Release();
    ResolvedProjectModel resolved;
    const SopAidStatus status = ResolveProjectModel(config, resolved, error);
    if (status != SopAidStatus::Ok) return status;

    SopAidProjectInitConfig project_config;
    project_config.inference.model_path = resolved.model_path.c_str();
    project_config.inference.model_format = resolved.model_format;
    project_config.inference.input_width = config.input_width;
    project_config.inference.input_height = config.input_height;
    project_config.inference.confidence_threshold = config.confidence_threshold;
    project_config.inference.nms_threshold = config.nms_threshold;
    project_config.inference.class_names_csv = resolved.class_names_csv.c_str();
    project_config.inference.use_cuda = config.use_cuda;
    project_config.inference.device_id = config.device_id;
    project_config.project_id = resolved.project_id.c_str();
    project_config.model_id = resolved.model_id.empty() ? nullptr : resolved.model_id.c_str();
    project_config.model_version = resolved.model_version.empty() ? nullptr : resolved.model_version.c_str();
    return InitProject(project_config, error);
}

SopAidStatus Inference::Evaluate(
    const cv::Mat& image,
    std::vector<SopAidDetection>& results,
    SopAidError* error) {
    if (!context_) {
        results.clear();
        SetError(error, SopAidStatus::InvalidArgument, "Inference is not initialized.");
        return SopAidStatus::InvalidArgument;
    }
    return runEvaluate(static_cast<ModelContext*>(context_), image, results, error);
}

SopAidStatus Inference::GetModelInfo(SopAidModelInfo& info, SopAidError* error) const {
    info = {};
    if (!context_) {
        SetError(error, SopAidStatus::InvalidArgument, "Inference is not initialized.");
        return SopAidStatus::InvalidArgument;
    }
    const auto* context = static_cast<const ModelContext*>(context_);
    copyText(info.project_id, context->project_id);
    copyText(info.model_id, context->model_id);
    copyText(info.model_version, context->model_version);
    copyText(info.model_path, context->model_path);
    copyText(info.backend, context->backend->backendName());
    info.input_width = context->input_width;
    info.input_height = context->input_height;
    info.class_count = context->class_count;
    info.resize_mode = context->resize_mode;
    SetError(error, SopAidStatus::Ok, "");
    return SopAidStatus::Ok;
}

void Inference::Release() {
    if (context_) {
        delete static_cast<ModelContext*>(context_);
        context_ = nullptr;
    }
}

}  // namespace sopaid
