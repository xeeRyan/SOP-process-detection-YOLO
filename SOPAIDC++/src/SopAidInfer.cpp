#include "SopAidInfer.h"

#include "IYoloBackend.h"
#include "YoloPostprocess.h"

#include <filesystem>
#include <memory>
#include <mutex>
#include <utility>

namespace {

struct ModelContext {
    explicit ModelContext(std::unique_ptr<IYoloBackend> backend_in) : backend(std::move(backend_in)) {}

    // 后端对象持有模型状态；一个句柄只绑定一个后端，并用锁串行化推理调用。
    std::unique_ptr<IYoloBackend> backend;
    std::mutex mutex;
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
ModelContext* initWithConfig(const SopAidInitConfig& config, SopAidError* error) {
    auto backend = createBackend(config, error);
    if (!backend) {
        return nullptr;
    }

    auto* context = new ModelContext(std::move(backend));
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
    return context->backend->evaluate(image, detections, error);
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
    context_ = initWithConfig(config, actual_error);
    return context_ ? SopAidStatus::Ok : actual_error->status;
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

void Inference::Release() {
    if (context_) {
        delete static_cast<ModelContext*>(context_);
        context_ = nullptr;
    }
}

}  // namespace sopaid
