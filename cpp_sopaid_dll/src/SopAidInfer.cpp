#include "SopAidInfer.h"

#include "IYoloBackend.h"
#include "YoloPostprocess.h"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <mutex>
#include <utility>

#include <opencv2/videoio.hpp>

namespace {

struct ModelContext {
    explicit ModelContext(std::unique_ptr<IYoloBackend> backend_in) : backend(std::move(backend_in)) {}

    // 鍚庣瀵硅薄鎸佹湁妯″瀷鐘舵€侊紱涓€涓彞鏌勫彧缁戝畾涓€涓悗绔紝骞剁敤閿佷覆琛屽寲鎺ㄧ悊璋冪敤銆?    
    std::unique_ptr<IYoloBackend> backend;
    std::mutex mutex;
};

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
    if (config.confidence_threshold < 0.0f || config.confidence_threshold > 1.0f ||
        config.nms_threshold < 0.0f || config.nms_threshold > 1.0f) {
        SetError(error, SopAidStatus::InvalidArgument, "confidence_threshold and nms_threshold must be in [0, 1].");
        return SopAidStatus::InvalidArgument;
    }
    if (config.device_id < 0) {
        SetError(error, SopAidStatus::InvalidArgument, "device_id must be non-negative.");
        return SopAidStatus::InvalidArgument;
    }

    return SopAidStatus::Ok;
}

SopAidModelFormat resolveFormat(const SopAidInitConfig& config) {
    return config.model_format == SopAidModelFormat::Auto ? DetectModelFormat(config.model_path) : config.model_format;
}

std::unique_ptr<IYoloBackend> createBackend(const SopAidInitConfig& config, SopAidError* error) {
    const SopAidStatus status = validateConfig(config, error);
    if (status != SopAidStatus::Ok) {
        return nullptr;
    }

    // 涓夌妯″瀷鍙湪鍚庣鍒涘缓闃舵涓嶅悓锛涘澶栫殑鍙ユ焺鍜?Evaluate 璋冪敤淇濇寔涓€鑷淬€?    
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

SopAidHandle initWithConfig(const SopAidInitConfig& config, SopAidError* error) {
    auto backend = createBackend(config, error);
    if (!backend) {
        return nullptr;
    }

    auto* context = new ModelContext(std::move(backend));
    SetError(error, SopAidStatus::Ok, "");
    return static_cast<SopAidHandle>(context);
}

SopAidHandle initWithFormat(const SopAidInitConfig* config, SopAidModelFormat format, SopAidError* error) {
    if (!config) {
        SetError(error, SopAidStatus::InvalidArgument, "config is null.");
        return nullptr;
    }

    SopAidInitConfig typed_config = *config;
    typed_config.model_format = format;
    return initWithConfig(typed_config, error);
}

SopAidStatus runEvaluate(const SopAidEvaluateInput& input, std::vector<SopAidDetection>& detections, SopAidError* error) {
    detections.clear();
    if (!input.handle || !input.image) {
        SetError(error, SopAidStatus::InvalidArgument, "Invalid Evaluate input.");
        return SopAidStatus::InvalidArgument;
    }

    auto* context = static_cast<ModelContext*>(input.handle);
    std::lock_guard<std::mutex> lock(context->mutex);
    return context->backend->evaluate(*input.image, detections, error);
}

SopAidStatus copyDetectionsToOutput(
    const std::vector<SopAidDetection>& detections,
    SopAidEvaluateOutput& output,
    SopAidError* error) {
    output.result_count = 0;
    if (!output.results || output.result_capacity < 0) {
        SetError(error, SopAidStatus::InvalidArgument, "Invalid Evaluate output.");
        return SopAidStatus::InvalidArgument;
    }

    const int32_t copy_count = std::min<int32_t>(static_cast<int32_t>(detections.size()), output.result_capacity);
    for (int32_t i = 0; i < copy_count; ++i) {
        output.results[i] = detections[static_cast<size_t>(i)];
    }
    output.result_count = copy_count;

    if (copy_count < static_cast<int32_t>(detections.size())) {
        SetError(error, SopAidStatus::InvalidArgument, "Result buffer capacity is smaller than detection count.");
        return SopAidStatus::InvalidArgument;
    }

    SetError(error, SopAidStatus::Ok, "");
    return SopAidStatus::Ok;
}

SopAidStatus validateVideoInput(const SopAidVideoEvaluateInput& input, SopAidError* error) {
    if (!input.handle || !input.video_path || std::string(input.video_path).empty()) {
        SetError(error, SopAidStatus::InvalidArgument, "Invalid video Evaluate input.");
        return SopAidStatus::InvalidArgument;
    }
    if (!std::filesystem::exists(input.video_path)) {
        SetError(error, SopAidStatus::FileNotFound, std::string("Video file does not exist: ") + input.video_path);
        return SopAidStatus::FileNotFound;
    }
    if (input.frame_step <= 0) {
        SetError(error, SopAidStatus::InvalidArgument, "frame_step must be positive.");
        return SopAidStatus::InvalidArgument;
    }

    return SopAidStatus::Ok;
}

SopAidStatus runEvaluateVideo(
    const SopAidVideoEvaluateInput& input,
    std::vector<SopAidVideoDetection>& video_detections,
    SopAidError* error) {
    video_detections.clear();

    const SopAidStatus input_status = validateVideoInput(input, error);
    if (input_status != SopAidStatus::Ok) {
        return input_status;
    }

    cv::VideoCapture capture(input.video_path);
    if (!capture.isOpened()) {
        SetError(error, SopAidStatus::InvalidArgument, std::string("Failed to open video: ") + input.video_path);
        return SopAidStatus::InvalidArgument;
    }

    const double fps = capture.get(cv::CAP_PROP_FPS);
    int64_t frame_index = 0;
    int32_t processed_frames = 0;
    cv::Mat frame;

    while (capture.read(frame)) {
        if (frame_index % input.frame_step == 0) {
            SopAidEvaluateInput frame_input;
            frame_input.handle = input.handle;
            frame_input.image = &frame;

            std::vector<SopAidDetection> frame_detections;
            const SopAidStatus status = runEvaluate(frame_input, frame_detections, error);
            if (status != SopAidStatus::Ok) {
                return status;
            }

            const double time_sec = fps > 0.0 ? static_cast<double>(frame_index) / fps : 0.0;
            for (const auto& detection : frame_detections) {
                SopAidVideoDetection video_detection;
                video_detection.frame_index = frame_index;
                video_detection.time_sec = time_sec;
                video_detection.detection = detection;
                video_detections.push_back(video_detection);
            }

            ++processed_frames;
            if (input.max_frames > 0 && processed_frames >= input.max_frames) {
                break;
            }
        }

        ++frame_index;
    }

    SetError(error, SopAidStatus::Ok, "");
    return SopAidStatus::Ok;
}

SopAidStatus copyVideoDetectionsToOutput(
    const std::vector<SopAidVideoDetection>& detections,
    SopAidVideoEvaluateOutput& output,
    SopAidError* error) {
    output.result_count = 0;
    if (!output.results || output.result_capacity < 0) {
        SetError(error, SopAidStatus::InvalidArgument, "Invalid video Evaluate output.");
        return SopAidStatus::InvalidArgument;
    }

    const int32_t copy_count = std::min<int32_t>(static_cast<int32_t>(detections.size()), output.result_capacity);
    for (int32_t i = 0; i < copy_count; ++i) {
        output.results[i] = detections[static_cast<size_t>(i)];
    }
    output.result_count = copy_count;

    if (copy_count < static_cast<int32_t>(detections.size())) {
        SetError(error, SopAidStatus::InvalidArgument, "Video result buffer capacity is smaller than detection count.");
        return SopAidStatus::InvalidArgument;
    }

    SetError(error, SopAidStatus::Ok, "");
    return SopAidStatus::Ok;
}

}  // namespace

extern "C" SOPAID_API SopAidHandle SopAid_Init(const SopAidInitConfig* config, SopAidError* error) {
    return initWithFormat(config, config ? config->model_format : SopAidModelFormat::Auto, error);
}

extern "C" SOPAID_API SopAidHandle SopAid_InitPt(const SopAidInitConfig* config, SopAidError* error) {
    return initWithFormat(config, SopAidModelFormat::Pt, error);
}

extern "C" SOPAID_API SopAidHandle SopAid_InitOnnx(const SopAidInitConfig* config, SopAidError* error) {
    return initWithFormat(config, SopAidModelFormat::Onnx, error);
}

extern "C" SOPAID_API SopAidHandle SopAid_InitEngine(const SopAidInitConfig* config, SopAidError* error) {
    return initWithFormat(config, SopAidModelFormat::Engine, error);
}

extern "C" SOPAID_API SopAidStatus SopAid_Evaluate(
    SopAidHandle handle,
    const cv::Mat* image,
    SopAidDetection* results,
    int32_t result_capacity,
    int32_t* result_count,
    SopAidError* error) {
    if (!result_count) {
        SetError(error, SopAidStatus::InvalidArgument, "result_count is null.");
        return SopAidStatus::InvalidArgument;
    }

    SopAidEvaluateInput input;
    input.handle = handle;
    input.image = image;

    SopAidEvaluateOutput output;
    output.results = results;
    output.result_capacity = result_capacity;
    output.result_count = 0;

    const SopAidStatus status = SopAid_EvaluateStruct(&input, &output, error);
    *result_count = output.result_count;
    return status;
}

extern "C" SOPAID_API SopAidStatus SopAid_EvaluateStruct(
    const SopAidEvaluateInput* input,
    SopAidEvaluateOutput* output,
    SopAidError* error) {
    if (output) {
        output->result_count = 0;
    }
    if (!input || !output) {
        SetError(error, SopAidStatus::InvalidArgument, "Invalid Evaluate argument.");
        return SopAidStatus::InvalidArgument;
    }

    std::vector<SopAidDetection> detections;
    const SopAidStatus status = runEvaluate(*input, detections, error);
    if (status != SopAidStatus::Ok) {
        return status;
    }

    return copyDetectionsToOutput(detections, *output, error);
}

extern "C" SOPAID_API SopAidStatus SopAid_EvaluateVideoStruct(
    const SopAidVideoEvaluateInput* input,
    SopAidVideoEvaluateOutput* output,
    SopAidError* error) {
    if (output) {
        output->result_count = 0;
    }
    if (!input || !output) {
        SetError(error, SopAidStatus::InvalidArgument, "Invalid video Evaluate argument.");
        return SopAidStatus::InvalidArgument;
    }

    std::vector<SopAidVideoDetection> detections;
    const SopAidStatus status = runEvaluateVideo(*input, detections, error);
    if (status != SopAidStatus::Ok) {
        return status;
    }

    return copyVideoDetectionsToOutput(detections, *output, error);
}

extern "C" SOPAID_API void SopAid_Release(SopAidHandle handle) {
    delete static_cast<ModelContext*>(handle);
}

namespace sopaid {

SOPAID_API SopAidHandle Init(const SopAidInitConfig& config, SopAidError* error) {
    return SopAid_Init(&config, error);
}

SOPAID_API SopAidHandle InitPt(const SopAidInitConfig& config, SopAidError* error) {
    return SopAid_InitPt(&config, error);
}

SOPAID_API SopAidHandle InitOnnx(const SopAidInitConfig& config, SopAidError* error) {
    return SopAid_InitOnnx(&config, error);
}

SOPAID_API SopAidHandle InitEngine(const SopAidInitConfig& config, SopAidError* error) {
    return SopAid_InitEngine(&config, error);
}

SOPAID_API SopAidStatus Evaluate(
    const SopAidEvaluateInput& input,
    SopAidEvaluateResult& output,
    SopAidError* error) {
    output.results.clear();
    return runEvaluate(input, output.results, error);
}

SOPAID_API SopAidStatus Evaluate(
    const SopAidEvaluateInput& input,
    SopAidEvaluateOutput& output,
    SopAidError* error) {
    return SopAid_EvaluateStruct(&input, &output, error);
}

SOPAID_API SopAidStatus Evaluate(
    SopAidHandle handle,
    const cv::Mat& image,
    std::vector<SopAidDetection>& results,
    SopAidError* error) {
    SopAidEvaluateInput input;
    input.handle = handle;
    input.image = &image;

    SopAidEvaluateResult output;
    const SopAidStatus status = Evaluate(input, output, error);
    results = std::move(output.results);
    return status;
}

SOPAID_API SopAidStatus EvaluateVideo(
    const SopAidVideoEvaluateInput& input,
    SopAidVideoEvaluateResult& output,
    SopAidError* error) {
    output.results.clear();
    return runEvaluateVideo(input, output.results, error);
}

SOPAID_API SopAidStatus EvaluateVideo(
    const SopAidVideoEvaluateInput& input,
    SopAidVideoEvaluateOutput& output,
    SopAidError* error) {
    return SopAid_EvaluateVideoStruct(&input, &output, error);
}

SOPAID_API void Release(SopAidHandle handle) {
    SopAid_Release(handle);
}

}  // namespace sopaid

