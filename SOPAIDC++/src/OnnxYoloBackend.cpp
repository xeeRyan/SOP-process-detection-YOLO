#include "IYoloBackend.h"
#include "YoloPostprocess.h"

#include <onnxruntime_cxx_api.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <memory>

namespace {

struct LetterboxInfo {
    float scale = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
};

float ClampFloat(float value, float min_value, float max_value) {
    return std::max(min_value, std::min(value, max_value));
}

cv::Rect MakeClippedRect(float x1, float y1, float x2, float y2, const cv::Size& image_size) {
    x1 = ClampFloat(x1, 0.0f, static_cast<float>(image_size.width));
    y1 = ClampFloat(y1, 0.0f, static_cast<float>(image_size.height));
    x2 = ClampFloat(x2, 0.0f, static_cast<float>(image_size.width));
    y2 = ClampFloat(y2, 0.0f, static_cast<float>(image_size.height));

    const int left = static_cast<int>(std::round(std::min(x1, x2)));
    const int top = static_cast<int>(std::round(std::min(y1, y2)));
    const int right = static_cast<int>(std::round(std::max(x1, x2)));
    const int bottom = static_cast<int>(std::round(std::max(y1, y2)));
    return cv::Rect(left, top, std::max(0, right - left), std::max(0, bottom - top));
}

}  // namespace

class OnnxYoloBackend final : public IYoloBackend {
public:
    explicit OnnxYoloBackend(const SopAidInitConfig& config)
        : env_(ORT_LOGGING_LEVEL_WARNING, "SOPAID_ONNX"),
          conf_threshold_(config.confidence_threshold),
          class_names_(ParseClassNames(config.class_names_csv)) {
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        const std::wstring model_path = std::filesystem::path(config.model_path).wstring();
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ = session_->GetInputNameAllocated(0, allocator).get();
        output_name_ = session_->GetOutputNameAllocated(0, allocator).get();

        const auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (input_shape.size() == 4 && input_shape[2] > 0 && input_shape[3] > 0) {
            input_height_ = static_cast<int>(input_shape[2]);
            input_width_ = static_cast<int>(input_shape[3]);
        } else {
            input_width_ = config.input_width;
            input_height_ = config.input_height;
        }
    }

    SopAidStatus evaluate(const cv::Mat& image, std::vector<SopAidDetection>& results, SopAidError* error) override {
        results.clear();
        if (image.empty()) {
            SetError(error, SopAidStatus::InvalidArgument, "Evaluate image is empty.");
            return SopAidStatus::InvalidArgument;
        }

        try {
            LetterboxInfo letterbox;
            std::vector<float> input_tensor = preprocess(image, letterbox);
            std::array<int64_t, 4> input_shape = {1, 3, input_height_, input_width_};
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_value = Ort::Value::CreateTensor<float>(
                memory_info,
                input_tensor.data(),
                input_tensor.size(),
                input_shape.data(),
                input_shape.size());

            const char* input_names[] = {input_name_.c_str()};
            const char* output_names[] = {output_name_.c_str()};
            auto outputs = session_->Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_value,
                1,
                output_names,
                1);

            if (outputs.empty() || !outputs[0].IsTensor()) {
                SetError(error, SopAidStatus::InferenceError, "ONNX Runtime produced no tensor output.");
                return SopAidStatus::InferenceError;
            }

            parseOutput(outputs[0], image.size(), letterbox, results);
            SetError(error, SopAidStatus::Ok, "");
            return SopAidStatus::Ok;
        } catch (const Ort::Exception& ex) {
            SetError(error, SopAidStatus::InferenceError, ex.what());
            return SopAidStatus::InferenceError;
        } catch (const cv::Exception& ex) {
            SetError(error, SopAidStatus::InferenceError, ex.what());
            return SopAidStatus::InferenceError;
        }
    }

private:
    std::vector<float> preprocess(const cv::Mat& image, LetterboxInfo& letterbox) const {
        // Ultralytics YOLO 默认使用 letterbox：保持原图比例，空白区域填充 114。
        const float scale = std::min(
            static_cast<float>(input_width_) / static_cast<float>(image.cols),
            static_cast<float>(input_height_) / static_cast<float>(image.rows));
        const int resized_width = static_cast<int>(std::round(image.cols * scale));
        const int resized_height = static_cast<int>(std::round(image.rows * scale));
        letterbox.scale = scale;
        letterbox.pad_x = (static_cast<float>(input_width_ - resized_width)) / 2.0f;
        letterbox.pad_y = (static_cast<float>(input_height_ - resized_height)) / 2.0f;

        cv::Mat resized;
        cv::resize(image, resized, cv::Size(resized_width, resized_height));

        cv::Mat padded(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
        const int left = static_cast<int>(std::round(letterbox.pad_x));
        const int top = static_cast<int>(std::round(letterbox.pad_y));
        resized.copyTo(padded(cv::Rect(left, top, resized_width, resized_height)));
        cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);

        std::vector<float> tensor(static_cast<size_t>(3 * input_height_ * input_width_));
        const int channel_size = input_height_ * input_width_;
        for (int y = 0; y < input_height_; ++y) {
            const cv::Vec3b* row = padded.ptr<cv::Vec3b>(y);
            for (int x = 0; x < input_width_; ++x) {
                const int offset = y * input_width_ + x;
                tensor[offset] = row[x][0] / 255.0f;
                tensor[channel_size + offset] = row[x][1] / 255.0f;
                tensor[channel_size * 2 + offset] = row[x][2] / 255.0f;
            }
        }
        return tensor;
    }

    void parseOutput(
        Ort::Value& output,
        const cv::Size& image_size,
        const LetterboxInfo& letterbox,
        std::vector<SopAidDetection>& results) const {
        const float* data = output.GetTensorData<float>();
        const auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 3) {
            return;
        }

        // 当前导出的 YOLO ONNX 输出为 [1, 300, 6]：x1,y1,x2,y2,score,class_id。
        if (shape[2] == 6) {
            parsePostprocessedOutput(data, static_cast<int>(shape[1]), image_size, letterbox, results);
            return;
        }

        // 兼容转置后的 [1, 6, boxes] 输出。
        if (shape[1] == 6) {
            parseTransposedPostprocessedOutput(data, static_cast<int>(shape[2]), image_size, letterbox, results);
        }
    }

    cv::Rect restoreBox(float x1, float y1, float x2, float y2, const cv::Size& image_size, const LetterboxInfo& letterbox) const {
        // 将模型输入图上的坐标还原到原始视频帧坐标，并裁剪到图像范围内。
        x1 = (x1 - letterbox.pad_x) / letterbox.scale;
        y1 = (y1 - letterbox.pad_y) / letterbox.scale;
        x2 = (x2 - letterbox.pad_x) / letterbox.scale;
        y2 = (y2 - letterbox.pad_y) / letterbox.scale;
        return MakeClippedRect(x1, y1, x2, y2, image_size);
    }

    void parsePostprocessedOutput(
        const float* data,
        int rows,
        const cv::Size& image_size,
        const LetterboxInfo& letterbox,
        std::vector<SopAidDetection>& results) const {
        for (int row = 0; row < rows; ++row) {
            const float* item = data + row * 6;
            const float confidence = item[4];
            if (confidence < conf_threshold_) {
                continue;
            }

            const int class_id = static_cast<int>(item[5]);
            const cv::Rect box = restoreBox(item[0], item[1], item[2], item[3], image_size, letterbox);
            if (box.empty()) {
                continue;
            }
            results.push_back(MakeDetection(class_id, class_names_, confidence, box));
        }
    }

    void parseTransposedPostprocessedOutput(
        const float* data,
        int rows,
        const cv::Size& image_size,
        const LetterboxInfo& letterbox,
        std::vector<SopAidDetection>& results) const {
        for (int row = 0; row < rows; ++row) {
            const float confidence = data[4 * rows + row];
            if (confidence < conf_threshold_) {
                continue;
            }

            const int class_id = static_cast<int>(data[5 * rows + row]);
            const cv::Rect box = restoreBox(
                data[row],
                data[rows + row],
                data[2 * rows + row],
                data[3 * rows + row],
                image_size,
                letterbox);
            if (box.empty()) {
                continue;
            }
            results.push_back(MakeDetection(class_id, class_names_, confidence, box));
        }
    }

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
    int input_width_ = 640;
    int input_height_ = 640;
    float conf_threshold_ = 0.25f;
    std::vector<std::string> class_names_;
};

std::unique_ptr<IYoloBackend> CreateOnnxBackend(const SopAidInitConfig& config, SopAidError* error) {
    if (!config.model_path || !std::filesystem::exists(config.model_path)) {
        SetError(error, SopAidStatus::FileNotFound, "ONNX model file does not exist.");
        return nullptr;
    }
    try {
        auto backend = std::make_unique<OnnxYoloBackend>(config);
        SetError(error, SopAidStatus::Ok, "");
        return backend;
    } catch (const Ort::Exception& ex) {
        SetError(error, SopAidStatus::BackendError, ex.what());
        return nullptr;
    }
}
