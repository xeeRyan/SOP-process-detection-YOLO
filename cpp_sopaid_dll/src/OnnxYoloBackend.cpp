#include "IYoloBackend.h"
#include "YoloPostprocess.h"

#include <onnxruntime_cxx_api.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <numeric>

class OnnxYoloBackend final : public IYoloBackend {
public:
    explicit OnnxYoloBackend(const SopAidInitConfig& config)
        : env_(ORT_LOGGING_LEVEL_WARNING, "SOPAID_ONNX"),
          conf_threshold_(config.confidence_threshold),
          nms_threshold_(config.nms_threshold),
          class_names_(ParseClassNames(config.class_names_csv)) {
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        if (config.use_cuda) {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = config.device_id;
            session_options_.AppendExecutionProvider_CUDA(cuda_options);
        }

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
            std::vector<float> input_tensor = preprocess(image);
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

            parseOutput(outputs[0], image.size(), results);
            ApplyClasswiseNms(results, conf_threshold_, nms_threshold_);
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
    std::vector<float> preprocess(const cv::Mat& image) const {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(input_width_, input_height_));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

        std::vector<float> tensor(static_cast<size_t>(3 * input_height_ * input_width_));
        const int channel_size = input_height_ * input_width_;
        for (int y = 0; y < input_height_; ++y) {
            const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);
            for (int x = 0; x < input_width_; ++x) {
                const int offset = y * input_width_ + x;
                tensor[offset] = row[x][0] / 255.0f;
                tensor[channel_size + offset] = row[x][1] / 255.0f;
                tensor[channel_size * 2 + offset] = row[x][2] / 255.0f;
            }
        }
        return tensor;
    }

    void parseOutput(Ort::Value& output, const cv::Size& image_size, std::vector<SopAidDetection>& results) const {
        const float* data = output.GetTensorData<float>();
        const auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 3) {
            return;
        }

        // 当前模型输出为 [1, 300, 6]，每行含义是 x1,y1,x2,y2,score,class_id。
        if (shape[2] == 6) {
            parsePostprocessedOutput(data, static_cast<int>(shape[1]), image_size, results);
            return;
        }

        // 兼容未带后处理的 YOLO 输出：[1, boxes, attrs] 或 [1, attrs, boxes]。
        if (shape[1] == 6) {
            parseTransposedPostprocessedOutput(data, static_cast<int>(shape[2]), image_size, results);
        }
    }

    void parsePostprocessedOutput(
        const float* data,
        int rows,
        const cv::Size& image_size,
        std::vector<SopAidDetection>& results) const {
        const float x_scale = static_cast<float>(image_size.width) / static_cast<float>(input_width_);
        const float y_scale = static_cast<float>(image_size.height) / static_cast<float>(input_height_);

        for (int row = 0; row < rows; ++row) {
            const float* item = data + row * 6;
            const float confidence = item[4];
            if (confidence < conf_threshold_) {
                continue;
            }

            const int class_id = static_cast<int>(item[5]);
            const int x1 = static_cast<int>(item[0] * x_scale);
            const int y1 = static_cast<int>(item[1] * y_scale);
            const int x2 = static_cast<int>(item[2] * x_scale);
            const int y2 = static_cast<int>(item[3] * y_scale);
            const cv::Rect box(x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1));
            results.push_back(MakeDetection(class_id, class_names_, confidence, box));
        }
    }

    void parseTransposedPostprocessedOutput(
        const float* data,
        int rows,
        const cv::Size& image_size,
        std::vector<SopAidDetection>& results) const {
        const float x_scale = static_cast<float>(image_size.width) / static_cast<float>(input_width_);
        const float y_scale = static_cast<float>(image_size.height) / static_cast<float>(input_height_);

        for (int row = 0; row < rows; ++row) {
            const float confidence = data[4 * rows + row];
            if (confidence < conf_threshold_) {
                continue;
            }

            const int class_id = static_cast<int>(data[5 * rows + row]);
            const int x1 = static_cast<int>(data[row] * x_scale);
            const int y1 = static_cast<int>(data[rows + row] * y_scale);
            const int x2 = static_cast<int>(data[2 * rows + row] * x_scale);
            const int y2 = static_cast<int>(data[3 * rows + row] * y_scale);
            const cv::Rect box(x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1));
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
    float nms_threshold_ = 0.70f;
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
