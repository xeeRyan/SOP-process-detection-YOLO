#include "IYoloBackend.h"
#include "YoloPostprocess.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>

#ifdef SOPAID_ENABLE_LIBTORCH
#include <torch/script.h>
#endif
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

cv::Rect RestoreBox(float x1, float y1, float x2, float y2, const cv::Size& image_size, const LetterboxInfo& letterbox) {
    x1 = (x1 - letterbox.pad_x) / letterbox.scale;
    y1 = (y1 - letterbox.pad_y) / letterbox.scale;
    x2 = (x2 - letterbox.pad_x) / letterbox.scale;
    y2 = (y2 - letterbox.pad_y) / letterbox.scale;
    return MakeClippedRect(x1, y1, x2, y2, image_size);
}

std::vector<float> PreprocessLetterbox(const cv::Mat& image, int input_width, int input_height, LetterboxInfo& letterbox) {
    const float scale = std::min(
        static_cast<float>(input_width) / static_cast<float>(image.cols),
        static_cast<float>(input_height) / static_cast<float>(image.rows));
    const int resized_width = static_cast<int>(std::round(image.cols * scale));
    const int resized_height = static_cast<int>(std::round(image.rows * scale));
    letterbox.scale = scale;
    letterbox.pad_x = (static_cast<float>(input_width - resized_width)) / 2.0f;
    letterbox.pad_y = (static_cast<float>(input_height - resized_height)) / 2.0f;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_width, resized_height));

    cv::Mat padded(input_height, input_width, CV_8UC3, cv::Scalar(114, 114, 114));
    const int left = static_cast<int>(std::round(letterbox.pad_x));
    const int top = static_cast<int>(std::round(letterbox.pad_y));
    resized.copyTo(padded(cv::Rect(left, top, resized_width, resized_height)));
    cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);

    std::vector<float> tensor(static_cast<size_t>(3 * input_height * input_width));
    const int channel_size = input_height * input_width;
    for (int y = 0; y < input_height; ++y) {
        const cv::Vec3b* row = padded.ptr<cv::Vec3b>(y);
        for (int x = 0; x < input_width; ++x) {
            const int offset = y * input_width + x;
            tensor[offset] = row[x][0] / 255.0f;
            tensor[channel_size + offset] = row[x][1] / 255.0f;
            tensor[channel_size * 2 + offset] = row[x][2] / 255.0f;
        }
    }
    return tensor;
}

void ParsePostprocessedRows(
    const float* data,
    int rows,
    int stride,
    const cv::Size& image_size,
    const LetterboxInfo& letterbox,
    float conf_threshold,
    const std::vector<std::string>& class_names,
    std::vector<SopAidDetection>& results) {
    for (int row = 0; row < rows; ++row) {
        const float* item = data + row * stride;
        const float confidence = item[4];
        if (confidence < conf_threshold) {
            continue;
        }

        const int class_id = static_cast<int>(item[5]);
        const cv::Rect box = RestoreBox(item[0], item[1], item[2], item[3], image_size, letterbox);
        if (!box.empty()) {
            results.push_back(MakeDetection(class_id, class_names, confidence, box));
        }
    }
}

}  // namespace

#ifdef SOPAID_ENABLE_LIBTORCH

class TorchScriptYoloBackend final : public IYoloBackend {
public:
    explicit TorchScriptYoloBackend(const SopAidInitConfig& config)
        : input_width_(config.input_width),
          input_height_(config.input_height),
          conf_threshold_(config.confidence_threshold),
          class_names_(ParseClassNames(config.class_names_csv)) {
        module_ = torch::jit::load(config.model_path);
        module_.eval();
    }

    SopAidStatus evaluate(const cv::Mat& image, std::vector<SopAidDetection>& results, SopAidError* error) override {
        results.clear();
        if (image.empty()) {
            SetError(error, SopAidStatus::InvalidArgument, "Evaluate image is empty.");
            return SopAidStatus::InvalidArgument;
        }

        try {
            LetterboxInfo letterbox;
            std::vector<float> input_data = PreprocessLetterbox(image, input_width_, input_height_, letterbox);
            torch::Tensor input_tensor = torch::from_blob(
                                             input_data.data(),
                                             {1, 3, input_height_, input_width_},
                                             torch::TensorOptions().dtype(torch::kFloat32))
                                             .clone();

            torch::NoGradGuard no_grad;
            torch::IValue output_value = module_.forward({input_tensor});
            torch::Tensor output_tensor = unpackOutputTensor(output_value);
            output_tensor = output_tensor.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
            parseOutput(output_tensor, image.size(), letterbox, results);

            SetError(error, SopAidStatus::Ok, "");
            return SopAidStatus::Ok;
        } catch (const c10::Error& ex) {
            SetError(error, SopAidStatus::InferenceError, ex.what_without_backtrace());
            return SopAidStatus::InferenceError;
        } catch (const cv::Exception& ex) {
            SetError(error, SopAidStatus::InferenceError, ex.what());
            return SopAidStatus::InferenceError;
        }
    }

private:
    torch::Tensor unpackOutputTensor(const torch::IValue& value) const {
        if (value.isTensor()) {
            return value.toTensor();
        }
        if (value.isTuple()) {
            const auto elements = value.toTuple()->elements();
            if (!elements.empty() && elements[0].isTensor()) {
                return elements[0].toTensor();
            }
        }
        if (value.isList()) {
            const auto list = value.toList();
            if (list.size() > 0 && list.get(0).isTensor()) {
                return list.get(0).toTensor();
            }
        }
        throw std::runtime_error("TorchScript output is not a tensor, tuple[0] tensor, or list[0] tensor.");
    }

    void parseOutput(
        const torch::Tensor& output,
        const cv::Size& image_size,
        const LetterboxInfo& letterbox,
        std::vector<SopAidDetection>& results) const {
        const auto sizes = output.sizes();
        const float* data = output.data_ptr<float>();

        if (sizes.size() == 3 && sizes[2] == 6) {
            ParsePostprocessedRows(data, static_cast<int>(sizes[1]), 6, image_size, letterbox, conf_threshold_, class_names_, results);
            return;
        }
        if (sizes.size() == 2 && sizes[1] == 6) {
            ParsePostprocessedRows(data, static_cast<int>(sizes[0]), 6, image_size, letterbox, conf_threshold_, class_names_, results);
            return;
        }
        if (sizes.size() == 3 && sizes[1] == 6) {
            const int rows = static_cast<int>(sizes[2]);
            for (int row = 0; row < rows; ++row) {
                const float confidence = data[4 * rows + row];
                if (confidence < conf_threshold_) {
                    continue;
                }
                const int class_id = static_cast<int>(data[5 * rows + row]);
                const cv::Rect box = RestoreBox(
                    data[row], data[rows + row], data[2 * rows + row], data[3 * rows + row], image_size, letterbox);
                if (!box.empty()) {
                    results.push_back(MakeDetection(class_id, class_names_, confidence, box));
                }
            }
        }
    }

    torch::jit::script::Module module_;
    int input_width_ = 640;
    int input_height_ = 640;
    float conf_threshold_ = 0.25f;
    std::vector<std::string> class_names_;
};

std::unique_ptr<IYoloBackend> CreateTorchScriptBackend(const SopAidInitConfig& config, SopAidError* error) {
    if (!config.model_path || !std::filesystem::exists(config.model_path)) {
        SetError(error, SopAidStatus::FileNotFound, "PT/TorchScript model file does not exist.");
        return nullptr;
    }
    try {
        auto backend = std::make_unique<TorchScriptYoloBackend>(config);
        SetError(error, SopAidStatus::Ok, "");
        return backend;
    } catch (const c10::Error& ex) {
        SetError(error, SopAidStatus::BackendError, ex.what_without_backtrace());
        return nullptr;
    }
}

#else

std::unique_ptr<IYoloBackend> CreateTorchScriptBackend(const SopAidInitConfig& config, SopAidError* error) {
    (void)config;
    SetError(
        error,
        SopAidStatus::UnsupportedModel,
        "PT/TorchScript backend is not enabled. Set EnableLibTorch=true and configure LibTorchRoot in SopAidInfer.user.props.");
    return nullptr;
}

#endif

