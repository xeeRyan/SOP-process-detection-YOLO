#include "IYoloBackend.h"
#include "YoloPostprocess.h"

#include <torch/script.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <vector>

namespace {

torch::Tensor ExtractTensor(const torch::jit::IValue& value) {
    if (value.isTensor()) {
        return value.toTensor();
    }
    if (value.isTuple()) {
        for (const auto& item : value.toTuple()->elements()) {
            if (item.isTensor()) {
                return item.toTensor();
            }
        }
    }
    if (value.isList()) {
        for (const auto& item : value.toListRef()) {
            if (item.isTensor()) {
                return item.toTensor();
            }
        }
    }
    throw std::runtime_error("TorchScript output does not contain a tensor.");
}

class TorchScriptYoloBackend final : public IYoloBackend {
public:
    explicit TorchScriptYoloBackend(const SopAidInitConfig& config)
        : input_width_(config.input_width),
          input_height_(config.input_height),
          confidence_threshold_(config.confidence_threshold),
          nms_threshold_(config.nms_threshold),
          class_names_(ParseClassNames(config.class_names_csv)) {
        if (config.use_cuda) {
            throw std::runtime_error(
                "This LibTorch installation is CPU-only. Install CUDA LibTorch before setting use_cuda=true.");
        }
        module_ = torch::jit::load(config.model_path, torch::kCPU);
        module_.eval();
    }

    SopAidStatus evaluate(
        const cv::Mat& image,
        std::vector<SopAidDetection>& results,
        SopAidError* error) override {
        results.clear();
        if (image.empty()) {
            SetError(error, SopAidStatus::InvalidArgument, "Evaluate image is empty.");
            return SopAidStatus::InvalidArgument;
        }
        try {
            cv::Mat resized;
            cv::resize(image, resized, cv::Size(input_width_, input_height_));
            cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
            auto tensor = torch::from_blob(
                              resized.data,
                              {1, input_height_, input_width_, 3},
                              torch::TensorOptions().dtype(torch::kUInt8))
                              .permute({0, 3, 1, 2})
                              .to(torch::kFloat32)
                              .div_(255.0)
                              .contiguous();

            torch::InferenceMode guard;
            torch::Tensor output = ExtractTensor(module_.forward({tensor})).to(torch::kCPU).to(torch::kFloat32).contiguous();
            parseOutput(output, image.size(), results);
            ApplyClasswiseNms(results, confidence_threshold_, nms_threshold_);
            SetError(error, SopAidStatus::Ok, "");
            return SopAidStatus::Ok;
        } catch (const c10::Error& exc) {
            SetError(error, SopAidStatus::InferenceError, exc.what());
            return SopAidStatus::InferenceError;
        } catch (const std::exception& exc) {
            SetError(error, SopAidStatus::InferenceError, exc.what());
            return SopAidStatus::InferenceError;
        }
    }

private:
    void parseOutput(
        torch::Tensor output,
        const cv::Size& image_size,
        std::vector<SopAidDetection>& results) const {
        if (output.dim() == 2) {
            output = output.unsqueeze(0);
        }
        if (output.dim() != 3) {
            throw std::runtime_error("TorchScript output must have shape [1,N,6] or [1,6,N].");
        }
        if (output.size(2) == 6) {
            appendRows(output[0], image_size, results);
            return;
        }
        if (output.size(1) == 6) {
            appendRows(output[0].transpose(0, 1).contiguous(), image_size, results);
            return;
        }
        throw std::runtime_error("TorchScript output is not a supported postprocessed YOLO tensor.");
    }

    void appendRows(
        const torch::Tensor& rows,
        const cv::Size& image_size,
        std::vector<SopAidDetection>& results) const {
        const float x_scale = static_cast<float>(image_size.width) / input_width_;
        const float y_scale = static_cast<float>(image_size.height) / input_height_;
        const auto accessor = rows.accessor<float, 2>();
        for (int64_t row = 0; row < rows.size(0); ++row) {
            const float confidence = accessor[row][4];
            if (confidence < confidence_threshold_) {
                continue;
            }
            const int class_id = static_cast<int>(accessor[row][5]);
            const int x1 = static_cast<int>(accessor[row][0] * x_scale);
            const int y1 = static_cast<int>(accessor[row][1] * y_scale);
            const int x2 = static_cast<int>(accessor[row][2] * x_scale);
            const int y2 = static_cast<int>(accessor[row][3] * y_scale);
            results.push_back(MakeDetection(
                class_id,
                class_names_,
                confidence,
                cv::Rect(x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1))));
        }
    }

    torch::jit::script::Module module_;
    int input_width_;
    int input_height_;
    float confidence_threshold_;
    float nms_threshold_;
    std::vector<std::string> class_names_;
};

}  // namespace

std::unique_ptr<IYoloBackend> CreateTorchScriptBackend(const SopAidInitConfig& config, SopAidError* error) {
    if (!config.model_path || !std::filesystem::exists(config.model_path)) {
        SetError(error, SopAidStatus::FileNotFound, "TorchScript model file does not exist.");
        return nullptr;
    }
    try {
        auto backend = std::make_unique<TorchScriptYoloBackend>(config);
        SetError(error, SopAidStatus::Ok, "");
        return backend;
    } catch (const c10::Error& exc) {
        SetError(
            error,
            SopAidStatus::BackendError,
            std::string("Failed to load TorchScript model. Ultralytics training checkpoints are not TorchScript: ") +
                exc.what());
        return nullptr;
    } catch (const std::exception& exc) {
        SetError(error, SopAidStatus::BackendError, exc.what());
        return nullptr;
    }
}
