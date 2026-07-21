#include "IYoloBackend.h"
#include "YoloPostprocess.h"

#include <NvInfer.h>
#define NOMINMAX
#include <Windows.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using CudaError = int;
using CudaStream = cudaStream_t;
constexpr CudaError kCudaSuccess = 0;
constexpr int kCudaMemcpyHostToDevice = 1;
constexpr int kCudaMemcpyDeviceToHost = 2;

class CudaApi {
public:
    CudaApi() {
        module_ = LoadLibraryW(L"cudart64_12.dll");
        if (!module_) {
            throw std::runtime_error("cudart64_12.dll was not found.");
        }
        load(cudaSetDevice_, "cudaSetDevice");
        load(cudaMalloc_, "cudaMalloc");
        load(cudaFree_, "cudaFree");
        load(cudaMemcpyAsync_, "cudaMemcpyAsync");
        load(cudaStreamCreate_, "cudaStreamCreate");
        load(cudaStreamDestroy_, "cudaStreamDestroy");
        load(cudaStreamSynchronize_, "cudaStreamSynchronize");
    }

    ~CudaApi() {
        if (module_) {
            FreeLibrary(module_);
        }
    }

    void setDevice(int device_id) const { check(cudaSetDevice_(device_id), "cudaSetDevice"); }
    void malloc(void** pointer, size_t bytes) const { check(cudaMalloc_(pointer, bytes), "cudaMalloc"); }
    void free(void* pointer) const {
        if (pointer) {
            cudaFree_(pointer);
        }
    }
    void createStream(CudaStream* stream) const { check(cudaStreamCreate_(stream), "cudaStreamCreate"); }
    void destroyStream(CudaStream stream) const {
        if (stream) {
            cudaStreamDestroy_(stream);
        }
    }
    void copyAsync(void* destination, const void* source, size_t bytes, int kind, CudaStream stream) const {
        check(cudaMemcpyAsync_(destination, source, bytes, kind, stream), "cudaMemcpyAsync");
    }
    void synchronize(CudaStream stream) const { check(cudaStreamSynchronize_(stream), "cudaStreamSynchronize"); }

private:
    template <typename T>
    void load(T& function, const char* name) {
        function = reinterpret_cast<T>(GetProcAddress(module_, name));
        if (!function) {
            throw std::runtime_error(std::string("CUDA Runtime function was not found: ") + name);
        }
    }
    static void check(CudaError status, const char* operation) {
        if (status != kCudaSuccess) {
            throw std::runtime_error(std::string(operation) + " failed with CUDA error " + std::to_string(status));
        }
    }

    HMODULE module_ = nullptr;
    CudaError(__stdcall* cudaSetDevice_)(int) = nullptr;
    CudaError(__stdcall* cudaMalloc_)(void**, size_t) = nullptr;
    CudaError(__stdcall* cudaFree_)(void*) = nullptr;
    CudaError(__stdcall* cudaMemcpyAsync_)(void*, const void*, size_t, int, CudaStream) = nullptr;
    CudaError(__stdcall* cudaStreamCreate_)(CudaStream*) = nullptr;
    CudaError(__stdcall* cudaStreamDestroy_)(CudaStream) = nullptr;
    CudaError(__stdcall* cudaStreamSynchronize_)(CudaStream) = nullptr;
};

class TensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR) {
            last_error_ = message ? message : "";
        }
    }
    const std::string& lastError() const { return last_error_; }

private:
    std::string last_error_;
};

template <typename T>
struct TensorRtDeleter {
    void operator()(T* pointer) const { delete pointer; }
};

size_t ElementCount(const nvinfer1::Dims& dimensions) {
    size_t count = 1;
    for (int i = 0; i < dimensions.nbDims; ++i) {
        if (dimensions.d[i] <= 0) {
            throw std::runtime_error("TensorRT tensor has an unresolved dynamic dimension.");
        }
        count *= static_cast<size_t>(dimensions.d[i]);
    }
    return count;
}

class TensorRtYoloBackend final : public IYoloBackend {
public:
    explicit TensorRtYoloBackend(const SopAidInitConfig& config)
        : input_width_(config.input_width),
          input_height_(config.input_height),
          confidence_threshold_(config.confidence_threshold),
          nms_threshold_(config.nms_threshold),
          class_names_(ParseClassNames(config.class_names_csv)) {
        cuda_.setDevice(config.device_id);
        std::ifstream stream(config.model_path, std::ios::binary | std::ios::ate);
        if (!stream) {
            throw std::runtime_error("Failed to open TensorRT engine.");
        }
        const auto size = static_cast<size_t>(stream.tellg());
        std::vector<char> engine_bytes(size);
        stream.seekg(0);
        stream.read(engine_bytes.data(), static_cast<std::streamsize>(size));

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            throw std::runtime_error("TensorRT runtime creation failed.");
        }
        engine_.reset(runtime_->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
        if (!engine_) {
            throw std::runtime_error("TensorRT engine deserialization failed: " + logger_.lastError());
        }
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("TensorRT execution context creation failed.");
        }

        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                input_name_ = name;
            } else if (output_name_.empty()) {
                output_name_ = name;
            }
        }
        if (input_name_.empty() || output_name_.empty()) {
            throw std::runtime_error("TensorRT engine must expose one input and at least one output tensor.");
        }

        nvinfer1::Dims input_dims = engine_->getTensorShape(input_name_.c_str());
        if (input_dims.nbDims != 4) {
            throw std::runtime_error("TensorRT input tensor must be NCHW.");
        }
        if (input_dims.d[2] > 0) input_height_ = input_dims.d[2];
        if (input_dims.d[3] > 0) input_width_ = input_dims.d[3];
        input_dims.d[0] = 1;
        input_dims.d[1] = 3;
        input_dims.d[2] = input_height_;
        input_dims.d[3] = input_width_;
        if (!context_->setInputShape(input_name_.c_str(), input_dims)) {
            throw std::runtime_error("TensorRT input shape configuration failed.");
        }

        output_dims_ = context_->getTensorShape(output_name_.c_str());
        input_elements_ = ElementCount(input_dims);
        output_elements_ = ElementCount(output_dims_);
        cuda_.malloc(&input_device_, input_elements_ * sizeof(float));
        cuda_.malloc(&output_device_, output_elements_ * sizeof(float));
        cuda_.createStream(&stream_);
        if (!context_->setTensorAddress(input_name_.c_str(), input_device_) ||
            !context_->setTensorAddress(output_name_.c_str(), output_device_)) {
            throw std::runtime_error("TensorRT tensor address configuration failed.");
        }
    }

    ~TensorRtYoloBackend() override {
        cuda_.destroyStream(stream_);
        cuda_.free(output_device_);
        cuda_.free(input_device_);
    }

    SopAidStatus evaluate(const cv::Mat& image, std::vector<SopAidDetection>& results, SopAidError* error) override {
        results.clear();
        if (image.empty()) {
            SetError(error, SopAidStatus::InvalidArgument, "Evaluate image is empty.");
            return SopAidStatus::InvalidArgument;
        }
        try {
            const auto input = preprocess(image);
            std::vector<float> output(output_elements_);
            cuda_.copyAsync(
                input_device_, input.data(), input.size() * sizeof(float), kCudaMemcpyHostToDevice, stream_);
            if (!context_->enqueueV3(stream_)) {
                throw std::runtime_error("TensorRT enqueueV3 failed: " + logger_.lastError());
            }
            cuda_.copyAsync(
                output.data(), output_device_, output.size() * sizeof(float), kCudaMemcpyDeviceToHost, stream_);
            cuda_.synchronize(stream_);
            parseOutput(output, image.size(), results);
            ApplyClasswiseNms(results, confidence_threshold_, nms_threshold_);
            SetError(error, SopAidStatus::Ok, "");
            return SopAidStatus::Ok;
        } catch (const std::exception& exc) {
            SetError(error, SopAidStatus::InferenceError, exc.what());
            return SopAidStatus::InferenceError;
        }
    }

private:
    std::vector<float> preprocess(const cv::Mat& image) const {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(input_width_, input_height_));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        std::vector<float> tensor(input_elements_);
        const int channel_size = input_width_ * input_height_;
        for (int y = 0; y < input_height_; ++y) {
            const auto* row = resized.ptr<cv::Vec3b>(y);
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
        const std::vector<float>& output,
        const cv::Size& image_size,
        std::vector<SopAidDetection>& results) const {
        int rows = 0;
        bool transposed = false;
        if (output_dims_.nbDims == 3 && output_dims_.d[2] == 6) {
            rows = output_dims_.d[1];
        } else if (output_dims_.nbDims == 3 && output_dims_.d[1] == 6) {
            rows = output_dims_.d[2];
            transposed = true;
        } else {
            throw std::runtime_error("TensorRT output must have shape [1,N,6] or [1,6,N].");
        }
        const float x_scale = static_cast<float>(image_size.width) / input_width_;
        const float y_scale = static_cast<float>(image_size.height) / input_height_;
        for (int row = 0; row < rows; ++row) {
            const auto value = [&](int attribute) {
                return transposed ? output[static_cast<size_t>(attribute * rows + row)]
                                  : output[static_cast<size_t>(row * 6 + attribute)];
            };
            const float confidence = value(4);
            if (confidence < confidence_threshold_) continue;
            const int class_id = static_cast<int>(value(5));
            const int x1 = static_cast<int>(value(0) * x_scale);
            const int y1 = static_cast<int>(value(1) * y_scale);
            const int x2 = static_cast<int>(value(2) * x_scale);
            const int y2 = static_cast<int>(value(3) * y_scale);
            results.push_back(MakeDetection(
                class_id,
                class_names_,
                confidence,
                cv::Rect(x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1))));
        }
    }

    TensorRtLogger logger_;
    CudaApi cuda_;
    std::unique_ptr<nvinfer1::IRuntime, TensorRtDeleter<nvinfer1::IRuntime>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TensorRtDeleter<nvinfer1::ICudaEngine>> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TensorRtDeleter<nvinfer1::IExecutionContext>> context_;
    std::string input_name_;
    std::string output_name_;
    nvinfer1::Dims output_dims_{};
    int input_width_;
    int input_height_;
    float confidence_threshold_;
    float nms_threshold_;
    std::vector<std::string> class_names_;
    size_t input_elements_ = 0;
    size_t output_elements_ = 0;
    void* input_device_ = nullptr;
    void* output_device_ = nullptr;
    CudaStream stream_ = nullptr;
};

}  // namespace

std::unique_ptr<IYoloBackend> CreateTensorRtBackend(const SopAidInitConfig& config, SopAidError* error) {
    if (!config.model_path || !std::filesystem::exists(config.model_path)) {
        SetError(error, SopAidStatus::FileNotFound, "TensorRT engine file does not exist.");
        return nullptr;
    }
    try {
        auto backend = std::make_unique<TensorRtYoloBackend>(config);
        SetError(error, SopAidStatus::Ok, "");
        return backend;
    } catch (const std::exception& exc) {
        SetError(error, SopAidStatus::BackendError, exc.what());
        return nullptr;
    }
}
