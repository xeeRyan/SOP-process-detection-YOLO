#include "IYoloBackend.h"
#include "YoloPostprocess.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SOPAID_ENABLE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#define NOMINMAX
#include <Windows.h>
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

void ParsePostprocessedOutput(
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

#ifdef SOPAID_ENABLE_TENSORRT

namespace {

using CudaError = int;
constexpr CudaError kCudaSuccess = 0;

class CudaApi {
public:
    CudaApi() {
        module_ = LoadLibraryW(L"cudart64_12.dll");
        if (!module_) throw std::runtime_error("cudart64_12.dll was not found.");
        load(cudaSetDevice_, "cudaSetDevice");
        load(cudaMalloc_, "cudaMalloc");
        load(cudaFree_, "cudaFree");
        load(cudaMemcpyAsync_, "cudaMemcpyAsync");
        load(cudaStreamCreate_, "cudaStreamCreate");
        load(cudaStreamDestroy_, "cudaStreamDestroy");
        load(cudaStreamSynchronize_, "cudaStreamSynchronize");
    }

    ~CudaApi() {
        if (module_) FreeLibrary(module_);
    }

    void setDevice(int device_id) const { check(cudaSetDevice_(device_id), "cudaSetDevice"); }
    void malloc(void** pointer, size_t bytes) const { check(cudaMalloc_(pointer, bytes), "cudaMalloc"); }
    void free(void* pointer) const { if (pointer) cudaFree_(pointer); }
    void createStream(cudaStream_t* stream) const { check(cudaStreamCreate_(stream), "cudaStreamCreate"); }
    void destroyStream(cudaStream_t stream) const { if (stream) cudaStreamDestroy_(stream); }
    void copyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) const {
        check(cudaMemcpyAsync_(dst, src, bytes, kind, stream), "cudaMemcpyAsync");
    }
    void synchronize(cudaStream_t stream) const {
        check(cudaStreamSynchronize_(stream), "cudaStreamSynchronize");
    }

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
            throw std::runtime_error(
                std::string(operation) + " failed with CUDA error " + std::to_string(status));
        }
    }

    HMODULE module_ = nullptr;
    CudaError(__stdcall* cudaSetDevice_)(int) = nullptr;
    CudaError(__stdcall* cudaMalloc_)(void**, size_t) = nullptr;
    CudaError(__stdcall* cudaFree_)(void*) = nullptr;
    CudaError(__stdcall* cudaMemcpyAsync_)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) = nullptr;
    CudaError(__stdcall* cudaStreamCreate_)(cudaStream_t*) = nullptr;
    CudaError(__stdcall* cudaStreamDestroy_)(cudaStream_t) = nullptr;
    CudaError(__stdcall* cudaStreamSynchronize_)(cudaStream_t) = nullptr;
};

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            last_message_ = message ? message : "";
        }
    }

private:
    std::string last_message_;
};

std::vector<char> ReadBinaryFile(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open TensorRT engine file.");
    }
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<size_t>(size));
    if (!file.read(data.data(), size)) {
        throw std::runtime_error("Failed to read TensorRT engine file.");
    }
    return data;
}

size_t Volume(const nvinfer1::Dims& dims) {
    size_t value = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            throw std::runtime_error("TensorRT tensor dimension is still dynamic.");
        }
        value *= static_cast<size_t>(std::max<int64_t>(1, dims.d[i]));
    }
    return value;
}

}  // namespace

class TensorRtYoloBackend final : public IYoloBackend {
public:
    explicit TensorRtYoloBackend(const SopAidInitConfig& config)
        : input_width_(config.input_width),
          input_height_(config.input_height),
          conf_threshold_(config.confidence_threshold),
          class_names_(ParseClassNames(config.class_names_csv)) {
        cuda_.setDevice(config.device_id);
        const std::vector<char> engine_data = ReadBinaryFile(config.model_path);
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime.");
        }

        engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize TensorRT engine.");
        }

        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("Failed to create TensorRT execution context.");
        }

        findIOTensors();
        configureInputShape();
        allocateBuffers();
        cuda_.createStream(&stream_);
    }

    ~TensorRtYoloBackend() override {
        cuda_.destroyStream(stream_);
        cuda_.free(input_device_);
        cuda_.free(output_device_);
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
            cuda_.copyAsync(
                input_device_,
                input_data.data(),
                input_data.size() * sizeof(float),
                cudaMemcpyHostToDevice,
                stream_);

            if (!context_->enqueueV3(stream_)) {
                SetError(error, SopAidStatus::InferenceError, "TensorRT enqueueV3 failed.");
                return SopAidStatus::InferenceError;
            }

            std::vector<float> output_data(output_element_count_);
            cuda_.copyAsync(
                output_data.data(),
                output_device_,
                output_data.size() * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream_);
            cuda_.synchronize(stream_);

            parseOutput(output_data.data(), image.size(), letterbox, results);
            SetError(error, SopAidStatus::Ok, "");
            return SopAidStatus::Ok;
        } catch (const std::exception& ex) {
            SetError(error, SopAidStatus::InferenceError, ex.what());
            return SopAidStatus::InferenceError;
        }
    }

private:
    void findIOTensors() {
        const int tensor_count = engine_->getNbIOTensors();
        for (int i = 0; i < tensor_count; ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (!name) {
                continue;
            }
            const nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
            if (mode == nvinfer1::TensorIOMode::kINPUT && input_name_.empty()) {
                input_name_ = name;
            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT && output_name_.empty()) {
                output_name_ = name;
            }
        }
        if (input_name_.empty() || output_name_.empty()) {
            throw std::runtime_error("TensorRT engine must have at least one input and one output tensor.");
        }
        if (engine_->getTensorDataType(input_name_.c_str()) != nvinfer1::DataType::kFLOAT ||
            engine_->getTensorDataType(output_name_.c_str()) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("TensorRT backend currently supports float32 input/output tensors only.");
        }
    }

    void configureInputShape() {
        nvinfer1::Dims input_dims = engine_->getTensorShape(input_name_.c_str());
        bool dynamic = false;
        for (int i = 0; i < input_dims.nbDims; ++i) {
            if (input_dims.d[i] < 0) {
                dynamic = true;
            }
        }
        if (dynamic) {
            if (!context_->setInputShape(input_name_.c_str(), nvinfer1::Dims4{1, 3, input_height_, input_width_})) {
                throw std::runtime_error("Failed to set TensorRT dynamic input shape.");
            }
        } else if (input_dims.nbDims == 4) {
            input_height_ = static_cast<int>(input_dims.d[2]);
            input_width_ = static_cast<int>(input_dims.d[3]);
        }
    }

    void allocateBuffers() {
        const nvinfer1::Dims input_dims = context_->getTensorShape(input_name_.c_str());
        output_dims_ = context_->getTensorShape(output_name_.c_str());
        const size_t input_bytes = Volume(input_dims) * sizeof(float);
        const size_t output_bytes = Volume(output_dims_) * sizeof(float);
        output_element_count_ = output_bytes / sizeof(float);

        cuda_.malloc(&input_device_, input_bytes);
        cuda_.malloc(&output_device_, output_bytes);
        if (!context_->setInputTensorAddress(input_name_.c_str(), input_device_)) {
            throw std::runtime_error("Failed to set TensorRT input tensor address.");
        }
        if (!context_->setTensorAddress(output_name_.c_str(), output_device_)) {
            throw std::runtime_error("Failed to set TensorRT output tensor address.");
        }
    }

    void parseOutput(const float* data, const cv::Size& image_size, const LetterboxInfo& letterbox, std::vector<SopAidDetection>& results) const {
        if (output_dims_.nbDims == 3 && output_dims_.d[2] == 6) {
            ParsePostprocessedOutput(data, static_cast<int>(output_dims_.d[1]), 6, image_size, letterbox, conf_threshold_, class_names_, results);
            return;
        }
        if (output_dims_.nbDims == 2 && output_dims_.d[1] == 6) {
            ParsePostprocessedOutput(data, static_cast<int>(output_dims_.d[0]), 6, image_size, letterbox, conf_threshold_, class_names_, results);
            return;
        }
        if (output_dims_.nbDims == 3 && output_dims_.d[1] == 6) {
            const int rows = static_cast<int>(output_dims_.d[2]);
            for (int row = 0; row < rows; ++row) {
                const float confidence = data[4 * rows + row];
                if (confidence < conf_threshold_) {
                    continue;
                }
                const int class_id = static_cast<int>(data[5 * rows + row]);
                const cv::Rect box = RestoreBox(data[row], data[rows + row], data[2 * rows + row], data[3 * rows + row], image_size, letterbox);
                if (!box.empty()) {
                    results.push_back(MakeDetection(class_id, class_names_, confidence, box));
                }
            }
        }
    }

    CudaApi cuda_;
    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::string input_name_;
    std::string output_name_;
    void* input_device_ = nullptr;
    void* output_device_ = nullptr;
    cudaStream_t stream_ = nullptr;
    nvinfer1::Dims output_dims_{};
    size_t output_element_count_ = 0;
    int input_width_ = 640;
    int input_height_ = 640;
    float conf_threshold_ = 0.25f;
    std::vector<std::string> class_names_;
};

std::unique_ptr<IYoloBackend> CreateTensorRtBackend(const SopAidInitConfig& config, SopAidError* error) {
    if (!config.model_path || !std::filesystem::exists(config.model_path)) {
        SetError(error, SopAidStatus::FileNotFound, "TensorRT engine file does not exist.");
        return nullptr;
    }
    try {
        auto backend = std::make_unique<TensorRtYoloBackend>(config);
        SetError(error, SopAidStatus::Ok, "");
        return backend;
    } catch (const std::exception& ex) {
        SetError(error, SopAidStatus::BackendError, ex.what());
        return nullptr;
    }
}

#else

std::unique_ptr<IYoloBackend> CreateTensorRtBackend(const SopAidInitConfig& config, SopAidError* error) {
    (void)config;
    SetError(error, SopAidStatus::UnsupportedModel, "TensorRT backend is not enabled. Set EnableTensorRT=true and configure TensorRTRoot in SopAidInfer.user.props.");
    return nullptr;
}

#endif
