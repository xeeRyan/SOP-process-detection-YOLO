#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

// SOPAID_API 控制 DLL 导出/导入：
// 编译 DLL 项目时导出符号，EXE 或其他调用方包含此头文件时导入符号。
#ifdef SOPAIDINFER_EXPORTS
#define SOPAID_API __declspec(dllexport)
#else
#define SOPAID_API __declspec(dllimport)
#endif

enum class SopAidModelFormat : int32_t {
    Auto = 0,
    Pt = 1,
    Onnx = 2,
    Engine = 3,
};

// 统一返回状态。调用失败时可同时读取 SopAidError.message 获取详细原因。
enum class SopAidStatus : int32_t {
    Ok = 0,
    InvalidArgument = 1,
    FileNotFound = 2,
    UnsupportedModel = 3,
    BackendError = 4,
    InferenceError = 5,
};

// 初始化模型时传入的配置；一个模型文件对应一个返回句柄。
// model_format 可显式指定，也可用 Auto 根据模型后缀自动识别。
struct SopAidInitConfig {
    const char* model_path = nullptr;
    SopAidModelFormat model_format = SopAidModelFormat::Auto;
    int32_t input_width = 640;
    int32_t input_height = 640;
    float confidence_threshold = 0.25f;
    float nms_threshold = 0.70f;
    const char* class_names_csv = "bearing,cover,tool";
    bool use_cuda = false;
    int32_t device_id = 0;
};

// 单个检测结果，坐标格式为左上角和右下角：x1,y1,x2,y2。
struct SopAidDetection {
    int32_t class_id = -1;
    char class_name[64] = {};
    float confidence = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

// 错误信息为可选输出；调用方需要诊断失败原因时传入。
struct SopAidError {
    SopAidStatus status = SopAidStatus::Ok;
    char message[512] = {};
};

namespace sopaid {

// 有状态 C++ 推理对象：
// Init(config) -> 重复 Evaluate(image, results) -> Release()。
// 对象内部管理模型句柄，调用方无需在每次 Evaluate 时传递 handle。
class SOPAID_API Inference {
public:
    Inference() = default;
    ~Inference();

    Inference(const Inference&) = delete;
    Inference& operator=(const Inference&) = delete;

    SopAidStatus Init(const SopAidInitConfig& config, SopAidError* error = nullptr);
    SopAidStatus Evaluate(
        const cv::Mat& image,
        std::vector<SopAidDetection>& results,
        SopAidError* error = nullptr);
    void Release();
    bool IsInitialized() const { return context_ != nullptr; }

private:
    void* context_ = nullptr;
};

}  // namespace sopaid
