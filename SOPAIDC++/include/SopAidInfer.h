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
    // 根据模型文件扩展名选择后端。
    Auto = 0,
    // 此处的 Pt 指可由 LibTorch 加载的 TorchScript，并非训练检查点。
    Pt = 1,
    Onnx = 2,
    Engine = 3,
};

enum class SopAidResizeMode : int32_t {
    // 等比例缩放并填边；检测框会在后处理阶段映射回原图坐标。
    Letterbox = 0,
    // 预留值。当前 V2 接口仅接受 Letterbox，防止训练与推理预处理不一致。
    Stretch = 1,
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

// 多SOP项目使用的V2配置。保留SopAidInitConfig原布局，避免旧调用端发生ABI破坏。
struct SopAidProjectInitConfig {
    SopAidProjectInitConfig() : struct_size(sizeof(*this)) {}

    uint32_t struct_size;
    uint32_t api_version = 2;
    SopAidInitConfig inference{};
    SopAidResizeMode resize_mode = SopAidResizeMode::Letterbox;
    const char* project_id = nullptr;
    const char* model_id = nullptr;
    const char* model_version = nullptr;
};

// 从当前 Python 多 SOP 项目目录初始化推理。DLL 自动读取项目类别和部署模型。
struct SopAidProjectDirectoryConfig {
    SopAidProjectDirectoryConfig() : struct_size(sizeof(*this)) {}

    uint32_t struct_size;
    uint32_t api_version = 2;
    const char* project_dir = nullptr;
    SopAidModelFormat preferred_model_format = SopAidModelFormat::Auto;
    int32_t input_width = 640;
    int32_t input_height = 640;
    float confidence_threshold = 0.25f;
    float nms_threshold = 0.70f;
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

struct SopAidModelInfo {
    // 下列字符串由 DLL 拷贝到定长缓冲区，调用结束后仍可安全读取。
    char project_id[128] = {};
    char model_id[128] = {};
    char model_version[64] = {};
    char model_path[512] = {};
    char backend[32] = {};
    int32_t input_width = 0;
    int32_t input_height = 0;
    int32_t class_count = 0;
    SopAidResizeMode resize_mode = SopAidResizeMode::Letterbox;
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
    SopAidStatus InitProject(const SopAidProjectInitConfig& config, SopAidError* error = nullptr);
    SopAidStatus InitProjectDirectory(const SopAidProjectDirectoryConfig& config, SopAidError* error = nullptr);
    SopAidStatus Evaluate(
        const cv::Mat& image,
        std::vector<SopAidDetection>& results,
        SopAidError* error = nullptr);
    SopAidStatus GetModelInfo(SopAidModelInfo& info, SopAidError* error = nullptr) const;
    void Release();
    bool IsInitialized() const { return context_ != nullptr; }

private:
    void* context_ = nullptr;
};

}  // namespace sopaid
