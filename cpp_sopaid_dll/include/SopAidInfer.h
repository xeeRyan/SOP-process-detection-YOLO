#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

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

enum class SopAidStatus : int32_t {
    Ok = 0,
    InvalidArgument = 1,
    FileNotFound = 2,
    UnsupportedModel = 3,
    BackendError = 4,
    InferenceError = 5,
};

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

// 鎵€鏈夋ā鍨嬪悗绔粺涓€杩斿洖鐨勬娴嬬粨鏋溿€?
struct SopAidDetection {
    int32_t class_id = -1;
    char class_name[64] = {};
    float confidence = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

// 閿欒淇℃伅涓哄彲閫夎緭鍑猴紱璋冪敤鏂归渶瑕佽瘖鏂け璐ュ師鍥犳椂浼犲叆銆?
struct SopAidError {
    SopAidStatus status = SopAidStatus::Ok;
    char message[512] = {};
};

using SopAidHandle = void*;

// 鍗曞抚鎺ㄧ悊杈撳叆缁撴瀯浣擄紱涓€涓彞鏌勫搴斾竴涓凡鍔犺浇妯″瀷銆?
struct SopAidEvaluateInput {
    SopAidHandle handle = nullptr;
    const cv::Mat* image = nullptr;
};

// C 椋庢牸杈撳嚭缁撴瀯浣擄紱缁撴灉鏁扮粍鍐呭瓨鐢辫皟鐢ㄦ柟鐢宠鍜岄噴鏀俱€?
struct SopAidEvaluateOutput {
    SopAidDetection* results = nullptr;
    int32_t result_capacity = 0;
    int32_t result_count = 0;
};

// C++ 椋庢牸杈撳嚭缁撴瀯浣擄紱DLL 灏嗘娴嬬粨鏋滃啓鍏?vector銆?
struct SopAidEvaluateResult {
    std::vector<SopAidDetection> results;
};

// 瑙嗛鎺ㄧ悊杈撳叆缁撴瀯浣擄紱DLL 鍐呴儴璐熻矗鎵撳紑瑙嗛骞堕€愬抚鎺ㄧ悊銆?
struct SopAidVideoEvaluateInput {
    SopAidHandle handle = nullptr;
    const char* video_path = nullptr;
    int32_t frame_step = 1;
    int32_t max_frames = 0;
};

// 瑙嗛鎺ㄧ悊涓殑鍗曟潯妫€娴嬬粨鏋滐紝棰濆璁板綍鏉ヨ嚜鍝竴甯с€?
struct SopAidVideoDetection {
    int64_t frame_index = 0;
    double time_sec = 0.0;
    SopAidDetection detection;
};

// C 椋庢牸瑙嗛杈撳嚭缁撴瀯浣擄紱缁撴灉鏁扮粍鍐呭瓨鐢辫皟鐢ㄦ柟鐢宠鍜岄噴鏀俱€?
struct SopAidVideoEvaluateOutput {
    SopAidVideoDetection* results = nullptr;
    int32_t result_capacity = 0;
    int32_t result_count = 0;
};

// C++ 椋庢牸瑙嗛杈撳嚭缁撴瀯浣擄紱DLL 灏嗘暣娈佃棰戠殑妫€娴嬬粨鏋滃啓鍏?vector銆?
struct SopAidVideoEvaluateResult {
    std::vector<SopAidVideoDetection> results;
};

extern "C" {
SOPAID_API SopAidHandle SopAid_Init(const SopAidInitConfig* config, SopAidError* error);
SOPAID_API SopAidHandle SopAid_InitPt(const SopAidInitConfig* config, SopAidError* error);
SOPAID_API SopAidHandle SopAid_InitOnnx(const SopAidInitConfig* config, SopAidError* error);
SOPAID_API SopAidHandle SopAid_InitEngine(const SopAidInitConfig* config, SopAidError* error);
SOPAID_API SopAidStatus SopAid_EvaluateStruct(
    const SopAidEvaluateInput* input,
    SopAidEvaluateOutput* output,
    SopAidError* error);
SOPAID_API SopAidStatus SopAid_Evaluate(
    SopAidHandle handle,
    const cv::Mat* image,
    SopAidDetection* results,
    int32_t result_capacity,
    int32_t* result_count,
    SopAidError* error);
SOPAID_API SopAidStatus SopAid_EvaluateVideoStruct(
    const SopAidVideoEvaluateInput* input,
    SopAidVideoEvaluateOutput* output,
    SopAidError* error);
SOPAID_API void SopAid_Release(SopAidHandle handle);
}

namespace sopaid {

SOPAID_API SopAidHandle Init(const SopAidInitConfig& config, SopAidError* error = nullptr);
SOPAID_API SopAidHandle InitPt(const SopAidInitConfig& config, SopAidError* error = nullptr);
SOPAID_API SopAidHandle InitOnnx(const SopAidInitConfig& config, SopAidError* error = nullptr);
SOPAID_API SopAidHandle InitEngine(const SopAidInitConfig& config, SopAidError* error = nullptr);

SOPAID_API SopAidStatus Evaluate(
    const SopAidEvaluateInput& input,
    SopAidEvaluateResult& output,
    SopAidError* error = nullptr);

SOPAID_API SopAidStatus Evaluate(
    const SopAidEvaluateInput& input,
    SopAidEvaluateOutput& output,
    SopAidError* error = nullptr);

SOPAID_API SopAidStatus Evaluate(
    SopAidHandle handle,
    const cv::Mat& image,
    std::vector<SopAidDetection>& results,
    SopAidError* error = nullptr);

SOPAID_API SopAidStatus EvaluateVideo(
    const SopAidVideoEvaluateInput& input,
    SopAidVideoEvaluateResult& output,
    SopAidError* error = nullptr);

SOPAID_API SopAidStatus EvaluateVideo(
    const SopAidVideoEvaluateInput& input,
    SopAidVideoEvaluateOutput& output,
    SopAidError* error = nullptr);

SOPAID_API void Release(SopAidHandle handle);

}  // namespace sopaid

