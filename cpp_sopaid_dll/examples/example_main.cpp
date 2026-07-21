#include "SopAidInfer.h"

#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    const char* model_path =
        argc > 1 ? argv[1] : "E:/Project/SOPAID/SOPAID/models/best_yolo26s.onnx";
    const char* video_path =
        argc > 2 ? argv[2] : "E:/Project/SOPAID/SOPAID/videos/smoke_60f.mp4";
    const bool use_cuda = argc > 3 && std::string(argv[3]) == "cuda";
    const int device_id = argc > 4 ? std::stoi(argv[4]) : 0;

    SopAidError error;
    SopAidInitConfig config;
    config.model_path = model_path;
    config.model_format = SopAidModelFormat::Auto;
    config.input_width = 640;
    config.input_height = 640;
    config.class_names_csv = "bearing,cover,tool";
    config.use_cuda = use_cuda;
    config.device_id = device_id;

    SopAidHandle handle = sopaid::Init(config, &error);
    if (!handle) {
        std::cerr << "Init failed: " << error.message << std::endl;
        return 1;
    }

    SopAidVideoEvaluateInput input;
    input.handle = handle;
    input.video_path = video_path;
    input.frame_step = 1;
    input.max_frames = 0;

    SopAidVideoEvaluateResult output;
    const SopAidStatus status = sopaid::EvaluateVideo(input, output, &error);
    if (status != SopAidStatus::Ok) {
        std::cerr << "Evaluate failed: " << error.message << std::endl;
        sopaid::Release(handle);
        return 2;
    }

    for (const auto& detection : output.results) {
        std::cout << "frame=" << detection.frame_index
                  << " time=" << detection.time_sec
                  << " " << detection.detection.class_name
                  << " " << detection.detection.confidence << " ["
                  << detection.detection.x1 << ", " << detection.detection.y1 << ", "
                  << detection.detection.x2 << ", " << detection.detection.y2 << "]"
                  << std::endl;
    }

    sopaid::Release(handle);
    return 0;
}
