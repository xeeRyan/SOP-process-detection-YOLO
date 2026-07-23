#include "SopAidInfer.h"
#include "SopAidHandPose.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// 默认输出位置对齐 Python 交付包结构：
// 每次测试结果放在 outputs 下的新建文件夹，日志统一放在 outputs/logs。
const char* kDefaultOutputRoot = "E:/Project/SOPAID/SOPAID/dist/SOP_PYD/outputs";
const char* kDefaultLogDir = "E:/Project/SOPAID/SOPAID/outputs/logs";
const char* kDefaultPalmModelPath = "E:/Project/SOPAID/SOPAID/dist/SOP_PYD/models/palm_detection_mediapipe_2023feb.onnx";
const char* kDefaultHandPoseModelPath = "E:/Project/SOPAID/SOPAID/dist/SOP_PYD/models/handpose_estimation_mediapipe_2023feb.onnx";

SopAidModelFormat inferFormatFromPath(const std::string& model_path);

// 一次前端/命令行请求对应一个配置对象。
// YOLO 和手部的可变初始化参数直接保存在各自 Init 结构体中，避免二次复制。
struct SopAidRequestConfig {
    // 每次运行需要调整的参数统一在这里修改。
    std::string model_path =
        "E:/Project/SOPAID/SOPAID/dist/SOP_PYD/models/best_yolo26n.torchscript";
    std::string video_path =
        "E:/Project/SOPAID/SOPAID/videos/sk.mp4";
    std::filesystem::path output_root =
        "E:/Project/SOPAID/SOPAID/outputs";
    std::string palm_model_path = kDefaultPalmModelPath;
    std::string handpose_model_path = kDefaultHandPoseModelPath;
    int sample_interval = 30;

    SopAidInitConfig yolo;
    SopAidHandInitConfig hand;

    SopAidRequestConfig() {
        // 以下为程序固定配置，不由当前请求修改。
        yolo.input_width = 640;
        yolo.input_height = 640;
        yolo.confidence_threshold = 0.25f;
        yolo.nms_threshold = 0.70f;
        yolo.class_names_csv = "bearing,cover,tool";
        yolo.use_cuda = true;
        yolo.device_id = 0;

        hand.max_num_hands = 2;
        hand.min_hand_detection_confidence = 0.5f;
        hand.min_hand_presence_confidence = 0.5f;
        hand.use_gpu = false;
    }

    void bindPaths() {
        yolo.model_path = model_path.c_str();
        yolo.model_format = inferFormatFromPath(model_path);
        hand.palm_model_path = palm_model_path.c_str();
        hand.handpose_model_path = handpose_model_path.c_str();
    }
};

std::string timestampForFile() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
    localtime_s(&local_time, &now_time);
    std::ostringstream stream;
    stream << std::put_time(&local_time, "%Y%m%d_%H%M%S");
    return stream.str();
}

std::string timestampForLine() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
    localtime_s(&local_time, &now_time);
    std::ostringstream stream;
    stream << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S");
    return stream.str();
}

class RunLogger {
public:
    explicit RunLogger(const std::filesystem::path& log_dir) {
        std::filesystem::create_directories(log_dir);
        log_path_ = log_dir / ("detect_" + timestampForFile() + ".log");
        stream_.open(log_path_, std::ios::out | std::ios::trunc);
    }

    void info(const std::string& message) { write("INFO", message); }
    void error(const std::string& message) { write("ERROR", message); }
    std::string path() const { return log_path_.string(); }

private:
    void write(const char* level, const std::string& message) {
        const std::string line = timestampForLine() + " [" + level + "] " + message;
        std::cout << message << std::endl;
        if (stream_.is_open()) {
            stream_ << line << std::endl;
            stream_.flush();
        }
    }

    std::filesystem::path log_path_;
    std::ofstream stream_;
};

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

SopAidModelFormat inferFormatFromPath(const std::string& model_path) {
    // EXE 根据模型后缀选择 DLL 初始化接口；真正的模型加载由 DLL 后端完成。
    const std::string ext = lowerCopy(std::filesystem::path(model_path).extension().string());
    if (ext == ".pt") return SopAidModelFormat::Pt;
    if (ext == ".onnx") return SopAidModelFormat::Onnx;
    if (ext == ".engine") return SopAidModelFormat::Engine;
    return SopAidModelFormat::Auto;
}

std::string formatName(SopAidModelFormat format) {
    switch (format) {
    case SopAidModelFormat::Pt: return "pt";
    case SopAidModelFormat::Onnx: return "onnx";
    case SopAidModelFormat::Engine: return "engine";
    default: return "auto";
    }
}

bool parseFloat(const char* text, float& value) {
    // 命令行传入的阈值需要先校验，避免错误字符串落入推理配置。
    try {
        size_t consumed = 0;
        const float parsed = std::stof(text, &consumed);
        if (consumed != std::string(text).size()) return false;
        value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}


bool parseInt(const char* text, int& value) {
    try {
        size_t consumed = 0;
        const int parsed = std::stoi(text, &consumed);
        if (consumed != std::string(text).size()) return false;
        value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<std::string> collectArgs(int argc, char** argv) {
    std::vector<std::string> args;
    args.reserve(argc);
    for (int i = 0; i < argc; ++i) {
        std::string value = argv[i] ? argv[i] : "";
        // VS“命令参数”里不能使用 cmd 的换行符 ^。
        // 如果误填了 ^，这里过滤掉，避免后续参数整体错位。
        if (value == "^") continue;
        args.push_back(std::move(value));
    }
    return args;
}

std::string jsonEscape(const std::string& value) {
    std::string escaped;
    for (char ch : value) {
        if (ch == '\\') escaped += "\\\\";
        else if (ch == '"') escaped += "\\\"";
        else escaped += ch;
    }
    return escaped;
}

std::filesystem::path makeRunDir(const std::string& model_path, const std::string& video_path, const std::filesystem::path& output_root) {
    const std::string model_stem = std::filesystem::path(model_path).stem().string();
    const std::string video_stem = std::filesystem::path(video_path).stem().string();
    return output_root / (model_stem + "_" + video_stem + "_" + timestampForFile());
}

void writeParamsJson(
    const std::filesystem::path& path,
    const std::string& output_video_path,
    const std::string& log_path,
    const SopAidRequestConfig& request,
    bool enable_hand_pose,
    const std::string& sample_frames_dir) {
    // 参数文件记录本次测试输入和输出位置，便于之后复现实验。
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    out << "{\n";
    const SopAidInitConfig& config = request.yolo;
    out << "  \"model_path\": \"" << jsonEscape(request.model_path) << "\",\n";
    out << "  \"video_path\": \"" << jsonEscape(request.video_path) << "\",\n";
    out << "  \"output_video\": \"" << jsonEscape(output_video_path) << "\",\n";
    out << "  \"log_path\": \"" << jsonEscape(log_path) << "\",\n";
    out << "  \"model_format\": \"" << formatName(config.model_format) << "\",\n";
    out << "  \"input_width\": " << config.input_width << ",\n";
    out << "  \"input_height\": " << config.input_height << ",\n";
    out << "  \"confidence_threshold\": " << config.confidence_threshold << ",\n";
    out << "  \"nms_threshold\": " << config.nms_threshold << ",\n";
    out << "  \"class_names_csv\": \"" << jsonEscape(config.class_names_csv ? config.class_names_csv : "") << "\",\n";
    out << "  \"use_cuda\": " << (config.use_cuda ? "true" : "false") << ",\n";
    out << "  \"device_id\": " << config.device_id << ",\n";
    out << "  \"enable_hand_pose\": " << (enable_hand_pose ? "true" : "false") << ",\n";
    out << "  \"palm_model_path\": \"" << jsonEscape(request.palm_model_path) << "\",\n";
    out << "  \"handpose_model_path\": \"" << jsonEscape(request.handpose_model_path) << "\",\n";
    out << "  \"max_num_hands\": " << request.hand.max_num_hands << ",\n";
    out << "  \"min_hand_detection_confidence\": " << request.hand.min_hand_detection_confidence << ",\n";
    out << "  \"min_hand_presence_confidence\": " << request.hand.min_hand_presence_confidence << ",\n";
    out << "  \"hand_use_gpu\": " << (request.hand.use_gpu ? "true" : "false") << ",\n";
    out << "  \"sample_interval\": " << request.sample_interval << ",\n";
    out << "  \"sample_frames_dir\": \"" << jsonEscape(sample_frames_dir) << "\"\n";
    out << "}\n";
}


std::string detectionToJson(const SopAidDetection& det) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(4);
    out << "{"
        << "\"class_id\":" << det.class_id << ","
        << "\"class_name\":\"" << jsonEscape(det.class_name) << "\","
        << "\"confidence\":" << det.confidence << ","
        << "\"box\":[" << det.x1 << "," << det.y1 << "," << det.x2 << "," << det.y2 << "]"
        << "}";
    return out.str();
}

std::string handToJson(const SopAidHandResult& hand) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(4);
    out << "{"
        << "\"hand_id\":" << hand.hand_id << ","
        << "\"confidence\":" << hand.confidence << ","
        << "\"handedness\":\"" << jsonEscape(hand.handedness) << "\","
        << "\"landmarks\":[";
    for (int i = 0; i < SOPAID_HAND_LANDMARK_COUNT; ++i) {
        if (i > 0) out << ",";
        const auto& point = hand.landmarks[i];
        out << "{"
            << "\"x\":" << point.x << ","
            << "\"y\":" << point.y << ","
            << "\"z\":" << point.z << ","
            << "\"visibility\":" << point.visibility
            << "}";
    }
    out << "]}";
    return out.str();
}

void writeFrameResultJsonl(
    std::ofstream& out,
    int frame_index,
    double time_sec,
    double yolo_infer_ms,
    double hand_infer_ms,
    double instant_fps,
    const std::vector<SopAidDetection>& detections,
    const std::vector<SopAidHandResult>& hands) {
    if (!out.is_open()) return;
    out.setf(std::ios::fixed);
    out.precision(4);
    out << "{"
        << "\"frame_index\":" << frame_index << ","
        << "\"time_sec\":" << time_sec << ","
        << "\"yolo_infer_ms\":" << yolo_infer_ms << ","
        << "\"hand_infer_ms\":" << hand_infer_ms << ","
        << "\"fps\":" << instant_fps << ","
        << "\"detections\":[";
    for (size_t i = 0; i < detections.size(); ++i) {
        if (i > 0) out << ",";
        out << detectionToJson(detections[i]);
    }
    out << "],"
        << "\"hands\":[";
    for (size_t i = 0; i < hands.size(); ++i) {
        if (i > 0) out << ",";
        out << handToJson(hands[i]);
    }
    out << "]}\n";
}

void drawPerfOverlay(cv::Mat& frame, double fps, double yolo_ms, double hand_ms, int frame_index) {
    std::ostringstream text;
    text.setf(std::ios::fixed);
    text.precision(1);
    text << "FPS " << fps << " | YOLO " << yolo_ms << " ms | Hand " << hand_ms << " ms | frame " << frame_index;
    const cv::Point origin(18, 32);
    cv::putText(frame, text.str(), origin + cv::Point(1, 1), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(frame, text.str(), origin, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}
cv::Scalar colorForClass(int class_id) {
    static const cv::Scalar colors[] = {
        cv::Scalar(0, 220, 255),
        cv::Scalar(0, 180, 0),
        cv::Scalar(255, 80, 80),
        cv::Scalar(180, 80, 255),
    };
    return colors[std::abs(class_id) % static_cast<int>(std::size(colors))];
}

std::string buildLabel(const SopAidDetection& det) {
    std::ostringstream stream;
    stream.setf(std::ios::fixed);
    stream.precision(3);
    stream << det.class_name << " " << det.confidence;
    return stream.str();
}

void drawDetections(cv::Mat& frame, const std::vector<SopAidDetection>& detections) {
    for (const auto& det : detections) {
        const int x1 = std::clamp(static_cast<int>(std::round(det.x1)), 0, frame.cols - 1);
        const int y1 = std::clamp(static_cast<int>(std::round(det.y1)), 0, frame.rows - 1);
        const int x2 = std::clamp(static_cast<int>(std::round(det.x2)), 0, frame.cols - 1);
        const int y2 = std::clamp(static_cast<int>(std::round(det.y2)), 0, frame.rows - 1);
        if (x2 <= x1 || y2 <= y1) continue;

        const cv::Scalar color = colorForClass(det.class_id);
        cv::rectangle(frame, cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)), color, 2);
        cv::putText(frame, buildLabel(det), cv::Point(x1, std::max(20, y1 - 8)), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv::LINE_AA);
    }
}


cv::Point2f landmarkPoint(const SopAidHandResult& hand, int index) {
    return cv::Point2f(hand.landmarks[index].x, hand.landmarks[index].y);
}

void drawHandPoses(cv::Mat& frame, const std::vector<SopAidHandResult>& hands) {
    static const int connections[][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 4},
        {0, 5}, {5, 6}, {6, 7}, {7, 8},
        {0, 9}, {9, 10}, {10, 11}, {11, 12},
        {0, 13}, {13, 14}, {14, 15}, {15, 16},
        {0, 17}, {17, 18}, {18, 19}, {19, 20}
    };
    const cv::Scalar line_color(255, 180, 0);
    const cv::Scalar point_color(0, 255, 255);
    for (const auto& hand : hands) {
        cv::Point2f points[SOPAID_HAND_LANDMARK_COUNT];
        for (int i = 0; i < SOPAID_HAND_LANDMARK_COUNT; ++i) {
            points[i] = landmarkPoint(hand, i);
        }

        for (const auto& edge : connections) {
            const auto& a = points[edge[0]];
            const auto& b = points[edge[1]];
            cv::line(frame,
                     cv::Point(static_cast<int>(std::round(a.x)), static_cast<int>(std::round(a.y))),
                     cv::Point(static_cast<int>(std::round(b.x)), static_cast<int>(std::round(b.y))),
                     line_color, 2, cv::LINE_AA);
        }
        for (int i = 0; i < SOPAID_HAND_LANDMARK_COUNT; ++i) {
            const auto& p = points[i];
            cv::circle(frame, cv::Point(static_cast<int>(std::round(p.x)), static_cast<int>(std::round(p.y))), 3, point_color, -1, cv::LINE_AA);
        }
    }
}

bool isSameHandTrack(const SopAidHandResult& previous, const SopAidHandResult& current) {
    return previous.hand_id == current.hand_id && std::string(previous.handedness) == std::string(current.handedness);
}

void smoothHandResults(const std::vector<SopAidHandResult>& previous, std::vector<SopAidHandResult>& current) {
    constexpr float kCurrentWeight = 0.65f;
    constexpr float kPreviousWeight = 1.0f - kCurrentWeight;
    for (auto& hand : current) {
        const auto it = std::find_if(previous.begin(), previous.end(), [&hand](const SopAidHandResult& prev) {
            return isSameHandTrack(prev, hand);
        });
        if (it == previous.end()) continue;

        for (int i = 0; i < SOPAID_HAND_LANDMARK_COUNT; ++i) {
            hand.landmarks[i].x = it->landmarks[i].x * kPreviousWeight + hand.landmarks[i].x * kCurrentWeight;
            hand.landmarks[i].y = it->landmarks[i].y * kPreviousWeight + hand.landmarks[i].y * kCurrentWeight;
            hand.landmarks[i].z = it->landmarks[i].z * kPreviousWeight + hand.landmarks[i].z * kCurrentWeight;
        }
    }
}

void printUsage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "  SOPAID.exe [model_path] [video_path] [output_root] [confidence_threshold] [nms_threshold] [palm_model_path] [handpose_model_path] [sample_interval]" << std::endl;
    std::cout << "  All arguments are optional; defaults are defined in SopAidRequestConfig." << std::endl;
    std::cout << "  sample_interval only controls saved sample images, not inference frequency." << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args = collectArgs(argc, argv);
    SopAidRequestConfig request;
    if (args.size() > 1) request.model_path = args[1];
    if (args.size() > 2) request.video_path = args[2];
    if (args.size() > 3) request.output_root = args[3];
    if (args.size() > 6) request.palm_model_path = args[6];
    if (args.size() > 7) request.handpose_model_path = args[7];

    if (args.size() > 8 && (!parseInt(args[8].c_str(), request.sample_interval) || request.sample_interval <= 0)) {
        std::cout << "Invalid sample_interval: " << args[8] << std::endl;
        return 1;
    }

    if (args.size() > 4 && !parseFloat(args[4].c_str(), request.yolo.confidence_threshold)) {
        std::cout << "Invalid confidence_threshold: " << args[4] << std::endl;
        return 1;
    }

    if (args.size() > 5 && !parseFloat(args[5].c_str(), request.yolo.nms_threshold)) {
        std::cout << "Invalid nms_threshold: " << args[5] << std::endl;
        return 1;
    }

    // 本 EXE 负责调用 DLL 做检测
    request.bindPaths();

    const std::filesystem::path run_dir =
        makeRunDir(request.model_path, request.video_path, request.output_root);
    std::filesystem::create_directories(run_dir);
    const std::filesystem::path output_video_path = run_dir / "result.mp4";
    const std::filesystem::path params_path = run_dir / "params.json";
    const std::filesystem::path frame_results_path = run_dir / "frame_results.jsonl";
    const std::filesystem::path sample_frames_dir = run_dir / "sample_frames";
    std::filesystem::create_directories(sample_frames_dir);

    const std::filesystem::path log_dir = kDefaultLogDir;
    RunLogger logger(log_dir);

    const bool enable_hand_pose =
        !request.palm_model_path.empty() &&
        !request.handpose_model_path.empty() &&
        std::filesystem::exists(request.palm_model_path) &&
        std::filesystem::exists(request.handpose_model_path);

    writeParamsJson(
        params_path,
        output_video_path.string(),
        logger.path(),
        request,
        enable_hand_pose,
        sample_frames_dir.string());

    logger.info("C++ SOPAID video inference started.");
    logger.info("Run dir: " + run_dir.string());
    logger.info("Params: " + params_path.string());
    logger.info("Output video: " + output_video_path.string());
    logger.info("Frame results: " + frame_results_path.string());
    logger.info("Sample frames: " + sample_frames_dir.string());
    logger.info("Sample interval: " + std::to_string(request.sample_interval));
    logger.info("Inference mode: every frame.");

    // 一个推理对象初始化一次，后续每帧只传入 Mat 并接收 vector 结果。
    SopAidError err;
    sopaid::Inference inference;
    if (inference.Init(request.yolo, &err) != SopAidStatus::Ok) {
        logger.error(std::string("Init failed: ") + err.message);
        return 1;
    }

    SopAidHandHandle hand_handle = nullptr;
    if (enable_hand_pose) {
        SopAidError hand_err;
        hand_handle = sopaid::HandInit(request.hand, &hand_err);
        if (!hand_handle) {
            logger.error(std::string("Hand pose init failed: ") + hand_err.message);
            inference.Release();
            return 1;
        }
        logger.info("Hand pose ONNX initialized.");
    } else {
        logger.info("Hand pose disabled: palm/handpose ONNX model files were not found.");
    }

    cv::VideoCapture capture(request.video_path);
    if (!capture.isOpened()) {
        logger.error("Open video failed: " + request.video_path);
        if (hand_handle) sopaid::HandRelease(hand_handle);
        inference.Release();
        return 2;
    }

    const double fps = capture.get(cv::CAP_PROP_FPS) > 0.0 ? capture.get(cv::CAP_PROP_FPS) : 25.0;
    const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

    // 使用 H.264/AVC 写入 MP4；mp4v 在部分 Windows 播放器中会显示为 FMP4，
    // 容易被误判为文件损坏。
    cv::VideoWriter writer(
        output_video_path.string(),
        cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
        fps,
        cv::Size(width, height));
    if (!writer.isOpened()) {
        logger.error("Create output video failed: " + output_video_path.string());
        if (hand_handle) sopaid::HandRelease(hand_handle);
        inference.Release();
        return 2;
    }

    std::ofstream frame_results_file(frame_results_path, std::ios::out | std::ios::trunc);
    int processed_frames = 0;
    int source_frame_index = 0;
    int sample_frame_count = 0;
    size_t detected_frame_count = 0;
    size_t yolo_eval_count = 0;
    size_t hand_eval_count = 0;
    size_t total_detection_count = 0;
    double total_yolo_ms = 0.0;
    double total_hand_ms = 0.0;
    double last_yolo_ms = 0.0;
    double last_hand_ms = 0.0;
    double last_instant_fps = 0.0;
    std::vector<SopAidDetection> last_detections;
    std::vector<SopAidHandResult> last_hands;
    cv::Mat frame;
    while (capture.read(frame)) {
        const auto frame_start = std::chrono::steady_clock::now();

        std::vector<SopAidDetection> detections;

        const auto yolo_start = std::chrono::steady_clock::now();
        const SopAidStatus status = inference.Evaluate(frame, detections, &err);
        const auto yolo_end = std::chrono::steady_clock::now();
        if (status != SopAidStatus::Ok) {
            logger.error(std::string("Evaluate failed: ") + err.message);
            if (hand_handle) sopaid::HandRelease(hand_handle);
            inference.Release();
            return 3;
        }
        const auto hand_start = std::chrono::steady_clock::now();
        std::vector<SopAidHandResult> hands;
        if (hand_handle) {
            // 手部模型使用原始帧推理；先画 YOLO 框会把矩形和文字带入手部模型，导致关键点偏移。
            SopAidError hand_err;
            const SopAidStatus hand_status = sopaid::HandEvaluate(hand_handle, frame, hands, &hand_err);
            if (hand_status != SopAidStatus::Ok) {
                logger.error(std::string("Hand Evaluate failed: ") + hand_err.message);
                sopaid::HandRelease(hand_handle);
                inference.Release();
                return 3;
            }
        }
        const auto hand_end = std::chrono::steady_clock::now();

        smoothHandResults(last_hands, hands);
        last_yolo_ms = std::chrono::duration<double, std::milli>(yolo_end - yolo_start).count();
        last_hand_ms = std::chrono::duration<double, std::milli>(hand_end - hand_start).count();
        const auto frame_end = std::chrono::steady_clock::now();
        const double frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        last_instant_fps = frame_ms > 0.0 ? 1000.0 / frame_ms : 0.0;

        last_detections = std::move(detections);
        last_hands = std::move(hands);
        total_yolo_ms += last_yolo_ms;
        total_hand_ms += last_hand_ms;
        total_detection_count += last_detections.size();
        ++yolo_eval_count;
        ++hand_eval_count;
        ++detected_frame_count;

        writeFrameResultJsonl(frame_results_file, source_frame_index, source_frame_index / fps, last_yolo_ms, last_hand_ms, last_instant_fps, last_detections, last_hands);

        drawDetections(frame, last_detections);
        drawHandPoses(frame, last_hands);
        drawPerfOverlay(frame, last_instant_fps, last_yolo_ms, last_hand_ms, source_frame_index);
        if (source_frame_index % request.sample_interval == 0) {
            std::ostringstream filename;
            filename << "frame_" << std::setw(6) << std::setfill('0') << source_frame_index << ".jpg";
            if (cv::imwrite((sample_frames_dir / filename.str()).string(), frame)) {
                ++sample_frame_count;
            } else {
                logger.error("Save sample frame failed: " + (sample_frames_dir / filename.str()).string());
            }
        }
        writer.write(frame);
        ++processed_frames;
        ++source_frame_index;
        if (processed_frames % 30 == 0) {
            std::ostringstream progress;
            progress.setf(std::ios::fixed);
            progress.precision(2);
            progress << "Progress frames: " << processed_frames
                     << ", FPS: " << last_instant_fps
                     << ", YOLO ms: " << last_yolo_ms
                     << ", Hand ms: " << last_hand_ms
                     << ", samples: " << sample_frame_count;
            logger.info(progress.str());
        }
    }

    // 显式关闭视频，确保 MP4 的索引信息完整写入后再报告保存成功。
    writer.release();
    capture.release();
    if (hand_handle) sopaid::HandRelease(hand_handle);
    inference.Release();
    logger.info("Output video saved: " + output_video_path.string());
    logger.info("Processed frames: " + std::to_string(processed_frames));
    logger.info("Detected frames: " + std::to_string(detected_frame_count));
    logger.info("Sample frame count: " + std::to_string(sample_frame_count));
    logger.info("YOLO eval count: " + std::to_string(yolo_eval_count));
    logger.info("Hand eval count: " + std::to_string(hand_eval_count));
    logger.info("Total detection count: " + std::to_string(total_detection_count));
    if (yolo_eval_count > 0) {
        logger.info("Average YOLO infer ms: " + std::to_string(total_yolo_ms / yolo_eval_count));
    }
    if (hand_eval_count > 0) {
        logger.info("Average hand infer ms: " + std::to_string(total_hand_ms / hand_eval_count));
    }
    if (detected_frame_count > 0) {
        logger.info("Average active frame FPS: " + std::to_string(1000.0 / ((total_yolo_ms + total_hand_ms) / detected_frame_count)));
    }
    logger.info("Frame results saved: " + frame_results_path.string());
    logger.info("C++ SOPAID video inference finished.");
    return 0;
}
