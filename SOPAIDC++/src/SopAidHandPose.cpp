#include "SopAidHandPose.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr int kPalmInputSize = 192;
constexpr int kHandInputSize = 224;
constexpr float kPalmNmsThreshold = 0.30f;

const int kHandConnections[][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 4},
    {0, 5}, {5, 6}, {6, 7}, {7, 8},
    {0, 9}, {9, 10}, {10, 11}, {11, 12},
    {0, 13}, {13, 14}, {14, 15}, {15, 16},
    {0, 17}, {17, 18}, {18, 19}, {19, 20}
};

struct PalmDetection {
    cv::Rect2f box;
    cv::Point2f landmarks[7];
    float score = 0.0f;
};

struct HandPoseContext {
    cv::dnn::Net palm_net;
    cv::dnn::Net hand_net;
    std::vector<cv::Point2f> anchors;
    SopAidHandInitConfig config;
    std::string palm_path;
    std::string handpose_path;
};

void SetHandError(SopAidError* error, SopAidStatus status, const std::string& message) {
    if (!error) return;
    error->status = status;
    std::memset(error->message, 0, sizeof(error->message));
    const auto count = std::min(message.size(), sizeof(error->message) - 1);
    std::memcpy(error->message, message.data(), count);
}

std::string chooseHandPosePath(const SopAidHandInitConfig& config) {
    if (config.handpose_model_path && config.handpose_model_path[0] != '\0') return config.handpose_model_path;
    if (config.model_path && config.model_path[0] != '\0') return config.model_path;
    return {};
}

SopAidStatus ValidateHandConfig(const SopAidHandInitConfig& config, std::string& palm_path, std::string& handpose_path, SopAidError* error) {
    palm_path = config.palm_model_path ? config.palm_model_path : "";
    handpose_path = chooseHandPosePath(config);
    if (palm_path.empty()) {
        SetHandError(error, SopAidStatus::InvalidArgument, "palm_model_path is empty.");
        return SopAidStatus::InvalidArgument;
    }
    if (handpose_path.empty()) {
        SetHandError(error, SopAidStatus::InvalidArgument, "handpose_model_path is empty.");
        return SopAidStatus::InvalidArgument;
    }
    if (!std::filesystem::exists(palm_path)) {
        SetHandError(error, SopAidStatus::FileNotFound, "Palm model file does not exist: " + palm_path);
        return SopAidStatus::FileNotFound;
    }
    if (!std::filesystem::exists(handpose_path)) {
        SetHandError(error, SopAidStatus::FileNotFound, "Handpose model file does not exist: " + handpose_path);
        return SopAidStatus::FileNotFound;
    }
    if (config.max_num_hands <= 0) {
        SetHandError(error, SopAidStatus::InvalidArgument, "max_num_hands must be positive.");
        return SopAidStatus::InvalidArgument;
    }
    return SopAidStatus::Ok;
}

std::vector<cv::Point2f> createPalmAnchors() {
    std::vector<cv::Point2f> anchors;
    anchors.reserve(2016);
    auto appendGrid = [&anchors](int grid, int repeat) {
        for (int y = 0; y < grid; ++y) {
            for (int x = 0; x < grid; ++x) {
                const cv::Point2f center((x + 0.5f) / grid, (y + 0.5f) / grid);
                for (int i = 0; i < repeat; ++i) anchors.push_back(center);
            }
        }
    };
    appendGrid(24, 2);
    appendGrid(12, 6);
    return anchors;
}

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

float rectIou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float inter = (a & b).area();
    const float uni = a.area() + b.area() - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<int> nmsPalm(const std::vector<PalmDetection>& detections, float threshold, int top_k) {
    std::vector<int> order(detections.size());
    for (int i = 0; i < static_cast<int>(order.size()); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&detections](int a, int b) {
        return detections[a].score > detections[b].score;
    });
    if (top_k > 0 && static_cast<int>(order.size()) > top_k) order.resize(top_k);

    std::vector<int> keep;
    for (int idx : order) {
        bool suppressed = false;
        for (int kept : keep) {
            if (rectIou(detections[idx].box, detections[kept].box) > threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) keep.push_back(idx);
    }
    return keep;
}

cv::Mat makeNhwcBlobFromRgb(const cv::Mat& rgb_image, int width, int height) {
    cv::Mat resized;
    cv::resize(rgb_image, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_AREA);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    int dims[] = {1, height, width, 3};
    cv::Mat blob(4, dims, CV_32F);
    std::memcpy(blob.ptr<float>(), resized.ptr<float>(), static_cast<size_t>(height) * width * 3 * sizeof(float));
    return blob;
}

cv::Mat preprocessPalm(const cv::Mat& image, cv::Point& pad_bias, float& ratio) {
    ratio = std::min(static_cast<float>(kPalmInputSize) / image.rows, static_cast<float>(kPalmInputSize) / image.cols);
    const int resized_h = std::max(1, static_cast<int>(image.rows * ratio));
    const int resized_w = std::max(1, static_cast<int>(image.cols * ratio));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_AREA);

    const int pad_h = kPalmInputSize - resized_h;
    const int pad_w = kPalmInputSize - resized_w;
    const int left = pad_w / 2;
    const int top = pad_h / 2;
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, pad_h - top, left, pad_w - left, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    pad_bias = cv::Point(static_cast<int>(left / ratio), static_cast<int>(top / ratio));
    return makeNhwcBlobFromRgb(rgb, kPalmInputSize, kPalmInputSize);
}

std::vector<PalmDetection> detectPalms(HandPoseContext& context, const cv::Mat& image) {
    cv::Point pad_bias;
    float ratio = 1.0f;
    cv::Mat input = preprocessPalm(image, pad_bias, ratio);
    context.palm_net.setInput(input);

    std::vector<cv::Mat> outputs;
    context.palm_net.forward(outputs, context.palm_net.getUnconnectedOutLayersNames());
    if (outputs.size() < 2 || outputs[0].total() < 2016 * 18 || outputs[1].total() < 2016) return {};

    const float* box_data = reinterpret_cast<const float*>(outputs[0].data);
    const float* score_data = reinterpret_cast<const float*>(outputs[1].data);
    const float scale = static_cast<float>(std::max(image.cols, image.rows));

    std::vector<PalmDetection> candidates;
    for (int i = 0; i < 2016 && i < static_cast<int>(context.anchors.size()); ++i) {
        const float score = sigmoid(score_data[i]);
        if (score < context.config.min_hand_detection_confidence) continue;

        const float* delta = box_data + i * 18;
        const cv::Point2f anchor = context.anchors[i];
        const float cx = delta[0] / kPalmInputSize + anchor.x;
        const float cy = delta[1] / kPalmInputSize + anchor.y;
        const float bw = delta[2] / kPalmInputSize;
        const float bh = delta[3] / kPalmInputSize;

        PalmDetection det;
        det.box = cv::Rect2f((cx - bw * 0.5f) * scale - pad_bias.x,
                             (cy - bh * 0.5f) * scale - pad_bias.y,
                             bw * scale,
                             bh * scale);
        det.score = score;
        for (int j = 0; j < 7; ++j) {
            det.landmarks[j].x = (delta[4 + j * 2] / kPalmInputSize + anchor.x) * scale - pad_bias.x;
            det.landmarks[j].y = (delta[5 + j * 2] / kPalmInputSize + anchor.y) * scale - pad_bias.y;
        }
        candidates.push_back(det);
    }

    const std::vector<int> keep = nmsPalm(candidates, kPalmNmsThreshold, 5000);
    std::vector<PalmDetection> results;
    for (int idx : keep) {
        results.push_back(candidates[idx]);
        if (static_cast<int>(results.size()) >= context.config.max_num_hands) break;
    }
    return results;
}

bool cropAndPadFromPalm(const cv::Mat& image, const cv::Rect2f& input_box, bool for_rotation,
                        cv::Mat& output, cv::Rect2f& output_box, cv::Point2f& bias) {
    cv::Rect2f box = input_box;
    const cv::Point2f shift = for_rotation ? cv::Point2f(0.0f, 0.0f) : cv::Point2f(0.0f, -0.4f);
    box.x += shift.x * box.width;
    box.y += shift.y * box.height;

    const float enlarge = for_rotation ? 4.0f : 3.0f;
    const cv::Point2f center(box.x + box.width * 0.5f, box.y + box.height * 0.5f);
    const cv::Size2f new_size(box.width * enlarge, box.height * enlarge);
    box = cv::Rect2f(center.x - new_size.width * 0.5f, center.y - new_size.height * 0.5f, new_size.width, new_size.height);

    const int x1 = std::clamp(static_cast<int>(std::floor(box.x)), 0, image.cols - 1);
    const int y1 = std::clamp(static_cast<int>(std::floor(box.y)), 0, image.rows - 1);
    const int x2 = std::clamp(static_cast<int>(std::ceil(box.x + box.width)), 0, image.cols);
    const int y2 = std::clamp(static_cast<int>(std::ceil(box.y + box.height)), 0, image.rows);
    if (x2 <= x1 || y2 <= y1) return false;

    output_box = cv::Rect2f(static_cast<float>(x1), static_cast<float>(y1), static_cast<float>(x2 - x1), static_cast<float>(y2 - y1));
    cv::Mat crop = image(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
    const int side_len = for_rotation
        ? std::max(1, static_cast<int>(std::sqrt(static_cast<float>(crop.rows * crop.rows + crop.cols * crop.cols))))
        : std::max(crop.rows, crop.cols);
    const int pad_h = side_len - crop.rows;
    const int pad_w = side_len - crop.cols;
    const int left = pad_w / 2;
    const int top = pad_h / 2;
    cv::copyMakeBorder(crop, output, top, pad_h - top, left, pad_w - left, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    bias = cv::Point2f(output_box.x - left, output_box.y - top);
    return !output.empty();
}

cv::Point2f applyAffine(const cv::Matx23f& matrix, const cv::Point2f& point) {
    return cv::Point2f(
        matrix(0, 0) * point.x + matrix(0, 1) * point.y + matrix(0, 2),
        matrix(1, 0) * point.x + matrix(1, 1) * point.y + matrix(1, 2));
}

bool inferHand(HandPoseContext& context, const cv::Mat& image, const PalmDetection& palm, int hand_id, SopAidHandResult& result) {
    cv::Mat interest;
    cv::Rect2f palm_box;
    cv::Point2f pad_bias;
    if (!cropAndPadFromPalm(image, palm.box, true, interest, palm_box, pad_bias)) return false;

    cv::Mat rgb;
    cv::cvtColor(interest, rgb, cv::COLOR_BGR2RGB);

    cv::Point2f local_landmarks[7];
    for (int i = 0; i < 7; ++i) local_landmarks[i] = palm.landmarks[i] - pad_bias;
    cv::Rect2f local_box = palm_box;
    local_box.x -= pad_bias.x;
    local_box.y -= pad_bias.y;

    const cv::Point2f p1 = local_landmarks[0];
    const cv::Point2f p2 = local_landmarks[2];
    float radians = static_cast<float>(CV_PI / 2.0 - std::atan2(-(p2.y - p1.y), p2.x - p1.x));
    radians = static_cast<float>(radians - 2.0 * CV_PI * std::floor((radians + CV_PI) / (2.0 * CV_PI)));
    const float angle = radians * 180.0f / static_cast<float>(CV_PI);
    const cv::Point2f center(local_box.x + local_box.width * 0.5f, local_box.y + local_box.height * 0.5f);

    cv::Mat rotation_mat = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Matx23f rot;
    for (int r = 0; r < 2; ++r) for (int c = 0; c < 3; ++c) rot(r, c) = static_cast<float>(rotation_mat.at<double>(r, c));

    cv::Mat rotated;
    cv::warpAffine(rgb, rotated, rotation_mat, rgb.size());

    cv::Point2f min_pt(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    cv::Point2f max_pt(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    for (const auto& p : local_landmarks) {
        const cv::Point2f rp = applyAffine(rot, p);
        min_pt.x = std::min(min_pt.x, rp.x);
        min_pt.y = std::min(min_pt.y, rp.y);
        max_pt.x = std::max(max_pt.x, rp.x);
        max_pt.y = std::max(max_pt.y, rp.y);
    }

    cv::Mat hand_crop;
    cv::Rect2f rotated_palm_box;
    cv::Point2f hand_crop_bias;
    if (!cropAndPadFromPalm(rotated, cv::Rect2f(min_pt.x, min_pt.y, max_pt.x - min_pt.x, max_pt.y - min_pt.y), false,
                            hand_crop, rotated_palm_box, hand_crop_bias)) {
        return false;
    }

    cv::Mat blob = makeNhwcBlobFromRgb(hand_crop, kHandInputSize, kHandInputSize);
    context.hand_net.setInput(blob);
    std::vector<cv::Mat> outputs;
    context.hand_net.forward(outputs, context.hand_net.getUnconnectedOutLayersNames());
    if (outputs.size() < 4 || outputs[0].total() < 63 || outputs[1].total() < 1 || outputs[2].total() < 1) return false;

    const float confidence = outputs[1].ptr<float>()[0];
    if (confidence < context.config.min_hand_presence_confidence) return false;
    const float handedness = outputs[2].ptr<float>()[0];
    const float* lm = outputs[0].ptr<float>();

    cv::Mat inv_rotation_mat;
    cv::invertAffineTransform(rotation_mat, inv_rotation_mat);
    cv::Matx23f inv_rot;
    for (int r = 0; r < 2; ++r) for (int c = 0; c < 3; ++c) inv_rot(r, c) = static_cast<float>(inv_rotation_mat.at<double>(r, c));

    // handpose 模型输出的是 224x224 输入图上的局部坐标。
    // 还原顺序必须严格反向执行：模型输入 -> 手部正方形 crop -> 旋转后的临时图 -> 原始图。
    // 之前用中心点近似反推，遇到补边或手掌旋转时会把 wrist(0) 和掌根整体拉歪。
    const float crop_to_model_x = static_cast<float>(hand_crop.cols) / kHandInputSize;
    const float crop_to_model_y = static_cast<float>(hand_crop.rows) / kHandInputSize;
    const float depth_scale = std::max(crop_to_model_x, crop_to_model_y);

    result = SopAidHandResult{};
    result.hand_id = hand_id;
    result.confidence = confidence;
    strncpy_s(result.handedness, sizeof(result.handedness), handedness >= 0.5f ? "right" : "left", _TRUNCATE);

    for (int i = 0; i < SOPAID_HAND_LANDMARK_COUNT; ++i) {
        const cv::Point2f point_in_crop(lm[i * 3] * crop_to_model_x,
                                        lm[i * 3 + 1] * crop_to_model_y);
        const cv::Point2f point_in_rotated = point_in_crop + hand_crop_bias;
        const cv::Point2f point_in_interest = applyAffine(inv_rot, point_in_rotated);
        const cv::Point2f point_in_image = point_in_interest + pad_bias;

        result.landmarks[i].x = std::clamp(point_in_image.x, 0.0f, static_cast<float>(image.cols - 1));
        result.landmarks[i].y = std::clamp(point_in_image.y, 0.0f, static_cast<float>(image.rows - 1));
        result.landmarks[i].z = lm[i * 3 + 2] * depth_scale;
        result.landmarks[i].visibility = confidence;
    }
    return true;
}

}  // namespace

extern "C" SOPAID_API SopAidHandHandle SopAidHand_Init(const SopAidHandInitConfig* config, SopAidError* error) {
    if (!config) {
        SetHandError(error, SopAidStatus::InvalidArgument, "hand config is null.");
        return nullptr;
    }

    std::string palm_path;
    std::string handpose_path;
    const SopAidStatus status = ValidateHandConfig(*config, palm_path, handpose_path, error);
    if (status != SopAidStatus::Ok) return nullptr;

    try {
        auto* context = new HandPoseContext();
        context->config = *config;
        context->palm_path = palm_path;
        context->handpose_path = handpose_path;
        context->anchors = createPalmAnchors();
        context->palm_net = cv::dnn::readNet(palm_path);
        context->hand_net = cv::dnn::readNet(handpose_path);

        // OpenCV Zoo 的手部 ONNX 模型使用 OpenCV DNN CPU 路径最稳定。
        context->palm_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        context->palm_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        context->hand_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        context->hand_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        SetHandError(error, SopAidStatus::Ok, "");
        return context;
    } catch (const cv::Exception& ex) {
        SetHandError(error, SopAidStatus::BackendError, std::string("Init hand ONNX failed: ") + ex.what());
    } catch (const std::exception& ex) {
        SetHandError(error, SopAidStatus::BackendError, std::string("Init hand ONNX failed: ") + ex.what());
    }
    return nullptr;
}

extern "C" SOPAID_API void SopAidHand_Release(SopAidHandHandle handle) {
    delete static_cast<HandPoseContext*>(handle);
}

namespace sopaid {

SOPAID_API SopAidHandHandle HandInit(const SopAidHandInitConfig& config, SopAidError* error) {
    return SopAidHand_Init(&config, error);
}

SOPAID_API SopAidStatus HandEvaluate(
    SopAidHandHandle handle,
    const cv::Mat& image,
    std::vector<SopAidHandResult>& results,
    SopAidError* error) {
    results.clear();
    if (!handle) {
        SetHandError(error, SopAidStatus::InvalidArgument, "hand handle is null.");
        return SopAidStatus::InvalidArgument;
    }
    if (image.empty()) {
        SetHandError(error, SopAidStatus::InvalidArgument, "hand Evaluate image is empty.");
        return SopAidStatus::InvalidArgument;
    }

    auto* context = static_cast<HandPoseContext*>(handle);
    try {
        const std::vector<PalmDetection> palms = detectPalms(*context, image);
        int hand_id = 0;
        for (const auto& palm : palms) {
            SopAidHandResult hand;
            if (inferHand(*context, image, palm, hand_id, hand)) {
                results.push_back(hand);
                ++hand_id;
            }
        }
        SetHandError(error, SopAidStatus::Ok, "");
        return SopAidStatus::Ok;
    } catch (const cv::Exception& ex) {
        SetHandError(error, SopAidStatus::BackendError, std::string("Hand Evaluate failed: ") + ex.what());
    } catch (const std::exception& ex) {
        SetHandError(error, SopAidStatus::BackendError, std::string("Hand Evaluate failed: ") + ex.what());
    }
    return SopAidStatus::BackendError;
}

SOPAID_API void HandRelease(SopAidHandHandle handle) {
    SopAidHand_Release(handle);
}

}  // namespace sopaid
