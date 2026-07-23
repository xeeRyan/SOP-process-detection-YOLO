#include "pch.h"

#include "SOPAID_wrapper.h"

#include <msclr/marshal_cppstd.h>

using namespace msclr::interop;

namespace {

	std::string ToStdString(String^ value)
	{
		return value == nullptr ? std::string() : marshal_as<std::string>(value);
	}

	String^ ToManagedString(const char* value)
	{
		return value == nullptr ? String::Empty : marshal_as<String^>(std::string(value));
	}

	SopAidModelFormat ToNativeModelFormat(SOPAIDwrapper::ModelFormat format)
	{
		return static_cast<SopAidModelFormat>(format);
	}

	SOPAIDwrapper::InferenceStatus ToManagedStatus(SopAidStatus status)
	{
		return static_cast<SOPAIDwrapper::InferenceStatus>(status);
	}

	bool TryCreateInputMat(array<Byte>^ imageData,
		int width,
		int height,
		int channels,
		int stride,
		cv::Mat& image,
		String^% errorMessage)
	{
		if (imageData == nullptr || imageData->Length == 0) {
			errorMessage = "imageData is null or empty.";
			return false;
		}

		if (width <= 0 || height <= 0) {
			errorMessage = "width and height must be positive.";
			return false;
		}

		if (channels != 1 && channels != 3 && channels != 4) {
			errorMessage = "channels must be 1, 3, or 4.";
			return false;
		}

		const int minStride = width * channels;
		if (stride < minStride) {
			errorMessage = "stride is smaller than width * channels.";
			return false;
		}

		const long long requiredBytes = static_cast<long long>(stride) * height;
		if (imageData->LongLength < requiredBytes) {
			errorMessage = "imageData length is smaller than stride * height.";
			return false;
		}

		return true;
	}

}

namespace SOPAIDwrapper {

	InferenceErrorInfo::InferenceErrorInfo()
	{
		Status = InferenceStatus::Ok;
		Message = String::Empty;
	}

	InferenceErrorInfo::InferenceErrorInfo(InferenceStatus status, String^ message)
	{
		Status = status;
		Message = message == nullptr ? String::Empty : message;
	}

	bool InferenceErrorInfo::Succeeded::get()
	{
		return Status == InferenceStatus::Ok;
	}

	InferenceConfig::InferenceConfig()
	{
		ModelPath = String::Empty;
		Format = ModelFormat::Auto;
		InputWidth = 640;
		InputHeight = 640;
		ConfidenceThreshold = 0.25f;
		NmsThreshold = 0.70f;
		ClassNamesCsv = "bearing,cover,tool";
		UseCuda = false;
		DeviceId = 0;
	}

	float DetectionResult::Width::get()
	{
		return X2 - X1;
	}

	float DetectionResult::Height::get()
	{
		return Y2 - Y1;
	}

	InferenceEvaluator::InferenceEvaluator()
		: inference_(nullptr),
		lastError_(gcnew InferenceErrorInfo())
	{
	}

	InferenceEvaluator::InferenceEvaluator(InferenceConfig^ config)
		: inference_(nullptr),
		lastError_(gcnew InferenceErrorInfo())
	{
		Init(config);
	}

	InferenceEvaluator::~InferenceEvaluator()
	{
		this->!InferenceEvaluator();
		GC::SuppressFinalize(this);
	}

	InferenceEvaluator::!InferenceEvaluator()
	{
		if (inference_ != nullptr) {
			inference_->Release();
			delete inference_;
			inference_ = nullptr;
		}
	}

	bool InferenceEvaluator::Init(InferenceConfig^ config)
	{
		if (config == nullptr) {
			SetLastError(InferenceStatus::InvalidArgument, "config is null.");
			return false;
		}

		if (String::IsNullOrWhiteSpace(config->ModelPath)) {
			SetLastError(InferenceStatus::InvalidArgument, "ModelPath is empty.");
			return false;
		}

		std::string modelPath = ToStdString(config->ModelPath);
		std::string classNames = ToStdString(config->ClassNamesCsv);

		SopAidInitConfig nativeConfig{};
		nativeConfig.model_path = modelPath.c_str();
		nativeConfig.model_format = ToNativeModelFormat(config->Format);
		nativeConfig.input_width = config->InputWidth;
		nativeConfig.input_height = config->InputHeight;
		nativeConfig.confidence_threshold = config->ConfidenceThreshold;
		nativeConfig.nms_threshold = config->NmsThreshold;
		nativeConfig.class_names_csv = classNames.empty() ? nullptr : classNames.c_str();
		nativeConfig.use_cuda = config->UseCuda;
		nativeConfig.device_id = config->DeviceId;

		if (inference_ == nullptr) {
			inference_ = new sopaid::Inference();
		}
		else {
			inference_->Release();
		}

		SopAidError error{};
		const SopAidStatus status = inference_->Init(nativeConfig, &error);
		SetLastError(error);
		return status == SopAidStatus::Ok;
	}

	bool InferenceEvaluator::Evaluate(array<Byte>^ imageData, int width, int height, int channels, List<DetectionResult^>^ results)
	{
		return Evaluate(imageData, width, height, channels, width * channels, results);
	}

	bool InferenceEvaluator::Evaluate(array<Byte>^ imageData,
		int width,
		int height,
		int channels,
		int stride,
		List<DetectionResult^>^ results)
	{
		if (results == nullptr) {
			SetLastError(InferenceStatus::InvalidArgument, "results is null.");
			return false;
		}
		results->Clear();

		if (inference_ == nullptr || !inference_->IsInitialized()) {
			SetLastError(InferenceStatus::InvalidArgument, "Inference evaluator is not initialized.");
			return false;
		}

		String^ validationError = nullptr;
		cv::Mat unused;
		if (!TryCreateInputMat(imageData, width, height, channels, stride, unused, validationError)) {
			SetLastError(InferenceStatus::InvalidArgument, validationError);
			return false;
		}

		pin_ptr<Byte> pinned = &imageData[0];
		cv::Mat image(height, width, CV_MAKETYPE(CV_8U, channels), pinned, stride);

		std::vector<SopAidDetection> nativeResults;
		SopAidError error{};
		const SopAidStatus status = inference_->Evaluate(image, nativeResults, &error);
		SetLastError(error);
		if (status != SopAidStatus::Ok) {
			return false;
		}

		for (const SopAidDetection& item : nativeResults) {
			DetectionResult^ result = gcnew DetectionResult();
			result->ClassId = item.class_id;
			result->ClassName = ToManagedString(item.class_name);
			result->Confidence = item.confidence;
			result->X1 = item.x1;
			result->Y1 = item.y1;
			result->X2 = item.x2;
			result->Y2 = item.y2;
			results->Add(result);
		}

		return true;
	}

	bool InferenceEvaluator::Release()
	{
		if (inference_ != nullptr) {
			inference_->Release();
			delete inference_;
			inference_ = nullptr;
		}

		SetLastError(InferenceStatus::Ok, String::Empty);
		return true;
	}

	bool InferenceEvaluator::IsInitialized::get()
	{
		return inference_ != nullptr && inference_->IsInitialized();
	}

	InferenceErrorInfo^ InferenceEvaluator::LastError::get()
	{
		return lastError_;
	}

	void InferenceEvaluator::SetLastError(InferenceStatus status, String^ message)
	{
		lastError_ = gcnew InferenceErrorInfo(status, message);
	}

	void InferenceEvaluator::SetLastError(const SopAidError& error)
	{
		SetLastError(ToManagedStatus(error.status), ToManagedString(error.message));
	}

	HandPoseConfig::HandPoseConfig()
	{
		ModelPath = String::Empty;
		PalmModelPath = String::Empty;
		HandPoseModelPath = String::Empty;
		MaxHands = 2;
		PalmScoreThreshold = 0.5f;
		HandScoreThreshold = 0.5f;
		UseGpu = false;
		DeviceId = 0;
	}

	HandPoseResult::HandPoseResult()
	{
		Landmarks = gcnew List<HandLandmarkResult^>();
	}

	HandPoseEvaluator::HandPoseEvaluator()
		: handle_(nullptr),
		initialized_(false),
		lastError_(gcnew InferenceErrorInfo())
	{
	}

	HandPoseEvaluator::HandPoseEvaluator(HandPoseConfig^ config)
		: handle_(nullptr),
		initialized_(false),
		lastError_(gcnew InferenceErrorInfo())
	{
		Init(config);
	}

	HandPoseEvaluator::~HandPoseEvaluator()
	{
		this->!HandPoseEvaluator();
		GC::SuppressFinalize(this);
	}

	HandPoseEvaluator::!HandPoseEvaluator()
	{
		if (handle_ != nullptr) {
			sopaid::HandRelease(handle_);
			handle_ = nullptr;
			initialized_ = false;
		}
	}

	bool HandPoseEvaluator::Init(HandPoseConfig^ config)
	{
		if (config == nullptr) {
			SetLastError(InferenceStatus::InvalidArgument, "config is null.");
			return false;
		}

		std::string modelPath = ToStdString(config->ModelPath);
		std::string palmModelPath = ToStdString(config->PalmModelPath);
		std::string handPoseModelPath = ToStdString(config->HandPoseModelPath);

		if (modelPath.empty() && (palmModelPath.empty() || handPoseModelPath.empty())) {
			SetLastError(InferenceStatus::InvalidArgument, "Set ModelPath, or set both PalmModelPath and HandPoseModelPath.");
			return false;
		}

		SopAidHandInitConfig nativeConfig{};
		nativeConfig.model_path = modelPath.empty() ? nullptr : modelPath.c_str();
		nativeConfig.palm_model_path = palmModelPath.empty() ? nullptr : palmModelPath.c_str();
		nativeConfig.handpose_model_path = handPoseModelPath.empty() ? nullptr : handPoseModelPath.c_str();
		nativeConfig.max_num_hands = config->MaxHands;
		nativeConfig.min_hand_detection_confidence = config->PalmScoreThreshold;
		nativeConfig.min_hand_presence_confidence = config->HandScoreThreshold;
		nativeConfig.min_tracking_confidence = config->HandScoreThreshold;
		nativeConfig.use_gpu = config->UseGpu;
		nativeConfig.device_id = config->DeviceId;

		if (handle_ != nullptr) {
			sopaid::HandRelease(handle_);
			handle_ = nullptr;
			initialized_ = false;
		}

		SopAidError error{};
		handle_ = sopaid::HandInit(nativeConfig, &error);
		initialized_ = handle_ != nullptr && error.status == SopAidStatus::Ok;
		SetLastError(error);
		return initialized_;
	}

	bool HandPoseEvaluator::Evaluate(array<Byte>^ imageData, int width, int height, int channels, List<HandPoseResult^>^ results)
	{
		return Evaluate(imageData, width, height, channels, width * channels, results);
	}

	bool HandPoseEvaluator::Evaluate(array<Byte>^ imageData,
		int width,
		int height,
		int channels,
		int stride,
		List<HandPoseResult^>^ results)
	{
		if (results == nullptr) {
			SetLastError(InferenceStatus::InvalidArgument, "results is null.");
			return false;
		}
		results->Clear();

		if (handle_ == nullptr || !initialized_) {
			SetLastError(InferenceStatus::InvalidArgument, "Hand pose evaluator is not initialized.");
			return false;
		}

		String^ validationError = nullptr;
		cv::Mat unused;
		if (!TryCreateInputMat(imageData, width, height, channels, stride, unused, validationError)) {
			SetLastError(InferenceStatus::InvalidArgument, validationError);
			return false;
		}

		pin_ptr<Byte> pinned = &imageData[0];
		cv::Mat image(height, width, CV_MAKETYPE(CV_8U, channels), pinned, stride);

		std::vector<SopAidHandResult> nativeResults;
		SopAidError error{};
		const SopAidStatus status = sopaid::HandEvaluate(handle_, image, nativeResults, &error);
		SetLastError(error);
		if (status != SopAidStatus::Ok) {
			return false;
		}

		for (const SopAidHandResult& item : nativeResults) {
			HandPoseResult^ result = gcnew HandPoseResult();
			result->HandId = item.hand_id;
			result->Confidence = item.confidence;
			result->Handedness = ToManagedString(item.handedness);

			for (int i = 0; i < SOPAID_HAND_LANDMARK_COUNT; ++i) {
				HandLandmarkResult^ landmark = gcnew HandLandmarkResult();
				landmark->X = item.landmarks[i].x;
				landmark->Y = item.landmarks[i].y;
				landmark->Z = item.landmarks[i].z;
				landmark->Visibility = item.landmarks[i].visibility;
				result->Landmarks->Add(landmark);
			}

			results->Add(result);
		}

		return true;
	}

	bool HandPoseEvaluator::Release()
	{
		if (handle_ != nullptr) {
			sopaid::HandRelease(handle_);
			handle_ = nullptr;
		}

		initialized_ = false;
		SetLastError(InferenceStatus::Ok, String::Empty);
		return true;
	}

	bool HandPoseEvaluator::IsInitialized::get()
	{
		return initialized_ && handle_ != nullptr;
	}

	InferenceErrorInfo^ HandPoseEvaluator::LastError::get()
	{
		return lastError_;
	}

	void HandPoseEvaluator::SetLastError(InferenceStatus status, String^ message)
	{
		lastError_ = gcnew InferenceErrorInfo(status, message);
	}

	void HandPoseEvaluator::SetLastError(const SopAidError& error)
	{
		SetLastError(ToManagedStatus(error.status), ToManagedString(error.message));
	}
}
