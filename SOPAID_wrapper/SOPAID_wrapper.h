#pragma once

#pragma push_macro("IServiceProvider")
#define IServiceProvider NativeIServiceProvider
#include <vector>
#include "SopAidHandPose.h"
#include "SopAidInfer.h"
#pragma pop_macro("IServiceProvider")

using namespace System;
using namespace System::Collections::Generic;

namespace SOPAIDwrapper {

	public enum class ModelFormat
	{
		Auto = 0,
		Pt = 1,
		Onnx = 2,
		Engine = 3
	};

	public enum class InferenceStatus
	{
		Ok = 0,
		InvalidArgument = 1,
		FileNotFound = 2,
		UnsupportedModel = 3,
		BackendError = 4,
		InferenceError = 5
	};

	public ref class InferenceErrorInfo
	{
	public:
		InferenceErrorInfo();
		InferenceErrorInfo(InferenceStatus status, String^ message);

		property InferenceStatus Status;
		property String^ Message;
		property bool Succeeded { bool get(); }
	};

	public ref class InferenceConfig
	{
	public:
		InferenceConfig();

		property String^ ModelPath;
		property ModelFormat Format;
		property int InputWidth;
		property int InputHeight;
		property float ConfidenceThreshold;
		property float NmsThreshold;
		property String^ ClassNamesCsv;
		property bool UseCuda;
		property int DeviceId;
	};

	public ref class DetectionResult
	{
	public:
		property int ClassId;
		property String^ ClassName;
		property float Confidence;
		property float X1;
		property float Y1;
		property float X2;
		property float Y2;
		property float Width { float get(); }
		property float Height { float get(); }
	};

	public ref class InferenceEvaluator
	{
	public:
		InferenceEvaluator();
		InferenceEvaluator(InferenceConfig^ config);
		~InferenceEvaluator();
		!InferenceEvaluator();

		bool Init(InferenceConfig^ config);
		bool Evaluate(array<Byte>^ imageData, int width, int height, int channels, List<DetectionResult^>^ results);
		bool Evaluate(array<Byte>^ imageData, int width, int height, int channels, int stride, List<DetectionResult^>^ results);
		bool Release();

		property bool IsInitialized { bool get(); }
		property InferenceErrorInfo^ LastError { InferenceErrorInfo^ get(); }

	private:
		sopaid::Inference* inference_;
		InferenceErrorInfo^ lastError_;

		void SetLastError(InferenceStatus status, String^ message);
		void SetLastError(const SopAidError& error);
	};

	public ref class HandPoseConfig
	{
	public:
		HandPoseConfig();

		property String^ ModelPath;
		property String^ PalmModelPath;
		property String^ HandPoseModelPath;
		property int MaxHands;
		property float PalmScoreThreshold;
		property float HandScoreThreshold;
		property bool UseGpu;
		property int DeviceId;
	};

	public ref class HandLandmarkResult
	{
	public:
		property float X;
		property float Y;
		property float Z;
		property float Visibility;
	};

	public ref class HandPoseResult
	{
	public:
		HandPoseResult();

		property int HandId;
		property float Confidence;
		property String^ Handedness;
		property List<HandLandmarkResult^>^ Landmarks;
	};

	public ref class HandPoseEvaluator
	{
	public:
		HandPoseEvaluator();
		HandPoseEvaluator(HandPoseConfig^ config);
		~HandPoseEvaluator();
		!HandPoseEvaluator();

		bool Init(HandPoseConfig^ config);
		bool Evaluate(array<Byte>^ imageData, int width, int height, int channels, List<HandPoseResult^>^ results);
		bool Evaluate(array<Byte>^ imageData, int width, int height, int channels, int stride, List<HandPoseResult^>^ results);
		bool Release();

		property bool IsInitialized { bool get(); }
		property InferenceErrorInfo^ LastError { InferenceErrorInfo^ get(); }

	private:
		SopAidHandHandle handle_;
		bool initialized_;
		InferenceErrorInfo^ lastError_;

		void SetLastError(InferenceStatus status, String^ message);
		void SetLastError(const SopAidError& error);
	};
}
