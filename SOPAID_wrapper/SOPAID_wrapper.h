#pragma once

struct SopAidError;
namespace sopaid {
	class Inference;
}

using namespace System;
using namespace System::Collections::Generic;

namespace SOPAIDwrapper {

	// 托管枚举值与原生 SopAidModelFormat 保持一致，可直接进行数值映射。
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

	// 保存最近一次原生调用的状态和错误文本；Succeeded 仅表示 Status == Ok。
	public ref class InferenceErrorInfo
	{
	public:
		InferenceErrorInfo();
		InferenceErrorInfo(InferenceStatus status, String^ message);

		property InferenceStatus Status;
		property String^ Message;
		property bool Succeeded { bool get(); }
	};

	// 单模型初始化参数。ClassNamesCsv 的顺序必须与模型输出类别 id 一致。
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

	// 项目目录初始化参数。包装器会让原生 DLL 从 project.json 解析类别及活动模型。
	public ref class ProjectInferenceConfig
	{
	public:
		ProjectInferenceConfig();

		property String^ ProjectDirectory;
		property ModelFormat PreferredFormat;
		property int InputWidth;
		property int InputHeight;
		property float ConfidenceThreshold;
		property float NmsThreshold;
		property bool UseCuda;
		property int DeviceId;
	};

	// 原生模型元数据的托管副本，不依赖原生定长字符缓冲区的生命周期。
	public ref class ModelInfo
	{
	public:
		property String^ ProjectId;
		property String^ ModelId;
		property String^ ModelVersion;
		property String^ ModelPath;
		property String^ Backend;
		property int InputWidth;
		property int InputHeight;
		property int ClassCount;
	};

	// 检测框坐标位于原始图像像素坐标系，采用左上/右下端点。
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

	// 托管推理门面，拥有一个原生 sopaid::Inference 实例。使用完毕后应调用 Dispose/Release。
	public ref class InferenceEvaluator
	{
	public:
		InferenceEvaluator();
		InferenceEvaluator(InferenceConfig^ config);
		InferenceEvaluator(ProjectInferenceConfig^ config);
		~InferenceEvaluator();
		!InferenceEvaluator();

		bool Init(InferenceConfig^ config);
		bool InitProject(ProjectInferenceConfig^ config);
		ModelInfo^ GetModelInfo();
		// imageData 支持 Gray/BGR/BGRA；无 stride 重载默认图像行连续。
		bool Evaluate(array<Byte>^ imageData, int width, int height, int channels, List<DetectionResult^>^ results);
		// stride 以字节为单位，可用于带行填充的 Bitmap/相机缓冲区。results 在调用开始时清空。
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

	// 手部模型参数。推荐同时指定 PalmModelPath 和 HandPoseModelPath。
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

	// x/y 为原图像素坐标，z 为模型相对深度，Visibility 为关键点可见度。
	public ref class HandLandmarkResult
	{
	public:
		property float X;
		property float Y;
		property float Z;
		property float Visibility;
	};

	// 单手的分类与 21 个关键点结果；HandId 不保证跨帧稳定。
	public ref class HandPoseResult
	{
	public:
		HandPoseResult();

		property int HandId;
		property float Confidence;
		property String^ Handedness;
		property List<HandLandmarkResult^>^ Landmarks;
	};

	// 手部姿态托管门面，内部句柄同时拥有掌心检测和关键点网络。
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
		void* handle_;
		bool initialized_;
		InferenceErrorInfo^ lastError_;

		void SetLastError(InferenceStatus status, String^ message);
		void SetLastError(const SopAidError& error);
	};
}
