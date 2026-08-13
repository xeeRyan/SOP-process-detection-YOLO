# SOPAID C++/CLI 包装器

本模块把 `SOPAIDC++/SOPAID.dll` 暴露给 .NET Framework 4.7.2 调用方。它是托管适配器，项目解析、模型选择和推理仍由原生 `sopaid::Inference` 负责。

## 多 SOP 项目初始化

```csharp
var config = new ProjectInferenceConfig
{
    ProjectDirectory = @"projects\SK_DEMO",
    PreferredFormat = ModelFormat.Auto,
    UseCuda = false
};

using (var evaluator = new InferenceEvaluator(config))
{
    ModelInfo info = evaluator.GetModelInfo();
    // evaluator.Evaluate(...)
}
```

原有 `InferenceConfig` 和按模型路径初始化接口继续保留。

## 图像约定

`Evaluate` 接受 1、3、4 通道 `byte[]` 和可选 stride。包装器会把灰度和 BGRA 转换成三通道 BGR，再调用原生推理接口。

## 运行依赖

Release 构建会收集 `SOPAID.dll`、OpenCV、ONNX Runtime 和 shared provider。TensorRT、LibTorch、CUDA/cuDNN 依赖仅在启用对应原生后端时另行交付。

本项目统一通过 C++/CLI 包装器连接 C# 与原生 C++。WPF 与包装器必须使用兼容的 .NET 目标框架；当前包装器目标为 .NET Framework 4.7.2，接入现有 `net10.0-windows` WPF 前需要先迁移包装器的目标框架和构建工具链。

## 源码构建依赖

`SOPAID_wrapper.dependencies.props` 定义 wrapper 使用的原生工程、OpenCV 和
ONNX Runtime 默认路径。完整源码目录应保持：

```text
SOPAID/
├─ SOPAIDC++/
└─ SOPAID_wrapper/
```

如果依赖安装在其他位置，可修改该 props 文件，或在
`SOPAIDC++/SopAidInfer.user.props` 中覆盖同名属性。不要只复制
`SOPAID_wrapper/` 源码目录后直接编译，因为它需要原生头文件、导入库和运行库。
