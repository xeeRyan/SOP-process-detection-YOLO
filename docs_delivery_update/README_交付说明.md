# SOPAID 前端交付包

交付日期：2026-07-30  
文档核对日期：2026-07-31  
环境：Windows 10/11 x64；Python Demo 为 `net10.0-windows`；C++/CLR 为 .NET Framework 4.7.2 x64。

## 开始对接

首先阅读：`接口说明\前端对接总说明.md`。

## 目录

- `C++推理SDK\frontend_bin`：完整 CLR/C++ 三后端运行目录。
- `C++推理SDK\runtime`：分类运行库副本。
- `Python运行程序\SOP_PYD`：已封装 Python 后端。
- `前端测试程序\Python_WPF_Demo`：Python TCP 联调工具。
- `models\hand_pose`：手部双 ONNX 模型。
- `models\base_yolo`：Python 训练基础 `.pt`，不能直接给 C++ LibTorch。
- `model_projects\keyboard`：5 类项目，ONNX + TorchScript。
- `model_projects\SK_DEMO`：3 类项目，ONNX + TorchScript + engine。
- `接口说明\python对接`：Python TCP 全套文档。
- `接口说明\C++CLR对接`：C# 单帧/摄像头、部署和参数文档。

## 接口边界

- Python TCP：传视频文件路径，负责项目、训练和完整 SOP，同步返回最终结果。
- C++/CLR：传单帧 `byte[]`，适合摄像头/图片，不负责 SOP 状态机。
- 当前 CLR 不提供视频路径方法；处理视频须由前端解码后逐帧调用。

## 快速验证

- Python：运行 `运行Python健康检查.cmd` 和 `启动Python_WPF测试Demo.cmd`。
- CLR：在 `C++推理SDK\frontend_bin` 运行 `运行CLR冒烟测试.cmd`。
- 正式 C# 引用 `SOPAID_wrapper.dll`，固定 x64，并复制 `frontend_bin` 全部文件到 EXE 目录。

## 限制

- Python 串行执行，无进度、任务 ID、查询、取消或逐帧推送。
- 前端取消等待不会停止 Python 任务。
- Python `detect` 不支持摄像头编号。
- engine 不保证跨 GPU/驱动/TensorRT 版本通用。
- `.pt` 推理实际要求 TorchScript；普通训练检查点不能直接由 C++ 加载。
- 部署不要只复制某个后端的少量 DLL。
