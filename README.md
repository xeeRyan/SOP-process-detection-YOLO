# SOP 工序检测 Demo

## Windows 环境重建

项目固定使用 Python 3.12、PyTorch CUDA 12.6 和 PyInstaller。电脑迁移后，在项目根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

脚本会安装 Python 3.12、创建 `.venv`、断点下载并校验 Torch wheel，然后安装运行与打包依赖。

日常运行可直接使用 `uv run`，也可以激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\check_env.py
python SOP_PYD.py detect health
```

本项目使用 YOLO26s 训练权重检测装配视频中的关键目标，并结合 MediaPipe 手部骨骼、固定 ROI 和 SOP 状态机判断操作是否按顺序完成。

## 项目结构

```text
SOPAID/
├── SOP_PYD.py              # TCP/命令行封装入口，最终 exe 打包入口
├── SOP_PYD.spec            # PyInstaller 打包配置
├── config/app_config.json  # 模型、ROI、阈值、输出目录等默认配置
├── SOPAIDC++/              # 原生 C++ 推理 DLL 与测试程序
├── SOPAID_wrapper/         # 面向 .NET 的 C++/CLI 封装
├── SOPAID_wrapper_test/    # C# 封装调用示例
├── DEEPAIY/                # 外部参考封装样例，不参与本项目运行
├── deploy/                 # ONNX/TensorRT 格式转换脚本
├── docs/                   # 部署说明
├── models/                 # 模型权重与 hand_landmarker.task
├── outputs/                # 检测结果与运行日志
├── scripts/                # 运行时算法核心模块
├── tools/                  # 训练、抽帧、数据集整理、ROI 标定工具
└── videos/                 # 测试视频
```

## C++ 与 .NET 封装

仓库同时提供原生 C++ 推理接口及其 .NET 调用链：

```text
SOPAIDC++/              # SOPAID.dll、SOPAIDExe 及三种模型后端源码
SOPAID_wrapper/         # 将原生 DLL 封装为 .NET 可调用接口
SOPAID_wrapper_test/    # C# 控制台测试项目
```

使用 Visual Studio 打开 `SOPAIDC++/SOPAID.sln` 编译原生 DLL。首次构建前，将
`SOPAIDC++/SopAidInfer.user.props.template` 复制为 `SopAidInfer.user.props`，并按本机环境配置
OpenCV、ONNX Runtime、TensorRT 和 LibTorch 路径。详细接口、运行参数及后端说明见
[`SOPAIDC++/README.md`](SOPAIDC++/README.md)。

`SOPAID_wrapper` 和 `SOPAID_wrapper_test` 分别为 C++/CLI 封装与 C# 调用示例；编译时需保证
平台目标一致（推荐 `x64`），并让测试程序能够找到 `SOPAID.dll`、`SOPAID_wrapper.dll` 及对应运行库。

## 运行时代码

`scripts/` 目录只保留最终推理运行需要的模块：

```text
config.py           # 默认路径、阈值、ROI、SOP 步骤等代码级配置
main_video.py       # 视频检测主流程
detector.py         # YOLO 模型加载与目标检测
hand_pose.py        # MediaPipe 手部骨骼检测
sop_logic.py        # SOP 顺序状态机
visualizer.py       # 检测框、ROI、骨骼和状态绘制
utils.py            # 通用工具函数
runtime_logging.py  # 运行日志
check_env.py        # 环境检查
```

## 训练工具代码

`tools/` 目录放算法开发阶段使用的工具：

```text
extract_frames.py    # 从视频抽帧
prepare_dataset.py   # 整理 YOLO 数据集
select_roi.py        # 交互式框选 ROI
training.py          # YOLO 训练入口
```

示例：

```powershell
python tools\extract_frames.py
python tools\prepare_dataset.py
python tools\select_roi.py
```

## 软件端封装入口

当前项目复用 `DEEPLEARN_PYD.py` 的封装形式，另建本项目入口 `SOP_PYD.py`。

无参数启动时默认监听 TCP 端口 `5000`：

```powershell
python SOP_PYD.py
```

显式启动 TCP 服务：

```powershell
python SOP_PYD.py tcp 5000
```

TCP 客户端发送 JSON 文件路径：

```text
model_file:D:\Python\SOPAID\config\detect_request.json
```

也可以发送普通 JSON 字符串：

```json
{
  "command": "detect",
  "params": {
    "video_path": "videos/sk.mp4",
    "output_dir": "outputs/yolo26s"
  }
}
```

返回为 UTF-8 JSON 字符串：

```json
{
  "status": "ok",
  "message": "",
  "data": {}
}
```

关闭单次连接：

```text
end
```

关闭服务：

```text
close
```

## 命令行调用

健康检查：

```powershell
python SOP_PYD.py detect health
```

检测 JSON 文件：

```powershell
python SOP_PYD.py detect_file config\detect_request.json
```

训练 JSON 文件：

```powershell
python SOP_PYD.py train_file config\train_request.json
```

## 默认输出

```text
outputs/yolo26s/result.mp4
outputs/yolo26s/result.json
outputs/logs/detect_*.log
```

## EXE 打包

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1
```

打包后运行：

```powershell
.\dist\SOP_PYD\SOP_PYD.exe detect health
.\dist\SOP_PYD\SOP_PYD.exe tcp 5000
```

最终交付时，建议将 `config/`、`models/`、`outputs/` 放在 `SOP_PYD.exe` 同级目录。

## 检测类别

YOLO 只负责以下目标类别：

```text
bearing
cover
tool
```

手部信息由 MediaPipe 手部骨骼模块提供。

SOP 顺序：

```text
bearing -> cover -> screw_action -> tool_return
```

## ONNX / TensorRT

导出 ONNX：

```powershell
python deploy/export_onnx.py --overwrite
```

