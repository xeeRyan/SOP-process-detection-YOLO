# SOPAID 多流程视频识别平台

SOPAID用于配置、训练和检测不同工业SOP流程。当前已完成Python主链路和WPF测试前端；
C++推理工程保留在仓库中，待Python协议冻结后再统一整理和接入。

## 当前能力

一个SOP项目独立包含：

- 识别类别定义；
- 原始流程视频；
- 抽帧图片和外部YOLO标注；
- ROI区域；
- SOP步骤、顺序和触发规则；
- 数据集、训练记录和模型版本；
- 检测结果、事件与运行日志。

检测流水线：

```text
视频
  -> Python模型推理
  -> 目标跟踪
  -> 对象/ROI事件
  -> SOP状态机
  -> result.json / result.mp4
```

流程触发器支持：

- 目标出现或进入ROI；
- 手部进入ROI；
- 对象出现、消失、进入、离开事件；
- 多条件 `all/any` 组合；
- 对象数量范围；
- 同一跟踪目标跨ROI移动；
- 条件连续保持指定时间；
- 必选/可选步骤、稳定帧和超时。

高级规则示例见
[`docs/advanced_workflow_triggers.md`](docs/advanced_workflow_triggers.md)。

## 核心目录

```text
SOPAID/
├─ SOP_PYD.py                 # Python TCP/命令行入口
├─ task_dispatcher.py         # 命令分发与参数解析
├─ config/                    # 默认服务与检测配置
├─ projects/                  # 各SOP项目配置和项目资产
├─ scripts/
│  ├─ inference/              # 统一Python推理接口及Ultralytics实现
│  ├─ project/                # 项目加载、校验、路径和原子写入
│  ├─ main_video.py           # 视频检测流水线
│  ├─ tracking.py             # 轻量IoU目标跟踪
│  ├─ events.py               # 对象和ROI事件层
│  ├─ sop_logic.py            # SOP状态机与高级触发器
│  ├─ hand_pose.py            # 手部关键点推理
│  └─ visualizer.py           # 结果视频绘制
├─ tools/                     # 抽帧、标注导入、数据集和训练
├─ frontend/SopAidTcpTester/  # WPF联调前端
├─ tests/                     # Python自动化测试
├─ deploy/                    # ONNX/TensorRT导出脚本
└─ docs/                      # 接口和部署文档
```

以下目录不属于当前Python运行主链：

- `SOPAIDC++/`：当前C++推理工程；
- `SOPAID_wrapper/`、`SOPAID_wrapper_test/`：旧.NET封装原型；
- `build/`、`dist/`、`outputs/`、`runs/`：生成产物。

这些目录暂未自动删除，避免误删依赖、模型或尚待整理的C++代码。

## SOP项目结构

```text
projects/<PROJECT_ID>/
├─ project.json
├─ rois.json
├─ workflow.json
├─ source_videos/
├─ frames/
├─ annotations/
├─ dataset/
├─ models/
├─ runs/
└─ outputs/
```

`project.json`管理类别和活动模型（`active_model_task` 支持 `detect`、`segment`、`pose`）；
`rois.json`保存归一化ROI；`workflow.json`定义步骤和触发规则。步骤可通过
`trigger.evidence.type` 选择 `bbox`、`mask` 或 `keypoints`。

同一项目需要多种视觉能力时，在 `project.json` 的 `model_profiles` 中按任务补充模型路径；
检测模型用于跟踪，分割/姿态模型只为对应步骤提供证据。

## 运行

### 全新设备复现

仓库使用 Git LFS 保存模型和原始视频。新设备必须先安装 Git LFS，随后执行：

```powershell
git clone https://github.com/xeeRyan/SOP-process-detection-YOLO.git
cd SOP-process-detection-YOLO
git lfs install
git lfs pull
uv sync --group build
.\.venv\Scripts\python.exe scripts\verify_checkout.py
```

`verify_checkout.py` 会检查活动项目模型、基础模型、手部模型、测试视频、
项目原始视频和已导入标注，并识别尚未下载的 Git LFS 指针文件。

仓库保留可复现所需的源码、锁文件、项目配置、模型、原始视频和标注；
`frames/`、`dataset/`、`runs/`、`outputs/`、`build/` 和 `dist/` 是可重新生成的产物，
不会重复存入版本库。

环境检查：

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py --project projects\SK_DEMO --video videos\sk.mp4
```

启动TCP服务，默认地址从`config/app_config.json`读取：

```powershell
.\.venv\Scripts\python.exe SOP_PYD.py
```

或指定端口：

```powershell
.\.venv\Scripts\python.exe SOP_PYD.py tcp 9000
```

健康检查：

```powershell
.\.venv\Scripts\python.exe SOP_PYD.py health
```

启动前端：

```powershell
dotnet run --project frontend\SopAidTcpTester\SopAidTcpTester.csproj
```

## TCP命令

当前支持：

```text
health
list_projects
create_project
get_project
update_project
save_rois
save_workflow
activate_project
extract_frames
import_annotations
build_dataset
train_project
detect
train
```

接口说明见[`docs/PYTHON_TCP_API.md`](docs/PYTHON_TCP_API.md)。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
dotnet build frontend\SopAidTcpTester\SopAidTcpTester.csproj -c Release
```

## 当前边界

- Python推理接口已经可用，C++接口尚未并入同一后端工厂；
- 当前跟踪器统一使用ByteTrack两阶段匹配和卡尔曼运动预测，复杂遮挡场景仍需用项目数据集专项验收；
- 高级规则提高了SOP表达能力，但不同SOP的检出率仍必须通过对应数据集验收；
- 拧紧、安装到位、折叠完成等动作不能只依赖目标框和ROI，可能需要姿态、动作或状态分类模型。
