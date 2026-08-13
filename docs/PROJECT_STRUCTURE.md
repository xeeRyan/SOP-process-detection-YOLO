# 项目结构说明

## 核心运行链

```text
SOP_PYD.py
  -> task_dispatcher.py
  -> scripts/main_video.py
  -> ModelRouter / DetectorBackend
  -> ByteTrackTracker
  -> SopEventEngine
  -> SOPStateMachine
  -> result.json / result.mp4
```

## 目录职责

| 目录 | 职责 | 是否纳入运行链 |
| --- | --- | --- |
| `scripts/` | Python 推理、跟踪、事件、SOP 状态机 | 是 |
| `scripts/inference/` | 检测后端、模型路由、多任务结果融合 | 是 |
| `scripts/project/` | 项目加载、路径安全、配置校验 | 是 |
| `tools/` | 抽帧、标注导入、数据集、训练 | 是 |
| `projects/` | 各 SOP 项目的 `project.json`、`rois.json`、`workflow.json` | 是 |
| `frontend/` | WPF 项目管理、ROI、训练和检测联调界面 | 是 |
| `tests/` | Python 自动化测试 | 否 |
| `docs/` | 接口、部署和算法说明 | 否 |
| `docs_delivery_update/` | 历史交付补充说明，新增接口以 `docs/` 和根 README 为准 | 否 |
| `deploy/` | ONNX、TorchScript、TensorRT 导出脚本 | 按需 |
| `SOPAIDC++/` | C++ 推理实现和工程 | 按需 |
| `SOPAID_wrapper/` | C++/CLR 封装工程 | 按需 |
| `datasets/`、`videos/`、`models/`、`outputs/`、`runs/` | 本地数据、模型和运行产物 | 资产目录 |

## 一个 SOP 项目的最小结构

```text
projects/<project_id>/
├─ project.json
├─ rois.json
└─ workflow.json
```

运行过程中生成的 `frames/`、`annotations/`、`dataset/`、`models/`、`runs/` 和 `outputs/` 均属于项目产物，不应混入核心源码目录。

## 维护规则

1. 业务类别、ROI 和步骤只写入项目配置，不写入核心算法。
2. 新模型通过 `model_profiles` 和统一后端接口接入。
3. 临时文件、缓存和构建产物不提交版本库。
4. 删除历史实现前先确认没有被脚本、工程文件或部署文档引用。
5. 每次算法改动至少运行对应单元测试和一次项目级视频回归。
