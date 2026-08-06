# Python代码目录与调用逻辑

## 1. 目录职责

```text
SOPAID/
├─ SOP_PYD.py                 # EXE/命令行/TCP服务入口
├─ task_dispatcher.py         # JSON命令路由、默认配置和路径解析
├─ tools/                     # 项目配置、数据准备、训练和模型转换
│  ├─ project_management.py   # 项目增删改查与激活
│  ├─ project_frames.py       # 项目视频导入和抽帧
│  ├─ annotation_import.py    # YOLO标注导入与校验
│  ├─ dataset_builder.py      # 项目数据集划分和清单生成
│  ├─ project_training.py     # 项目训练编排和模型版本登记
│  ├─ training.py             # Ultralytics训练及格式导出底层实现
│  ├─ training_control.py     # 训练进度文件和安全停止控制
│  └─ project_model_conversion.py # 三种部署格式转换和清单更新
├─ scripts/                   # 检测运行时与项目领域模型
│  ├─ main_video.py           # 视频检测总流程
│  ├─ inference/              # 统一检测接口及Python后端
│  ├─ project/                # 项目加载、校验和原子存储
│  ├─ tracking.py             # 目标跟踪
│  ├─ events.py               # 对象/ROI事件生成
│  ├─ sop_logic.py            # SOP步骤推进
│  ├─ hand_pose.py            # 手部关键点检测
│  ├─ visualizer.py           # 结果绘制
│  └─ runtime_logging.py      # 任务结构化日志
├─ deploy/                    # TorchScript/ONNX/TensorRT独立导出器
├─ tests/                     # Python单元与接口回归测试
└─ DEEPAIY/                   # 旧版本兼容代码，不进入当前主调用链
```

`scripts/detector.py`和`scripts/legacy_sk_config.py`属于兼容层。当前测试和旧配置仍依赖它们，不能作为普通冗余文件删除。

## 2. TCP调用主链

```text
前端JSON请求
  → SOP_PYD.handle_client
  → SOP_PYD.parse_request_bytes
  → task_dispatcher.run_task
  → 对应command处理函数
  → tools或scripts业务模块
  → 统一JSON响应
```

TCP采用“一次连接、一次请求”。`SOP_PYD.py`只负责协议和进程入口，不能在入口层增加训练、项目或推理业务判断。

## 3. 项目训练调用链

```text
train_project
  → task_dispatcher.run_train_project
  → tools.project_training.train_project_model
  → 校验项目、数据集、模型ID和版本
  → tools.training.train_yolo_model
  → Ultralytics YOLO.train
  → 每轮回调更新 outputs/training/training_status.json
  → 检查 outputs/training/stoptrain.json
  → 复制最终权重到 models/<model_id>/<version>/best.pt
  → register_project_model
  → 写入 model_manifest.json
  → 可选更新 project.json 的 active_model
```

正常完成使用训练产生的`best.pt`；用户要求停止时，在当前轮安全结束后使用该轮`last.pt`作为项目正式`best.pt`。

## 4. 模型转换调用链

```text
convert_project_model
  → task_dispatcher.run_convert_project_model
  → tools.project_model_conversion.convert_project_model
  → tools.training.export_model_formats
  ├─ deploy.export_torchscript.export_torchscript
  ├─ deploy.export_onnx.export_onnx
  └─ deploy.export_tensorrt.export_tensorrt
  → 更新 model_manifest.json/artifacts
```

Engine依赖ONNX。仅请求Engine时，存在`best.onnx`则复用，否则先生成ONNX。

## 5. 视频检测调用链

```text
detect
  → task_dispatcher.run_detect
  → scripts.main_video.process_video
  → scripts.project.load_sop_project
  → scripts.inference.factory.build_detector
  → 逐帧推理
  → tracking目标关联
  → events事件生成
  → sop_logic步骤推进
  → visualizer绘制
  → 输出视频和result.json
```

## 6. 维护边界

- 新的外部JSON命令只在`task_dispatcher.py`登记。
- 项目文件写入统一使用`scripts.project.storage`的原子写方法。
- 训练只生成源checkpoint，部署格式必须走独立转换命令。
- 推理后端必须实现`scripts.inference.base`定义的统一接口。
- 不应从其他模块调用以下划线开头的私有函数。
- `DEEPAIY`、`detector.py`和`legacy_sk_config.py`只能作为兼容层维护，新功能不得继续堆入其中。
