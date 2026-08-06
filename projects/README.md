# SOP 项目目录

每一种 SOP 使用一个独立目录。运行时通过 `sop_project_dir` 选择项目；未指定时加载
`SK_DEMO`。

## 必需文件

```text
<project_id>/
├─ project.json
├─ rois.json
└─ workflow.json
```

- `project.json`：项目名称、识别类别及当前配置文件。
- `rois.json`：操作区域，推荐使用 `normalized` 的 0～1 坐标。
- `workflow.json`：有序步骤和触发条件。

第一版触发器支持 `object_in_roi`、`object_present` 和 `hand_in_roi`。

## 导入视频与抽帧

```json
{
  "command": "extract_frames",
  "params": {
    "sop_project_dir": "projects/SK_DEMO",
    "video_paths": ["videos/sk.mp4"],
    "frames_per_second": 2.0,
    "max_frames_per_video": 300,
    "copy_videos": true,
    "jpeg_quality": 95
  }
}
```

也可以用 `frame_interval` 指定每隔多少原始帧保存一张，设置后优先于
`frames_per_second`。视频保存到 `source_videos/`，图片保存到
`frames/<video_id>/`，清单写入 `frames_manifest.json`。每张图片初始状态为
`unlabeled`，供下一步标注模块使用。

## 导入外部 YOLO 标注

外部标注工具导出的 `.txt` 文件必须与抽帧图片同名，每行格式为
`class_id center_x center_y width height`，坐标使用 0～1 归一化值。

```json
{
  "command": "import_annotations",
  "params": {
    "sop_project_dir": "projects/SK_DEMO",
    "labels_dir": "D:\\external_annotations\\labels",
    "overwrite": true,
    "mark_missing_as_empty": false
  }
}
```

有效标签写入 `annotations/<video_id>/`，导入结果写入
`annotation_import_report.json`。报告会区分有效、空标注、缺失、格式错误、无匹配图片、
同名冲突和因禁止覆盖而跳过的标签。只有外部工具明确为负样本时，才应启用
`mark_missing_as_empty`。

## 构建训练、验证和测试集

数据集按 `video_id` 分组划分，同一流程视频的相邻帧不会进入不同集合。

```json
{
  "command": "build_dataset",
  "params": {
    "sop_project_dir": "projects/SK_DEMO",
    "dataset_name": "dataset_v1",
    "train_ratio": 0.7,
    "val_ratio": 0.2,
    "test_ratio": 0.1,
    "seed": 42,
    "require_all_annotated": true,
    "require_all_splits": true
  }
}
```

结果写入 `dataset/<dataset_name>/`，包含标准 YOLO `images/`、`labels/`、
`data.yaml` 和 `dataset_manifest.json`。默认要求所有帧都已标注，并要求每个非零比例的
集合至少拥有一个独立视频。视频不足时应补充素材，不能退化成随机拆分相邻帧。

## 项目级模型训练与登记

```json
{
  "command": "train_project",
  "params": {
    "sop_project_dir": "projects/SK_DEMO",
    "dataset_name": "dataset_v1",
    "model_id": "sk_detector",
    "model_version": "1.0.0",
    "base_model_path": "models/yolo26n.pt",
    "epochs": 100,
    "batch": 8,
    "imgsz": 640,
    "workers": 0,
    "device": null,
    "export_torchscript": true,
    "export_onnx": true,
    "set_active": true
  }
}
```

训练运行目录为 `runs/<model_id>_<version>/`，正式模型写入
`models/<model_id>/<version>/`。`model_manifest.json` 记录数据集版本、基础模型、训练参数、
验证指标和导出文件。启用 `set_active` 后，`project.json` 的 `active_model` 会更新，后续检测
请求不传 `model_path` 时自动使用该版本。

## 项目创建和配置

前端使用以下命令完成 SOP 配置生命周期：

```text
list_projects
create_project
get_project
update_project
save_rois
save_workflow
activate_project
```

新项目首先处于 `draft`。`save_rois` 接收完整 `rois` 对象，`save_workflow` 接收完整
`workflow` 对象。流程中的类别和 ROI 引用会交叉校验。只有完整配置通过校验后，
`activate_project` 才能将状态改成 `active`。

类别应在抽帧和标注前定义。项目一旦存在 `frames_manifest.json` 或已生成数据集，接口会
锁定类别定义，防止类别 ID 改动导致外部 YOLO 标签含义错乱。
