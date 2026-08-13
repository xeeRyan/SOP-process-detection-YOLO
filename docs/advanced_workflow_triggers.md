# 高级 SOP 触发器

高级触发器填写在 `workflow.json` 的步骤 `trigger` 中。前端流程步骤表选择对应
触发方式后，将完整触发器对象填写到“高级触发 JSON”列。

## 组合条件

`operator` 支持 `all` 和 `any`。

```json
{
  "type": "composite",
  "operator": "all",
  "stable_frames": 3,
  "conditions": [
    {
      "type": "object_in_roi",
      "class_name": "screw",
      "roi_id": "work",
      "confidence": 0.4
    },
    {
      "type": "object_in_roi",
      "class_name": "screwdriver",
      "roi_id": "work",
      "confidence": 0.4
    }
  ]
}
```

## 对象计数

`roi_id`可省略，省略时统计整帧。使用跟踪ID去重。

```json
{
  "type": "object_count",
  "class_name": "screw",
  "roi_id": "work",
  "min_count": 4,
  "max_count": 4,
  "confidence": 0.4,
  "stable_frames": 3
}
```

## 跨 ROI 移动

只有同一个 `track_id` 从来源区域移动到目标区域时才触发。

```json
{
  "type": "object_transition",
  "class_name": "part",
  "from_roi_id": "material_bin",
  "to_roi_id": "work"
}
```

## 持续时间

内层条件必须连续成立；中断后重新计时。

```json
{
  "type": "duration",
  "duration_sec": 2.0,
  "condition": {
    "type": "object_in_roi",
    "class_name": "tool",
    "roi_id": "work",
    "confidence": 0.4
  }
}
```

四类触发器可以嵌套，例如在组合条件中使用对象计数或持续时间。为避免单帧误检，
组合和计数触发器仍建议配置 `stable_frames`。

## 视觉证据

`object_in_roi` 可通过 `evidence` 选择更适合动作的视觉证据：

```json
{
  "type": "object_in_roi",
  "class_name": "part",
  "roi_id": "work",
  "evidence": {
    "type": "mask",
    "min_roi_overlap": 0.35,
    "stable_frames": 3
  }
}
```

`bbox` 使用检测框中心点，`mask` 使用实例分割区域覆盖率，`keypoints` 使用姿态
关键点进入 ROI。项目 `project.json` 的 `active_model_task` 必须分别配置为
`detect`、`segment` 或 `pose`；混合任务时可在 `model_profiles` 中同时配置多个任务模型，
否则项目加载阶段会拒绝不匹配的工作流。

运行时会按当前步骤调度模型：检测模型持续用于跟踪，分割和姿态模型只在当前步骤
需要对应证据时运行；结果中的 `pipeline.inference_counts` 可用于检查推理开销。

## 通用属性序列与批次

工作流可以在根节点声明多轴属性交替约束。约束只消费检测结果中的通用
`attributes` 字段，不绑定具体物料名称；第一项可用 `initial: "auto"` 自动确定，
后续按各轴的 `values` 循环交替。批次开始步骤加入
`constraints.batch_policy.reset_sequence_on_step_ids` 后，会重新初始化所有序列。
