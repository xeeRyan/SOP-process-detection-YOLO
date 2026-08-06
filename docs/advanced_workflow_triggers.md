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
