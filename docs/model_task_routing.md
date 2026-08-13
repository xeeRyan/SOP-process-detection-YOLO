# 多任务模型路由

模型路由器根据当前 SOP 步骤的 `trigger.evidence.type` 选择视觉任务：

| evidence.type | 模型任务 |
| --- | --- |
| `bbox` 或未声明 | `detect` |
| `mask` | `segment` |
| `keypoints` | `pose` |

检测任务在可用时作为跟踪基准持续运行，其余任务只在当前步骤需要时调用。步骤切换会清空任务缓存，避免上一工序的视觉结果污染下一工序。

模型后端采用懒加载：项目启动时只登记模型路径，首次进入需要该任务的步骤时才创建后端；任务结束后统一释放已加载后端。

## 项目配置

在 `project.json` 中为任务声明模型。`inference_interval_frames` 可为某个任务单独设置推理间隔，未声明时使用请求级的全局间隔。

```json
{
  "active_model_task": "detect",
  "model_profiles": {
    "detect": {
      "task": "detect",
      "path": "models/detect/best.pt",
      "inference_interval_frames": 2
    },
    "segment": {
      "task": "segment",
      "path": "models/segment/best.pt",
      "inference_interval_frames": 1
    },
    "pose": {
      "task": "pose",
      "path": "models/pose/best.pt",
      "inference_interval_frames": 2
    }
  }
}
```

结果 JSON 的 `pipeline.router` 会记录可用任务、任务间隔、后端信息和实际推理次数，便于检查模型是否被按需调度。

## 多任务结果融合

检测结果作为主轨迹，分割、姿态和属性模型按 bbox IoU 与主轨迹匹配，
将 `mask`、`keypoints`、`attributes` 合并到同一个 `Detection`，因此 SOP
事件和证据判断可以继续使用统一的 `track_id`。

## ROI 事件去抖

`tracking` 配置支持连续帧确认，避免单帧误检直接触发区域事件：

```json
{
  "event_enter_stable_frames": 2,
  "event_exit_stable_frames": 2,
  "event_lost_tolerance_frames": 8
}
```

进入或离开 ROI 必须连续满足配置帧数；短暂遮挡仍由 `event_lost_tolerance_frames` 保持原有轨迹。
## Multi-class group counting

`object_count` also accepts `class_names` to count several configured classes as one group:

```json
{
  "type": "object_count",
  "class_names": ["glove_front", "glove_back"],
  "roi_id": "merge_zone",
  "min_count": 10,
  "max_count": 10
}
```

Tracked objects are deduplicated by `track_id`, which is useful for multi-class materials,
batch merging, and packaging completeness checks.
