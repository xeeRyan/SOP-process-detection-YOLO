"""按 SOP 步骤调度视觉任务的通用模型路由器。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping

import numpy as np

from scripts.inference.base import DetectorBackend
from scripts.inference.types import Detection


class ModelRouter:
    """选择、缓存并统计 detect/segment/pose 等视觉模型的调用。

    路由器只依赖工作流的 ``trigger`` 和后端统一接口，不包含任何具体物料
    或 SOP 业务规则。检测任务默认持续提供跟踪基准，其余任务按当前步骤的
    ``evidence.type`` 按需调用。
    """

    def __init__(
        self,
        detectors: Mapping[str, DetectorBackend] | None = None,
        *,
        detector_factories: Mapping[str, Callable[[], DetectorBackend]] | None = None,
        default_interval: int = 1,
        task_intervals: Mapping[str, int] | None = None,
    ) -> None:
        if default_interval <= 0:
            raise ValueError("default_interval 必须是正整数")
        self.detectors = dict(detectors or {})
        self._detector_factories = dict(detector_factories or {})
        duplicate_tasks = set(self.detectors) & set(self._detector_factories)
        if duplicate_tasks:
            raise ValueError(
                "detectors 和 detector_factories 不能重复声明任务: "
                + ", ".join(sorted(duplicate_tasks))
            )
        self._configured_tasks = set(self.detectors) | set(self._detector_factories)
        self.default_interval = int(default_interval)
        self.task_intervals = {
            str(task): int(interval)
            for task, interval in (task_intervals or {}).items()
        }
        if any(interval <= 0 for interval in self.task_intervals.values()):
            raise ValueError("task_intervals 中的间隔必须是正整数")
        self._cache: dict[str, list[Detection]] = {}
        self.inference_counts: dict[str, int] = {
            task: 0 for task in self._configured_tasks
        }

    @property
    def available_tasks(self) -> set[str]:
        return set(self._configured_tasks)

    @property
    def loaded_tasks(self) -> set[str]:
        return set(self.detectors)

    @staticmethod
    def tasks_for_trigger(trigger: dict[str, Any] | None) -> set[str]:
        """根据证据声明递归推导当前步骤所需模型任务。"""

        if not isinstance(trigger, dict):
            return {"detect"}
        evidence = trigger.get("evidence")
        evidence_type = (
            str(evidence.get("type", "bbox")).strip().lower()
            if isinstance(evidence, dict)
            else "bbox"
        )
        trigger_type = str(trigger.get("type", "")).strip().lower()
        if evidence_type == "mask":
            tasks = {"segment"}
        elif evidence_type == "keypoints":
            tasks = {"pose"}
        elif trigger_type in {"composite", "duration", "hand_in_roi"}:
            tasks = set()
        else:
            tasks = {"detect"}
        for condition in trigger.get("conditions", []):
            if isinstance(condition, dict):
                tasks.update(ModelRouter.tasks_for_trigger(condition))
        nested = trigger.get("condition")
        if isinstance(nested, dict):
            tasks.update(ModelRouter.tasks_for_trigger(nested))
        return tasks

    def infer(
        self,
        frame: np.ndarray,
        trigger: dict[str, Any] | None,
        frame_id: int,
        *,
        force: bool = False,
    ) -> tuple[dict[str, list[Detection]], set[str]]:
        """执行当前步骤所需任务，返回所有结果和本帧新推理任务。"""

        if frame_id < 0:
            raise ValueError("frame_id 不能小于 0")
        tasks = self.tasks_for_trigger(trigger)
        if "detect" in self.detectors:
            tasks.add("detect")
        tasks.intersection_update(self._configured_tasks)

        raw_by_task: dict[str, list[Detection]] = {}
        fresh_tasks: set[str] = set()
        for task in sorted(tasks):
            interval = self.task_intervals.get(task, self.default_interval)
            should_run = force or task not in self._cache or frame_id % interval == 0
            if should_run:
                self._cache[task] = self._get_detector(task).detect(frame)
                self.inference_counts[task] = self.inference_counts.get(task, 0) + 1
                fresh_tasks.add(task)
            raw_by_task[task] = self._cache[task]
        return raw_by_task, fresh_tasks

    def clear_cache(self) -> None:
        self._cache.clear()

    def metadata(self) -> dict[str, Any]:
        return {
            "available_tasks": sorted(self._configured_tasks),
            "loaded_tasks": sorted(self.detectors),
            "default_interval": self.default_interval,
            "task_intervals": dict(self.task_intervals),
            "inference_counts": dict(self.inference_counts),
            "backends": {
                task: detector.metadata()
                for task, detector in sorted(self.detectors.items())
            },
        }

    def _get_detector(self, task: str) -> DetectorBackend:
        detector = self.detectors.get(task)
        if detector is not None:
            return detector
        factory = self._detector_factories.get(task)
        if factory is None:
            raise KeyError(f"未配置视觉任务后端: {task}")
        detector = factory()
        self.detectors[task] = detector
        return detector

    def close(self) -> None:
        for detector in self.detectors.values():
            detector.close()
