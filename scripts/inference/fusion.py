"""将辅助视觉任务结果融合到同一条跟踪轨迹。"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping

from scripts.inference.types import Detection


class DetectionFusion:
    """把分割、姿态和属性结果挂接到主跟踪检测上。

    主检测结果提供稳定的 ``track_id``；辅助任务只补充 ``mask``、
    ``keypoints`` 和 ``attributes``，避免同一目标在结果列表中重复出现。
    """

    def __init__(self, *, iou_threshold: float = 0.3) -> None:
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold 必须在 0 到 1 之间")
        self.iou_threshold = float(iou_threshold)

    def merge(
        self,
        primary: Iterable[Detection],
        auxiliary_by_task: Mapping[str, Iterable[Detection]],
    ) -> list[Detection]:
        merged = list(primary)
        for task in sorted(auxiliary_by_task):
            for auxiliary in auxiliary_by_task[task]:
                match_index = self._best_match(auxiliary, merged)
                if match_index is None:
                    merged.append(auxiliary)
                    continue
                merged[match_index] = self._attach(merged[match_index], auxiliary, task)
        return merged

    def _best_match(
        self,
        auxiliary: Detection,
        candidates: list[Detection],
    ) -> int | None:
        matches = [
            (index, _bbox_iou(auxiliary.bbox, candidate.bbox))
            for index, candidate in enumerate(candidates)
            if candidate.class_name == auxiliary.class_name
        ]
        if not matches:
            return None
        index, score = max(matches, key=lambda item: item[1])
        return index if score >= self.iou_threshold else None

    @staticmethod
    def _attach(primary: Detection, auxiliary: Detection, task: str) -> Detection:
        attributes = primary.attributes
        if auxiliary.attributes is not None:
            attributes = {
                **(primary.attributes or {}),
                **auxiliary.attributes,
            }
        return replace(
            primary,
            mask=auxiliary.mask if auxiliary.mask is not None else primary.mask,
            keypoints=(
                auxiliary.keypoints
                if auxiliary.keypoints is not None
                else primary.keypoints
            ),
            attributes=attributes,
        )


def _bbox_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0

