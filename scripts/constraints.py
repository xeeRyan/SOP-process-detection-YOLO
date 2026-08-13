"""与业务无关的时序属性约束和批次重置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scripts.inference import Detection


@dataclass(frozen=True)
class SequenceAxis:
    """一个可交替属性轴，例如方向、表面或姿态。"""

    attribute: str
    values: tuple[str, ...]
    mode: str = "alternate"
    initial: str = "auto"

    def __post_init__(self) -> None:
        if not self.attribute.strip():
            raise ValueError("sequence axis attribute不能为空")
        if len(self.values) < 2 or len(set(self.values)) != len(self.values):
            raise ValueError("sequence axis values至少需要两个不重复值")
        if self.mode != "alternate":
            raise ValueError(f"不支持的sequence axis mode: {self.mode}")
        if self.initial != "auto" and self.initial not in self.values:
            raise ValueError("sequence axis initial必须是auto或values中的值")


class AttributeSequenceConstraint:
    """按事件消费检测属性，并校验多轴交替序列。

    约束只关心通用属性名，不包含任何具体业务类别；每个批次可通过
    ``reset`` 重新随机初始化第一项。
    """

    def __init__(
        self,
        constraint_id: str,
        axes: list[SequenceAxis],
        *,
        class_name: str | None = None,
        roi_id: str | None = None,
        event_type: str = "object_enter_roi",
    ) -> None:
        self.constraint_id = constraint_id
        self.axes = tuple(axes)
        self.class_name = class_name
        self.roi_id = roi_id
        self.event_type = event_type
        self.reset()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AttributeSequenceConstraint":
        constraint_id = str(config.get("id", "")).strip()
        if not constraint_id:
            raise ValueError("sequence constraint id不能为空")
        raw_axes = config.get("axes")
        if not isinstance(raw_axes, list) or not raw_axes:
            raise ValueError(f"sequence constraint {constraint_id}必须定义axes")
        axes: list[SequenceAxis] = []
        for index, raw_axis in enumerate(raw_axes):
            if not isinstance(raw_axis, dict):
                raise ValueError(f"sequence constraint {constraint_id}.axes[{index}]必须是对象")
            raw_values = raw_axis.get("values")
            if not isinstance(raw_values, list) or not all(isinstance(value, str) for value in raw_values):
                raise ValueError(f"sequence constraint {constraint_id}.axes[{index}].values必须是字符串数组")
            axes.append(
                SequenceAxis(
                    attribute=str(raw_axis.get("attribute", "")),
                    values=tuple(raw_values),
                    mode=str(raw_axis.get("mode", "alternate")),
                    initial=str(raw_axis.get("initial", "auto")),
                )
            )
        return cls(
            constraint_id,
            axes,
            class_name=str(config["class_name"]) if config.get("class_name") else None,
            roi_id=str(config["roi_id"]) if config.get("roi_id") else None,
            event_type=str(config.get("event", "object_enter_roi")),
        )

    def reset(self) -> None:
        self._last_values: dict[str, str] = {}
        self._count = 0

    def matches(self, event_type: str, class_name: str, roi_id: str | None) -> bool:
        return (
            event_type == self.event_type
            and (self.class_name is None or class_name == self.class_name)
            and (self.roi_id is None or roi_id == self.roi_id)
        )

    def consume(self, detection: Detection) -> tuple[bool, str]:
        attributes = detection.attributes or {}
        candidate: dict[str, str] = {}
        for axis in self.axes:
            value = attributes.get(axis.attribute)
            if value is None:
                return False, f"缺少属性 {axis.attribute}"
            candidate[axis.attribute] = str(value)
            if candidate[axis.attribute] not in axis.values:
                return False, f"属性 {axis.attribute}={candidate[axis.attribute]} 不在允许值内"

        if self._count == 0:
            for axis in self.axes:
                if axis.initial != "auto" and candidate[axis.attribute] != axis.initial:
                    return False, f"属性 {axis.attribute}期望 {axis.initial}，实际 {candidate[axis.attribute]}"
            for axis in self.axes:
                self._last_values[axis.attribute] = candidate[axis.attribute]
            self._count = 1
            return True, ""

        for axis in self.axes:
            previous = self._last_values[axis.attribute]
            expected = axis.values[(axis.values.index(previous) + 1) % len(axis.values)]
            actual = candidate[axis.attribute]
            if actual != expected:
                return False, f"属性 {axis.attribute}期望 {expected}，实际 {actual}"

        self._last_values = candidate
        self._count += 1
        return True, ""

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.constraint_id,
            "count": self._count,
            "last_values": dict(self._last_values),
            "axes": [
                {
                    "attribute": axis.attribute,
                    "values": list(axis.values),
                    "mode": axis.mode,
                    "initial": axis.initial,
                }
                for axis in self.axes
            ],
        }
