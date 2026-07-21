from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from scripts.config import (
    SCREW_BIN_ROI,
    STEP_DEFINITIONS,
    STEP_STABLE_FRAMES,
    TOOL_HOLD_FRAMES,
    TOOL_HOME_ROI,
    WORK_ROI,
)
from scripts.detector import Detection
from scripts.utils import labels_in_roi, point_in_roi


# result.json 中 steps 数组的单步结果结构。
@dataclass
class StepResult:
    """SOP 步骤结果对象。

    该对象会出现在 result.json 的 steps 字段中，供软件端展示每一步状态。
    """

    step_id: int
    name: str
    key: str
    trigger_classes: list[str]
    roi_name: str
    allow_hand_pose: bool = False
    status: str = "pending"
    frame: int | None = None
    time: float | None = None
    trigger_source: str | None = None

    def to_dict(self) -> dict:
        """转换为 result.json 中的步骤字典。"""

        return {
            "step_id": self.step_id,
            "name": self.name,
            "key": self.key,
            "trigger_classes": self.trigger_classes,
            "roi": self.roi_name,
            "allow_hand_pose": self.allow_hand_pose,
            "status": self.status,
            "frame": self.frame,
            "time": self.time,
            "trigger_source": self.trigger_source,
        }


# SOP 顺序判定核心：按配置的步骤顺序逐帧推进状态。
class SOPStateMachine:
    """SOP 顺序判定模块。

    对接用途：
    - 输入每帧 YOLO 检测结果和可选手部骨骼结果。
    - 输出每个 SOP 步骤的状态、完成帧号、完成时间、触发来源和最终 OK/NG。
    - 第 3 步 screw_action 支持 tool 检测框或手部骨骼进入螺丝盘 ROI。
    """

    def __init__(
        self,
        work_roi: Sequence[int] = WORK_ROI,
        screw_bin_roi: Sequence[int] = SCREW_BIN_ROI,
        tool_home_roi: Sequence[int] = TOOL_HOME_ROI,
        step_stable_frames: int = STEP_STABLE_FRAMES,
        tool_hold_frames: int = TOOL_HOLD_FRAMES,
        step_enabled: dict[str, bool] | None = None,
        trigger_sources: dict[str, str] | None = None,
        step_timeouts_sec: dict[str, float] | None = None,
    ) -> None:
        self.rois = {
            "work": work_roi,
            "screw_bin": screw_bin_roi,
            "tool_home": tool_home_roi,
        }
        enabled_steps = step_enabled or {}
        self.steps = [
            StepResult(**step)
            for step in STEP_DEFINITIONS
            if enabled_steps.get(step["key"], True)
        ]
        self.current_index = 0
        self.final_result = "OK" if not self.steps else "NG"
        self.reason = ""
        self._stable_counter = 0
        self._stable_trigger_source: str | None = None
        self.step_stable_frames = step_stable_frames
        self.tool_hold_frames = tool_hold_frames
        self.trigger_sources = trigger_sources or {}
        self.step_timeouts_sec = step_timeouts_sec or {}
        self._step_started_at: float | None = None

    @property
    def current_step(self) -> StepResult | None:
        if self.current_index >= len(self.steps):
            return None
        return self.steps[self.current_index]

    @property
    def state_text(self) -> str:
        if self.final_result == "OK":
            return "FINISHED"
        if self.reason:
            return "NG"
        step = self.current_step
        return f"WAIT_{step.key.upper()}" if step else "FINISHED"

    def update(
        self,
        detections: list[Detection],
        frame_id: int,
        time_sec: float,
        hands: list[Any] | None = None,
    ) -> None:
        """更新状态机。

        参数：
        - detections: 当前帧目标检测结果。
        - frame_id: 当前帧序号。
        - time_sec: 当前帧时间，单位秒。
        - hands: 当前帧手部骨骼结果，来自 hand_pose.HandPoseDetector。
        """

        if self.final_result == "OK" or self.reason:
            return

        step = self.current_step
        if step is None:
            self.final_result = "OK"
            return

        if self._step_started_at is None:
            self._step_started_at = time_sec
        timeout_sec = self.step_timeouts_sec.get(step.key)
        if timeout_sec is not None and timeout_sec <= 0:
            raise ValueError(f"步骤 {step.key} 的超时时间必须大于 0")
        if timeout_sec is not None and time_sec - self._step_started_at > timeout_sec:
            self.reason = f"步骤超时: {step.key} 超过 {timeout_sec} 秒未完成"
            self._mark_remaining_failed()
            return

        roi = self.rois[step.roi_name]
        in_roi = labels_in_roi(detections, roi)
        expected = set(step.trigger_classes)
        trigger_source_mode = self.trigger_sources.get(step.key, "auto")
        if trigger_source_mode not in {"auto", "yolo", "hand_pose"}:
            raise ValueError(f"步骤 {step.key} 的 trigger_source 只支持 auto、yolo、hand_pose")
        yolo_sources = sorted(expected & in_roi) if trigger_source_mode in {"auto", "yolo"} else []
        yolo_hit = bool(yolo_sources)
        hand_hit = (
            step.allow_hand_pose
            and trigger_source_mode in {"auto", "hand_pose"}
            and self._hand_pose_in_roi(hands or [], roi)
        )
        trigger_source = self._resolve_trigger_source(yolo_sources, hand_hit)

        # 顺序错误输出到 reason 字段，供软件端直接展示或记录。
        later_work_classes = {
            trigger_class
            for item in self.steps[self.current_index + 1 :]
            if item.roi_name == "work"
            for trigger_class in item.trigger_classes
        }
        wrong_classes = sorted(labels_in_roi(detections, self.rois["work"]) & later_work_classes)
        if step.roi_name == "work" and wrong_classes and not yolo_hit:
            self.reason = f"顺序错误: 当前等待 {step.key}, 但检测到 {wrong_classes[0]} 先进入装配区"
            self._mark_remaining_failed()
            return

        if trigger_source is None:
            self._stable_counter = 0
            self._stable_trigger_source = None
            return

        # 连续稳定帧计数，避免单帧误检导致步骤误触发。
        self._stable_counter += 1
        self._stable_trigger_source = trigger_source
        required_frames = self.tool_hold_frames if step.key == "tool_return" else self.step_stable_frames
        if self._stable_counter < required_frames:
            return

        step.status = "done"
        step.frame = frame_id
        step.time = round(time_sec, 3)
        step.trigger_source = self._stable_trigger_source
        self.current_index += 1
        self._stable_counter = 0
        self._stable_trigger_source = None
        self._step_started_at = time_sec

        if self.current_index >= len(self.steps):
            self.final_result = "OK"

    def finalize(self, video_name: str) -> dict:
        """生成最终 SOP 结果字典。"""

        if self.final_result != "OK" and not self.reason:
            missing = [step.name for step in self.steps if step.status != "done"]
            self.reason = "步骤缺失: " + "、".join(missing)
            self._mark_remaining_failed()

        return {
            "video_name": video_name,
            "final_result": self.final_result,
            "steps": [step.to_dict() for step in self.steps],
            "reason": self.reason,
        }

    def _mark_remaining_failed(self) -> None:
        for step in self.steps:
            if step.status != "done":
                step.status = "failed"

    def _hand_pose_in_roi(self, hands: list[Any], roi: Sequence[int]) -> bool:
        """判断任意手部关键点是否进入指定 ROI。"""

        return any(point_in_roi((x, y), roi) for hand in hands for x, y, _ in hand.points)

    def _resolve_trigger_source(self, yolo_sources: list[str], hand_hit: bool) -> str | None:
        """返回当前步骤触发来源，供 result.json 记录。"""

        yolo_source = "+".join(yolo_sources)
        if yolo_source and hand_hit:
            return f"{yolo_source}+hand_pose"
        if yolo_source:
            return yolo_source
        if hand_hit:
            return "hand_pose"
        return None
