"""按工作流触发条件推进SOP步骤状态。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from scripts.config import STEP_STABLE_FRAMES, TOOL_HOLD_FRAMES
from scripts.constraints import AttributeSequenceConstraint
from scripts.inference import Detection
from scripts.utils import bbox_center_in_roi, labels_in_roi, mask_roi_overlap, point_in_roi


# result.json 中 steps 数组的单步结果结构。
@dataclass
class StepResult:
    """SOP 步骤结果对象。

    该对象会出现在 result.json 的 steps 字段中，供软件端展示每一步状态。
    """

    step_id: int | str
    name: str
    key: str
    trigger_classes: list[str]
    roi_name: str
    allow_hand_pose: bool = False
    trigger_type: str = "object_in_roi"
    minimum_confidence: float = 0.0
    stable_frames: int | None = None
    timeout_sec: float | None = None
    required: bool = True
    event_type: str | None = None
    trigger_config: dict[str, Any] | None = None
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
            "trigger_type": self.trigger_type,
            "minimum_confidence": self.minimum_confidence,
            "required": self.required,
            "event_type": self.event_type,
            "trigger_config": self.trigger_config,
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
        *,
        workflow: dict[str, Any],
        rois: dict[str, Sequence[int]],
        step_stable_frames: int = STEP_STABLE_FRAMES,
        tool_hold_frames: int = TOOL_HOLD_FRAMES,
        step_enabled: dict[str, bool] | None = None,
        trigger_sources: dict[str, str] | None = None,
        step_timeouts_sec: dict[str, float] | None = None,
    ) -> None:
        if not workflow:
            raise ValueError("SOP状态机必须提供非空 workflow")
        if not rois:
            raise ValueError("SOP状态机必须提供非空 rois")
        self.rois = dict(rois)
        enabled_steps = step_enabled or {}
        step_definitions = self._build_step_definitions(workflow)
        self.steps = [
            StepResult(**step)
            for step in step_definitions
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
        self._duration_started_at: dict[str, float] = {}
        self._batch_counter_states: dict[str, dict[str, Any]] = {}
        workflow_constraints = workflow.get("constraints", {})
        self._attribute_sequences = [
            AttributeSequenceConstraint.from_config(item)
            for item in workflow_constraints.get("attribute_sequences", [])
        ] if isinstance(workflow_constraints, dict) else []
        batch_policy = workflow_constraints.get("batch_policy", {}) if isinstance(workflow_constraints, dict) else {}
        self._reset_sequence_on_step_ids = set(batch_policy.get("reset_sequence_on_step_ids", [])) if isinstance(batch_policy, dict) else set()
        self._last_step_key: str | None = None

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

    @property
    def counter_summaries(self) -> dict[str, dict[str, Any]]:
        """Return JSON-safe cumulative ROI batch counters."""

        summaries: dict[str, dict[str, Any]] = {}
        for key, state in self._batch_counter_states.items():
            class_counts: dict[str, int] = {}
            for class_value in state.get("class_by_track", {}).values():
                class_counts[class_value] = class_counts.get(class_value, 0) + 1
            summaries[key] = {
                "roi_id": state["roi_id"],
                "count": len(state["counted_track_ids"]),
                "visible_count": state["visible_count"],
                "class_counts": class_counts,
                "minimum_count": state["minimum_count"],
                "empty_frames": state["empty_frames"],
                "required_empty_frames": state["required_empty_frames"],
                "status": state["status"],
            }
        return summaries

    def update(
        self,
        detections: list[Detection],
        frame_id: int,
        time_sec: float,
        hands: list[Any] | None = None,
        events: list[Any] | None = None,
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

        if step.key != self._last_step_key:
            if step.key in self._reset_sequence_on_step_ids:
                for constraint in self._attribute_sequences:
                    constraint.reset()
            self._last_step_key = step.key

        constraint_error = self._validate_attribute_sequences(detections, events or [])
        if constraint_error:
            self.reason = constraint_error
            self._mark_remaining_failed()
            return

        if self._step_started_at is None:
            self._step_started_at = time_sec
        timeout_sec = self.step_timeouts_sec.get(step.key, step.timeout_sec)
        if timeout_sec is not None and timeout_sec <= 0:
            raise ValueError(f"步骤 {step.key} 的超时时间必须大于 0")
        if timeout_sec is not None and time_sec - self._step_started_at > timeout_sec:
            if not step.required:
                step.status = "skipped"
                self.current_index += 1
                self._reset_step_progress(time_sec)
                self.update(detections, frame_id, time_sec, hands=hands, events=events)
                return
            self.reason = f"步骤超时: {step.key} 超过 {timeout_sec} 秒未完成"
            self._mark_remaining_failed()
            return

        trigger_source = self._evaluate_step_trigger(
            step, detections, hands or [], events or [], time_sec
        )
        yolo_hit = trigger_source is not None and trigger_source != "hand_pose"

        # 可选步骤不能阻塞后续必需步骤。后续必需条件已经出现时，将当前
        # 可选步骤标记为 skipped，并用同一帧继续判断下一步骤。
        if not step.required and trigger_source is None:
            later_required = next((item for item in self.steps[self.current_index + 1 :] if item.required), None)
            if later_required is not None and self._evaluate_step_trigger(
                later_required, detections, hands or [], events or [], time_sec
            ):
                step.status = "skipped"
                self.current_index += 1
                self._reset_step_progress(time_sec)
                self.update(detections, frame_id, time_sec, hands=hands, events=events)
                return

        # 顺序错误输出到 reason 字段，供软件端直接展示或记录。
        later_work_classes = {
            trigger_class
            for item in self.steps[self.current_index + 1 :]
            if item.required and item.roi_name == "work"
            for trigger_class in item.trigger_classes
        }
        work_roi = self.rois.get("work")
        wrong_classes = sorted(labels_in_roi(detections, work_roi) & later_work_classes) if work_roi else []
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
        required_frames = 1 if (
            step.event_type or step.trigger_type in {"object_transition", "duration"}
        ) else step.stable_frames or (
            self.tool_hold_frames if step.key == "tool_return" else self.step_stable_frames
        )
        if self._stable_counter < required_frames:
            return

        step.status = "done"
        step.frame = frame_id
        step.time = round(time_sec, 3)
        step.trigger_source = self._stable_trigger_source
        self.current_index += 1
        self._reset_step_progress(time_sec)

        if self.current_index >= len(self.steps):
            self.final_result = "OK"

    def finalize(self, video_name: str) -> dict:
        """生成最终 SOP 结果字典。"""

        for step in self.steps:
            if not step.required and step.status == "pending":
                step.status = "skipped"
        if self.final_result != "OK" and not self.reason:
            missing = [step.name for step in self.steps if step.required and step.status != "done"]
            if missing:
                self.reason = "步骤缺失: " + "、".join(missing)
                self._mark_remaining_failed()
            else:
                self.final_result = "OK"

        return {
            "video_name": video_name,
            "final_result": self.final_result,
            "steps": [step.to_dict() for step in self.steps],
            "reason": self.reason,
            "constraints": [constraint.summary() for constraint in self._attribute_sequences],
            "counters": self.counter_summaries,
        }

    def _validate_attribute_sequences(
        self,
        detections: list[Detection],
        events: list[Any],
    ) -> str | None:
        if not self._attribute_sequences:
            return None
        by_track_id = {
            detection.track_id: detection
            for detection in detections
            if detection.track_id is not None
        }
        for event in events:
            for constraint in self._attribute_sequences:
                if not constraint.matches(event.event_type, event.class_name, event.roi_id):
                    continue
                detection = by_track_id.get(event.track_id)
                if detection is None:
                    return f"属性序列约束 {constraint.constraint_id} 缺少事件目标检测结果"
                valid, detail = constraint.consume(detection)
                if not valid:
                    return f"属性序列约束 {constraint.constraint_id} 校验失败: {detail}"
        return None

    def _mark_remaining_failed(self) -> None:
        for step in self.steps:
            if step.status != "done":
                step.status = "failed" if step.required else "skipped"

    def _reset_step_progress(self, time_sec: float) -> None:
        self._stable_counter = 0
        self._stable_trigger_source = None
        self._step_started_at = time_sec
        self._duration_started_at.clear()

    def _evaluate_step_trigger(
        self,
        step: StepResult,
        detections: list[Detection],
        hands: list[Any],
        events: list[Any],
        time_sec: float,
    ) -> str | None:
        if step.trigger_type in {"composite", "object_count", "object_transition", "roi_batch_removed", "duration"}:
            matched, sources = self._evaluate_condition(
                step.trigger_config or {},
                detections,
                hands,
                events,
                time_sec,
                step.key,
            )
            return "+".join(sorted(sources)) if matched else None
        if step.event_type:
            matched = [
                event
                for event in events
                if event.event_type == step.event_type
                and (not step.trigger_classes or event.class_name in step.trigger_classes)
                and (not step.roi_name or event.roi_id == step.roi_name)
            ]
            if matched:
                return "+".join(sorted({event.class_name for event in matched}))
            return None
        roi = self.rois.get(step.roi_name)
        eligible = [item for item in detections if item.conf >= step.minimum_confidence]
        if step.trigger_type == "object_present":
            matched_classes = {item.class_name for item in eligible}
        else:
            if roi is None:
                raise ValueError(f"步骤 {step.key} 引用了不存在的 ROI: {step.roi_name}")
            matched_classes = {
                item.class_name
                for item in eligible
                if self._detection_matches_evidence(item, roi, step.trigger_config or {})
            }
        trigger_source_mode = self.trigger_sources.get(step.key, "auto")
        if trigger_source_mode not in {"auto", "yolo", "hand_pose"}:
            raise ValueError(f"步骤 {step.key} 的 trigger_source 只支持 auto、yolo、hand_pose")
        yolo_sources = (
            sorted(set(step.trigger_classes) & matched_classes)
            if trigger_source_mode in {"auto", "yolo"}
            else []
        )
        hand_hit = (
            (step.allow_hand_pose or step.trigger_type == "hand_in_roi")
            and trigger_source_mode in {"auto", "hand_pose"}
            and roi is not None
            and self._hand_pose_in_roi(hands, roi)
        )
        return self._resolve_trigger_source(yolo_sources, hand_hit)

    def _evaluate_condition(
        self,
        condition: dict[str, Any],
        detections: list[Detection],
        hands: list[Any],
        events: list[Any],
        time_sec: float,
        state_key: str,
    ) -> tuple[bool, set[str]]:
        condition_type = str(condition.get("type", "object_in_roi"))
        minimum_confidence = float(condition.get("confidence", 0.0))
        eligible = [item for item in detections if item.conf >= minimum_confidence]
        class_name = str(condition.get("class_name", "")).strip()
        class_names = {
            str(value).strip()
            for value in condition.get("class_names", [])
            if isinstance(value, str) and value.strip()
        }
        roi_id = str(condition.get("roi_id", "")).strip()

        if condition_type == "composite":
            operator = str(condition.get("operator", "all")).lower()
            evaluated = [
                self._evaluate_condition(
                    item,
                    detections,
                    hands,
                    events,
                    time_sec,
                    f"{state_key}.{index}",
                )
                for index, item in enumerate(condition.get("conditions", []))
            ]
            matched = all(item[0] for item in evaluated) if operator == "all" else any(
                item[0] for item in evaluated
            )
            sources = {
                source
                for item_matched, item_sources in evaluated
                if item_matched
                for source in item_sources
            }
            return matched, sources

        if condition_type == "object_count":
            roi = self.rois.get(roi_id) if roi_id else None
            matched_detections = [
                item
                for item in eligible
                if (
                    (not class_name and not class_names)
                    or item.class_name == class_name
                    or item.class_name in class_names
                )
                and (roi is None or bbox_center_in_roi(item.bbox, roi))
            ]
            count = len({item.track_id for item in matched_detections if item.track_id is not None})
            count += sum(1 for item in matched_detections if item.track_id is None)
            minimum = int(condition.get("min_count", condition.get("count", 1)))
            maximum = condition.get("max_count")
            matched = count >= minimum and (maximum is None or count <= int(maximum))
            group_name = class_name or "+".join(sorted(class_names)) or "*"
            return matched, {f"count:{group_name}={count}"} if matched else set()

        if condition_type == "roi_batch_removed":
            roi = self.rois.get(roi_id)
            if roi is None:
                return False, set()
            matched_detections = [
                item
                for item in eligible
                if item.track_id is not None
                and (
                    (not class_name and not class_names)
                    or item.class_name == class_name
                    or item.class_name in class_names
                )
                and self._detection_matches_evidence(item, roi, condition)
            ]
            state = self._batch_counter_states.setdefault(
                state_key,
                {
                    "roi_id": roi_id,
                    "candidate_hits": {},
                    "counted_track_ids": set(),
                    "class_by_track": {},
                    "visible_count": 0,
                    "minimum_count": int(condition.get("min_count", 1)),
                    "empty_frames": 0,
                    "required_empty_frames": int(condition.get("empty_stable_frames", 2)),
                    "status": "waiting",
                    "removed": False,
                },
            )
            count = len(state["counted_track_ids"])
            if state["removed"]:
                return True, {f"batch_removed:{roi_id}:count={count}"}

            visible_by_track = {item.track_id: item for item in matched_detections}
            state["visible_count"] = len(visible_by_track)
            stable_frames = int(condition.get("count_stable_frames", 2))
            candidate_hits: dict[int, int] = state["candidate_hits"]
            for track_id, item in visible_by_track.items():
                if track_id in state["counted_track_ids"]:
                    continue
                candidate_hits[track_id] = candidate_hits.get(track_id, 0) + 1
                if candidate_hits[track_id] >= stable_frames:
                    state["counted_track_ids"].add(track_id)
                    state["class_by_track"][track_id] = item.class_name
                    candidate_hits.pop(track_id, None)
            for track_id in list(candidate_hits):
                if track_id not in visible_by_track:
                    candidate_hits.pop(track_id, None)

            count = len(state["counted_track_ids"])
            if visible_by_track:
                state["empty_frames"] = 0
                state["status"] = "ready_to_remove" if count >= state["minimum_count"] else "stacking"
            elif count >= state["minimum_count"]:
                state["empty_frames"] += 1
                state["status"] = "clearing"
                if state["empty_frames"] >= state["required_empty_frames"]:
                    state["removed"] = True
                    state["status"] = "removed"
                    return True, {f"batch_removed:{roi_id}:count={count}"}
            return False, set()

        if condition_type == "object_transition":
            from_roi_id = str(condition.get("from_roi_id", "")).strip()
            to_roi_id = str(condition.get("to_roi_id", "")).strip()
            matched_events = [
                event
                for event in events
                if event.event_type == "object_move_roi"
                and (not class_name or event.class_name == class_name)
                and event.from_roi_id == from_roi_id
                and event.to_roi_id == to_roi_id
            ]
            return bool(matched_events), {
                f"move:{event.class_name}:{from_roi_id}->{to_roi_id}"
                for event in matched_events
            }

        if condition_type == "object_event":
            event_type = str(condition.get("event", "")).strip()
            matched_events = [
                event
                for event in events
                if event.event_type == event_type
                and (not class_name or event.class_name == class_name)
                and (not roi_id or event.roi_id == roi_id)
            ]
            return bool(matched_events), {event.class_name for event in matched_events}

        if condition_type == "duration":
            nested = condition.get("condition", {})
            nested_matched, sources = self._evaluate_condition(
                nested,
                detections,
                hands,
                events,
                time_sec,
                f"{state_key}.condition",
            )
            if not nested_matched:
                self._duration_started_at.pop(state_key, None)
                return False, set()
            started_at = self._duration_started_at.setdefault(state_key, time_sec)
            required_seconds = float(condition.get("duration_sec", 0.0))
            return time_sec - started_at >= required_seconds, sources

        if condition_type == "object_present":
            matched = {
                item.class_name
                for item in eligible
                if not class_name or item.class_name == class_name
            }
            return bool(matched), matched

        if condition_type == "hand_in_roi":
            roi = self.rois.get(roi_id)
            matched = roi is not None and self._hand_pose_in_roi(hands, roi)
            return matched, {"hand_pose"} if matched else set()

        if condition_type == "object_in_roi":
            roi = self.rois.get(roi_id)
            if roi is None:
                return False, set()
            matched = {
                item.class_name
                for item in eligible
                if (not class_name or item.class_name == class_name)
                and self._detection_matches_evidence(item, roi, condition)
            }
            return bool(matched), matched

        return False, set()

    def _hand_pose_in_roi(self, hands: list[Any], roi: Sequence[int]) -> bool:
        """判断任意手部关键点是否进入指定 ROI。"""

        return any(point_in_roi((x, y), roi) for hand in hands for x, y, _ in hand.points)

    @staticmethod
    def _detection_matches_evidence(
        detection: Detection,
        roi: Sequence[int],
        trigger: dict[str, Any],
    ) -> bool:
        evidence = trigger.get("evidence", {})
        if not isinstance(evidence, dict):
            evidence = {}
        evidence_type = str(evidence.get("type", "bbox")).strip().lower()
        if evidence_type == "bbox":
            return bbox_center_in_roi(detection.bbox, roi)
        if evidence_type == "mask":
            threshold = float(evidence.get("min_roi_overlap", 0.35))
            return detection.mask is not None and mask_roi_overlap(detection.mask, roi) >= threshold
        if evidence_type == "keypoints":
            indices = evidence.get("indices")
            selected = detection.keypoints or []
            if isinstance(indices, list):
                selected = [selected[index] for index in indices if isinstance(index, int) and 0 <= index < len(selected)]
            minimum_score = float(evidence.get("min_keypoint_score", 0.0))
            return any(
                len(point) >= 3
                and float(point[2]) >= minimum_score
                and point_in_roi((float(point[0]), float(point[1])), roi)
                for point in selected
            )
        raise ValueError(f"不支持的视觉证据类型: {evidence_type}")

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

    @staticmethod
    def _build_step_definitions(workflow: dict[str, Any]) -> list[dict[str, Any]]:
        """将项目 workflow.json 转成状态机内部步骤结构。"""

        definitions: list[dict[str, Any]] = []
        for step in sorted(workflow.get("steps", []), key=lambda item: item["order"]):
            trigger = step["trigger"]
            trigger_type = trigger["type"]
            class_name = trigger.get("class_name")
            definitions.append(
                {
                    "step_id": step["order"],
                    "key": str(step["id"]),
                    "name": str(step["name"]),
                    "trigger_classes": [str(class_name)] if class_name else [],
                    "roi_name": str(trigger.get("roi_id", "")),
                    "allow_hand_pose": bool(trigger.get("allow_hand_pose", trigger_type == "hand_in_roi")),
                    "trigger_type": trigger_type,
                    "minimum_confidence": float(trigger.get("confidence", 0.0)),
                    "stable_frames": int(trigger.get("stable_frames", 3)),
                    "timeout_sec": float(step["timeout_sec"]) if step.get("timeout_sec") is not None else None,
                    "required": bool(step.get("required", True)),
                    "event_type": (
                        str(trigger.get("event"))
                        if trigger.get("event")
                        else "object_enter_roi" if trigger.get("require_transition") else None
                    ),
                    "trigger_config": dict(trigger),
                }
            )
        return definitions
