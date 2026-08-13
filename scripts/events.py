"""根据跟踪目标变化生成SOP对象与ROI事件。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from scripts.inference import Detection
from scripts.utils import bbox_center_in_roi


@dataclass(frozen=True)
class SopEvent:
    event_type: str
    frame_id: int
    time_sec: float
    class_name: str
    track_id: int
    roi_id: str | None = None
    from_roi_id: str | None = None
    to_roi_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "type": self.event_type,
            "frame": self.frame_id,
            "time": round(self.time_sec, 3),
            "class_name": self.class_name,
            "track_id": self.track_id,
            "roi_id": self.roi_id,
            "from_roi_id": self.from_roi_id,
            "to_roi_id": self.to_roi_id,
        }


class SopEventEngine:
    """将带 track_id 的检测结果转换为对象和 ROI 状态变化事件。"""

    def __init__(
        self,
        rois: dict[str, Sequence[int]],
        lost_tolerance_frames: int = 8,
        enter_stable_frames: int = 1,
        exit_stable_frames: int = 1,
    ) -> None:
        if lost_tolerance_frames < 0:
            raise ValueError("lost_tolerance_frames 不能小于 0")
        if enter_stable_frames < 1 or exit_stable_frames < 1:
            raise ValueError("enter_stable_frames 和 exit_stable_frames 必须大于 0")
        self.rois = rois
        self.lost_tolerance_frames = lost_tolerance_frames
        self.enter_stable_frames = enter_stable_frames
        self.exit_stable_frames = exit_stable_frames
        self._visible: dict[int, tuple[str, set[str]]] = {}
        self._missing_frames: dict[int, int] = {}
        self._last_roi_by_track: dict[int, str] = {}
        self._enter_counts: dict[tuple[int, str], int] = {}
        self._exit_counts: dict[tuple[int, str], int] = {}

    def update(self, detections: list[Detection], frame_id: int, time_sec: float) -> list[SopEvent]:
        events: list[SopEvent] = []
        current: dict[int, tuple[str, set[str]]] = {}
        for detection in detections:
            if detection.track_id is None:
                continue
            inside = {
                roi_id
                for roi_id, roi in self.rois.items()
                if bbox_center_in_roi(detection.bbox, roi)
            }
            current[detection.track_id] = (detection.class_name, inside)
            previous = self._visible.get(detection.track_id)
            self._missing_frames.pop(detection.track_id, None)
            if previous is None:
                events.append(SopEvent("object_appear", frame_id, time_sec, detection.class_name, detection.track_id))
                previous_rois = inside
                confirmed_inside = set(inside)
            else:
                previous_rois = previous[1]
                confirmed_inside = set(previous_rois)
            for key in list(self._enter_counts):
                if key[0] == detection.track_id and key[1] not in inside - previous_rois:
                    self._enter_counts.pop(key, None)
            for key in list(self._exit_counts):
                if key[0] == detection.track_id and key[1] in inside:
                    self._exit_counts.pop(key, None)
            for roi_id in sorted(inside - previous_rois):
                key = (detection.track_id, roi_id)
                self._enter_counts[key] = self._enter_counts.get(key, 0) + 1
                if self._enter_counts[key] < self.enter_stable_frames:
                    continue
                self._enter_counts.pop(key, None)
                confirmed_inside.add(roi_id)
                events.append(SopEvent("object_enter_roi", frame_id, time_sec, detection.class_name, detection.track_id, roi_id))
                from_roi_id = self._last_roi_by_track.get(detection.track_id)
                if from_roi_id is not None and from_roi_id != roi_id:
                    events.append(
                        SopEvent(
                            "object_move_roi",
                            frame_id,
                            time_sec,
                            detection.class_name,
                            detection.track_id,
                            roi_id,
                            from_roi_id,
                            roi_id,
                        )
                    )
                self._last_roi_by_track[detection.track_id] = roi_id
            for roi_id in sorted(previous_rois - inside):
                key = (detection.track_id, roi_id)
                self._exit_counts[key] = self._exit_counts.get(key, 0) + 1
                if self._exit_counts[key] < self.exit_stable_frames:
                    continue
                self._exit_counts.pop(key, None)
                confirmed_inside.discard(roi_id)
                events.append(SopEvent("object_exit_roi", frame_id, time_sec, detection.class_name, detection.track_id, roi_id))
            for roi_id in inside & previous_rois:
                self._enter_counts.pop((detection.track_id, roi_id), None)
                self._exit_counts.pop((detection.track_id, roi_id), None)
            if confirmed_inside and detection.track_id not in self._last_roi_by_track:
                self._last_roi_by_track[detection.track_id] = sorted(confirmed_inside)[0]
            current[detection.track_id] = (detection.class_name, confirmed_inside)

        retained: dict[int, tuple[str, set[str]]] = {}
        for track_id, (class_name, previous_rois) in self._visible.items():
            if track_id in current:
                continue
            missing_frames = self._missing_frames.get(track_id, 0) + 1
            if missing_frames <= self.lost_tolerance_frames:
                self._missing_frames[track_id] = missing_frames
                retained[track_id] = (class_name, previous_rois)
                self._clear_pending(track_id)
                continue
            for roi_id in sorted(previous_rois):
                events.append(SopEvent("object_exit_roi", frame_id, time_sec, class_name, track_id, roi_id))
            events.append(SopEvent("object_disappear", frame_id, time_sec, class_name, track_id))
            self._missing_frames.pop(track_id, None)
            self._last_roi_by_track.pop(track_id, None)
            self._clear_pending(track_id)

        self._visible = {**retained, **current}
        return events

    def _clear_pending(self, track_id: int) -> None:
        self._enter_counts = {
            key: value for key, value in self._enter_counts.items() if key[0] != track_id
        }
        self._exit_counts = {
            key: value for key, value in self._exit_counts.items() if key[0] != track_id
        }
