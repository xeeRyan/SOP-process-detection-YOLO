"""SOPAID唯一目标跟踪实现：ByteTrack两阶段关联与卡尔曼运动预测。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import lap
import numpy as np

from scripts.inference import Detection


class TrackState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


class _KalmanBoxFilter:
    """使用中心点、尺寸及其速度预测检测框的常速度卡尔曼滤波器。"""

    def __init__(self, bbox: list[float]) -> None:
        self._state = np.zeros((8, 1), dtype=np.float64)
        self._state[:4, 0] = _bbox_to_measurement(bbox)
        self._transition = np.eye(8, dtype=np.float64)
        self._transition[0, 4] = 1.0
        self._transition[1, 5] = 1.0
        self._transition[2, 6] = 1.0
        self._transition[3, 7] = 1.0
        self._measurement = np.zeros((4, 8), dtype=np.float64)
        self._measurement[:4, :4] = np.eye(4, dtype=np.float64)
        self._covariance = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 25.0, 25.0])
        self._process_noise = np.diag([1.0, 1.0, 0.25, 0.25, 4.0, 4.0, 1.0, 1.0])
        self._measurement_noise = np.diag([2.0, 2.0, 4.0, 4.0])

    def predict(self) -> list[float]:
        self._state = self._transition @ self._state
        self._covariance = (
            self._transition @ self._covariance @ self._transition.T
            + self._process_noise
        )
        self._clamp_size()
        return _measurement_to_bbox(self._state[:4, 0])

    def update(self, bbox: list[float]) -> list[float]:
        observation = _bbox_to_measurement(bbox).reshape(4, 1)
        innovation = observation - self._measurement @ self._state
        innovation_covariance = (
            self._measurement @ self._covariance @ self._measurement.T
            + self._measurement_noise
        )
        gain = (
            self._covariance
            @ self._measurement.T
            @ np.linalg.inv(innovation_covariance)
        )
        self._state = self._state + gain @ innovation
        identity = np.eye(8, dtype=np.float64)
        self._covariance = (identity - gain @ self._measurement) @ self._covariance
        self._clamp_size()
        return _measurement_to_bbox(self._state[:4, 0])

    def _clamp_size(self) -> None:
        self._state[2, 0] = max(1.0, self._state[2, 0])
        self._state[3, 0] = max(1.0, self._state[3, 0])


@dataclass
class _Track:
    track_id: int
    class_name: str
    bbox: list[float]
    hits: int
    state: TrackState
    motion_filter: _KalmanBoxFilter
    last_confidence: float
    last_attributes: dict[str, str | int | float | bool] | None = None
    missed_frames: int = 0


class ByteTrackTracker:
    """高低置信度两阶段关联、全局分配和运动预测跟踪器。

    高置信度框优先匹配；低置信度框只能恢复未匹配轨迹，不能创建新轨迹。
    每帧先通过卡尔曼滤波预测位置，再用观测框校正状态。
    """

    def __init__(
        self,
        high_confidence: float = 0.5,
        low_confidence: float = 0.1,
        new_track_confidence: float = 0.6,
        iou_threshold: float = 0.3,
        second_match_iou_threshold: float = 0.2,
        max_missing_frames: int = 8,
        min_confirmed_hits: int = 1,
    ) -> None:
        if not 0 <= low_confidence <= high_confidence <= new_track_confidence <= 1:
            raise ValueError(
                "置信度阈值必须满足 0 <= low_confidence <= high_confidence "
                "<= new_track_confidence <= 1"
            )
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold 必须在 0 到 1 之间")
        if not 0 <= second_match_iou_threshold <= 1:
            raise ValueError("second_match_iou_threshold 必须在 0 到 1 之间")
        if max_missing_frames < 0:
            raise ValueError("max_missing_frames 不能小于 0")
        if min_confirmed_hits < 1:
            raise ValueError("min_confirmed_hits 必须大于等于 1")
        self.high_confidence = high_confidence
        self.low_confidence = low_confidence
        self.new_track_confidence = new_track_confidence
        self.iou_threshold = iou_threshold
        self.second_match_iou_threshold = second_match_iou_threshold
        self.max_missing_frames = max_missing_frames
        self.min_confirmed_hits = min_confirmed_hits
        self.reset()

    def reset(self) -> None:
        """清除全部轨迹，并从1重新分配ID。"""

        self._next_track_id = 1
        self._tracks: dict[int, _Track] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        """关联当前帧，只返回已确认且带track_id的检测结果。"""

        self._predict_tracks()
        high_indices = [
            index for index, detection in enumerate(detections)
            if detection.conf >= self.high_confidence
        ]
        low_indices = [
            index for index, detection in enumerate(detections)
            if self.low_confidence <= detection.conf < self.high_confidence
        ]

        assignments = _global_assign(
            self._tracks,
            list(self._tracks),
            detections,
            high_indices,
            self.iou_threshold,
        )
        matched_track_ids = set(assignments.values())
        second_assignments = _global_assign(
            self._tracks,
            [track_id for track_id in self._tracks if track_id not in matched_track_ids],
            detections,
            low_indices,
            self.second_match_iou_threshold,
        )
        assignments.update(second_assignments)
        matched_track_ids.update(second_assignments.values())

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_track_ids:
                continue
            track.missed_frames += 1
            track.state = TrackState.LOST
            if track.missed_frames > self.max_missing_frames or (
                track.hits < self.min_confirmed_hits and track.missed_frames > 0
            ):
                del self._tracks[track_id]

        tracked_by_index: dict[int, Detection] = {}
        for detection_index, track_id in assignments.items():
            track = self._tracks[track_id]
            detection = detections[detection_index]
            track.bbox = track.motion_filter.update(detection.bbox)
            track.last_confidence = detection.conf
            track.last_attributes = (
                dict(detection.attributes)
                if detection.attributes is not None
                else track.last_attributes
            )
            track.hits += 1
            track.missed_frames = 0
            track.state = (
                TrackState.CONFIRMED
                if track.hits >= self.min_confirmed_hits
                else TrackState.TENTATIVE
            )
            if track.state is TrackState.CONFIRMED:
                tracked_by_index[detection_index] = replace(detection, track_id=track_id)

        for detection_index in high_indices:
            if detection_index in assignments:
                continue
            detection = detections[detection_index]
            if detection.conf < self.new_track_confidence:
                continue
            track = self._new_track(detection)
            if track.state is TrackState.CONFIRMED:
                tracked_by_index[detection_index] = replace(detection, track_id=track.track_id)

        return [tracked_by_index[index] for index in sorted(tracked_by_index)]

    def predict(self) -> list[Detection]:
        """返回确认轨迹的预测框，不增加丢失计数。"""

        self._predict_tracks()
        return [
            Detection(
                class_name=track.class_name,
                conf=track.last_confidence,
                bbox=list(track.bbox),
                track_id=track.track_id,
                attributes=(
                    dict(track.last_attributes)
                    if track.last_attributes is not None
                    else None
                ),
            )
            for track in self._tracks.values()
            if track.state is TrackState.CONFIRMED
        ]

    def _new_track(self, detection: Detection) -> _Track:
        track = _Track(
            track_id=self._next_track_id,
            class_name=detection.class_name,
            bbox=list(detection.bbox),
            hits=1,
            state=(
                TrackState.CONFIRMED
                if self.min_confirmed_hits == 1
                else TrackState.TENTATIVE
            ),
            motion_filter=_KalmanBoxFilter(detection.bbox),
            last_confidence=detection.conf,
            last_attributes=(
                dict(detection.attributes)
                if detection.attributes is not None
                else None
            ),
        )
        self._tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    def _predict_tracks(self) -> None:
        for track in self._tracks.values():
            track.bbox = track.motion_filter.predict()


def _global_assign(
    tracks: dict[int, _Track],
    track_ids: list[int],
    detections: list[Detection],
    detection_indices: list[int],
    iou_threshold: float,
) -> dict[int, int]:
    if not track_ids or not detection_indices:
        return {}

    invalid_cost = 2.0
    costs = np.full((len(track_ids), len(detection_indices)), invalid_cost, dtype=np.float64)
    for track_index, track_id in enumerate(track_ids):
        track = tracks[track_id]
        for column, detection_index in enumerate(detection_indices):
            detection = detections[detection_index]
            if track.class_name != detection.class_name:
                continue
            iou = _bbox_iou(track.bbox, detection.bbox)
            if iou >= iou_threshold:
                costs[track_index, column] = 1.0 - iou

    _, row_assignments, _ = lap.lapjv(costs, extend_cost=True)
    return {
        detection_indices[column]: track_ids[track_index]
        for track_index, column in enumerate(row_assignments)
        if column >= 0 and costs[track_index, column] < invalid_cost
    }


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


def _bbox_to_measurement(bbox: list[float]) -> np.ndarray:
    width = max(1.0, float(bbox[2]) - float(bbox[0]))
    height = max(1.0, float(bbox[3]) - float(bbox[1]))
    return np.array(
        [
            float(bbox[0]) + width / 2.0,
            float(bbox[1]) + height / 2.0,
            width,
            height,
        ],
        dtype=np.float64,
    )


def _measurement_to_bbox(measurement: np.ndarray) -> list[float]:
    center_x, center_y, width, height = (float(value) for value in measurement)
    width = max(1.0, width)
    height = max(1.0, height)
    return [
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    ]
