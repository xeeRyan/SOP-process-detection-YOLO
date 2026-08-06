"""轻量目标关联和短时丢失容忍跟踪。"""

from __future__ import annotations

from dataclasses import dataclass, replace

from scripts.inference import Detection


@dataclass
class _Track:
    track_id: int
    class_name: str
    bbox: list[float]
    missed_frames: int = 0


class SimpleObjectTracker:
    """按类别和 IoU 关联相邻帧检测框的轻量跟踪器。"""

    def __init__(self, iou_threshold: float = 0.2, max_missing_frames: int = 8) -> None:
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self._next_track_id = 1
        self._tracks: dict[int, _Track] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        for track in self._tracks.values():
            track.missed_frames += 1

        candidates: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            for track_id, track in self._tracks.items():
                if track.class_name != detection.class_name:
                    continue
                score = _bbox_iou(track.bbox, detection.bbox)
                if score >= self.iou_threshold:
                    candidates.append((score, track_id, detection_index))

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        assignments: dict[int, int] = {}
        for _, track_id, detection_index in sorted(candidates, reverse=True):
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            assignments[detection_index] = track_id

        tracked: list[Detection] = []
        for index, detection in enumerate(detections):
            track_id = assignments.get(index)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[track_id] = _Track(track_id, detection.class_name, list(detection.bbox))
            else:
                track = self._tracks[track_id]
                track.bbox = list(detection.bbox)
                track.missed_frames = 0
            tracked.append(replace(detection, track_id=track_id))

        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missed_frames > self.max_missing_frames
        ]
        for track_id in expired:
            del self._tracks[track_id]
        return tracked


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
