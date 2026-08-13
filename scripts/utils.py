"""检测流程共享的通用辅助函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


# 检测框几何工具：计算 bbox 中心点。
def bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    """返回检测框中心点坐标。"""

    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


# ROI 几何工具：判断点是否落在区域内。
def point_in_roi(point: tuple[float, float], roi: Sequence[int]) -> bool:
    """判断点是否位于 ROI 内。"""

    x, y = point
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2


# SOP 判定使用检测框中心点是否进入 ROI。
def bbox_center_in_roi(bbox: Sequence[float], roi: Sequence[int]) -> bool:
    """判断检测框中心点是否位于 ROI 内。"""

    return point_in_roi(bbox_center(bbox), roi)


# 文件工具：创建目录并返回 Path。
def ensure_dir(path: Path) -> Path:
    """确保目录存在，并返回该目录路径。"""

    path.mkdir(parents=True, exist_ok=True)
    return path


# SOP 工具：获取某个 ROI 内出现过的类别集合。
def labels_in_roi(detections: Iterable, roi: Sequence[int]) -> set[str]:
    """返回中心点落入 ROI 的检测类别集合。"""

    return {
        detection.class_name
        for detection in detections
        if bbox_center_in_roi(detection.bbox, roi)
    }


def mask_roi_overlap(mask: Sequence[Sequence[float]] | None, roi: Sequence[int]) -> float:
    """Return the fraction of a segmentation polygon covered by ``roi``."""

    if not mask or len(mask) < 3:
        return 0.0
    points = np.asarray(mask, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        return 0.0
    x1, y1, x2, y2 = [max(0, int(round(value))) for value in roi]
    max_x = max(x2, int(np.ceil(points[:, 0].max()))) + 1
    max_y = max(y2, int(np.ceil(points[:, 1].max()))) + 1
    if max_x <= 1 or max_y <= 1:
        return 0.0
    canvas = np.zeros((max_y, max_x), dtype=np.uint8)
    cv2.fillPoly(canvas, [np.round(points).astype(np.int32)], 1)
    total = int(canvas.sum())
    if total <= 0:
        return 0.0
    intersection = int(canvas[y1:y2, x1:x2].sum())
    return intersection / total
