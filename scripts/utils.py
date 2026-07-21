from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence


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


# 文件工具：写出接口 JSON。
def write_json(path: Path, data: dict) -> None:
    """将接口结果写入 UTF-8 JSON 文件。"""

    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# 路径工具：解析外部传入路径。
def normalize_path(path: str | Path) -> Path:
    """展开并解析路径，供外部传参时使用。"""

    return Path(path).expanduser().resolve()


# SOP 工具：获取某个 ROI 内出现过的类别集合。
def labels_in_roi(detections: Iterable, roi: Sequence[int]) -> set[str]:
    """返回中心点落入 ROI 的检测类别集合。"""

    return {
        detection.class_name
        for detection in detections
        if bbox_center_in_roi(detection.bbox, roi)
    }
