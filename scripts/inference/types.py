"""跨推理后端共享的检测结果数据类型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """统一的单目标检测结果，坐标格式为原始图像像素 xyxy。"""

    class_name: str
    conf: float
    bbox: list[float]
    track_id: int | None = None
    class_id: int | None = None

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class": self.class_name,
            "class_name": self.class_name,
            "conf": round(self.conf, 4),
            "confidence": round(self.conf, 4),
            "bbox": [round(value, 2) for value in self.bbox],
            "track_id": self.track_id,
        }
