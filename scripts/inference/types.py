"""跨推理后端共享的检测结果数据类型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """统一的单目标视觉结果，坐标格式为原始图像像素 xyxy。

    ``mask`` 和 ``keypoints`` 是可选证据：检测/跟踪链只依赖 bbox，
    SOP 规则可按工作流声明消费分割轮廓或姿态关键点。
    """

    class_name: str
    conf: float
    bbox: list[float]
    track_id: int | None = None
    class_id: int | None = None
    mask: list[list[float]] | None = None
    keypoints: list[list[float]] | None = None
    attributes: dict[str, str | int | float | bool] | None = None

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class": self.class_name,
            "class_name": self.class_name,
            "conf": round(self.conf, 4),
            "confidence": round(self.conf, 4),
            "bbox": [round(value, 2) for value in self.bbox],
            "track_id": self.track_id,
            "mask": (
                [[round(point[0], 2), round(point[1], 2)] for point in self.mask]
                if self.mask is not None
                else None
            ),
            "keypoints": (
                [[round(value, 2) for value in point] for point in self.keypoints]
                if self.keypoints is not None
                else None
            ),
            "attributes": dict(self.attributes) if self.attributes is not None else None,
        }
