from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from scripts.config import CONFIDENCE_THRESHOLD, DEFAULT_NMS_THRESHOLD, TARGET_CLASSES


# 统一的检测结果对象，供 SOP 状态机、可视化和接口返回共同使用。
@dataclass(frozen=True)
class Detection:
    """目标检测输出对象。"""

    class_name: str
    conf: float
    bbox: list[float]

    def to_dict(self) -> dict:
        """转换为可写入 JSON 或接口返回的字典。"""

        return {
            "class": self.class_name,
            "conf": round(self.conf, 4),
            "bbox": [round(value, 2) for value in self.bbox],
        }


# YOLO26s 检测器，对外只暴露 detect(frame) 这一类简单接口。
class YOLODetector:
    """YOLO26s 目标检测服务封装。"""

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = CONFIDENCE_THRESHOLD,
        target_classes: Iterable[str] = TARGET_CLASSES,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"未找到模型权重: {model_path}. "
                "请先训练并放置 models/best_yolo26s.pt，"
                "或通过 --model 指定已有权重。"
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("缺少 ultralytics，请先安装 requirements.txt 中的依赖。") from exc

        self.model = YOLO(str(model_path))
        self.conf_threshold = conf_threshold
        self.target_classes = set(target_classes)
        self.nms_threshold = nms_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """检测单帧 OpenCV BGR 图像。"""

        results = self.model.predict(frame, conf=self.conf_threshold, iou=self.nms_threshold, verbose=False)
        if not results:
            return []

        result = results[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]

            # 只保留 SOP 工序检测关心的类别，减少后续状态机干扰。
            if self.target_classes and class_name not in self.target_classes:
                continue

            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detections.append(Detection(class_name=class_name, conf=conf, bbox=xyxy))

        return detections


def build_detector(
    model_path: str | Path,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    target_classes: Iterable[str] = TARGET_CLASSES,
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
) -> YOLODetector:
    """创建目标检测器，供主流程和 API 复用。"""

    return YOLODetector(
        model_path,
        conf_threshold=conf_threshold,
        target_classes=target_classes,
        nms_threshold=nms_threshold,
    )
