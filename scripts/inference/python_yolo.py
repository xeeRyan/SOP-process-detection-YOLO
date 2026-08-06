"""基于Ultralytics的Python模型推理后端。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from scripts.config import CONFIDENCE_THRESHOLD, DEFAULT_NMS_THRESHOLD
from scripts.legacy_sk_config import TARGET_CLASSES
from scripts.inference.base import DetectorBackend
from scripts.inference.types import Detection


class PythonYoloBackend(DetectorBackend):
    """基于Ultralytics Python API的PT/ONNX/TensorRT推理实现。"""

    backend_name = "python"
    runtime_name = "ultralytics"

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = CONFIDENCE_THRESHOLD,
        target_classes: Iterable[str] = TARGET_CLASSES,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        device: str | int | None = None,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"未找到模型权重: {model_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("缺少 ultralytics，请先安装 requirements.txt 中的依赖。") from exc

        self.model_path = model_path
        self.model = YOLO(str(model_path))
        self.conf_threshold = conf_threshold
        self.target_classes = set(target_classes)
        self.nms_threshold = nms_threshold
        self.device = resolve_inference_device(model_path, device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        predict_options = {
            "conf": self.conf_threshold,
            "iou": self.nms_threshold,
            "verbose": False,
        }
        if self.device is not None:
            predict_options["device"] = self.device
        results = self.model.predict(frame, **predict_options)
        if not results or results[0].boxes is None:
            return []

        detections: list[Detection] = []
        result = results[0]
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            if self.target_classes and class_name not in self.target_classes:
                continue
            detections.append(
                Detection(
                    class_name=class_name,
                    conf=float(box.conf[0]),
                    bbox=box.xyxy[0].tolist(),
                    class_id=class_id,
                )
            )
        return detections


def resolve_inference_device(
    model_path: Path,
    requested: str | int | None,
) -> str | int | None:
    """自动模式下让ONNX默认使用CPU，避免CUDA张量绑定到CPU会话。"""

    if requested not in (None, "", "auto"):
        return requested
    if model_path.suffix.lower() == ".onnx":
        return "cpu"
    return None
