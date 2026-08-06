"""依据模型格式和后端选项创建统一检测器。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from scripts.config import CONFIDENCE_THRESHOLD, DEFAULT_NMS_THRESHOLD
from scripts.legacy_sk_config import TARGET_CLASSES
from scripts.inference.base import DetectorBackend
from scripts.inference.python_yolo import PythonYoloBackend


PYTHON_BACKEND_NAMES = {"python", "ultralytics"}


def build_detector(
    model_path: str | Path,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    target_classes: Iterable[str] = TARGET_CLASSES,
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    device: str | int | None = None,
    backend: str = "python",
) -> DetectorBackend:
    normalized = str(backend or "python").strip().lower()
    if normalized not in PYTHON_BACKEND_NAMES:
        raise ValueError(
            f"当前Python阶段不支持推理后端 {backend!r}；"
            f"可选值: {', '.join(sorted(PYTHON_BACKEND_NAMES))}"
        )
    return PythonYoloBackend(
        model_path=model_path,
        conf_threshold=conf_threshold,
        target_classes=target_classes,
        nms_threshold=nms_threshold,
        device=device,
    )
