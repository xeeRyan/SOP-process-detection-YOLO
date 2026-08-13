"""依据模型格式和后端选项创建统一检测器。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from scripts.config import CONFIDENCE_THRESHOLD, DEFAULT_NMS_THRESHOLD
from scripts.inference.base import DetectorBackend
from scripts.inference.python_yolo import PythonYoloBackend


PYTHON_BACKEND_NAMES = {"python", "ultralytics"}


def build_detector(
    model_path: str | Path,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    target_classes: Iterable[str] | None = None,
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    device: str | int | None = None,
    backend: str = "python",
    *,
    task: str = "detect",
) -> DetectorBackend:
    normalized = str(backend or "python").strip().lower()
    if normalized not in PYTHON_BACKEND_NAMES:
        raise ValueError(
            f"当前Python阶段不支持推理后端 {backend!r}；"
            f"可选值: {', '.join(sorted(PYTHON_BACKEND_NAMES))}"
        )
    if target_classes is None:
        raise ValueError("build_detector必须显式提供 target_classes（来自SOP项目配置）")
    target_classes = tuple(str(item) for item in target_classes)
    if not target_classes:
        raise ValueError("target_classes不能为空")
    return PythonYoloBackend(
        model_path=model_path,
        task=task,
        conf_threshold=conf_threshold,
        target_classes=target_classes,
        nms_threshold=nms_threshold,
        device=device,
    )
