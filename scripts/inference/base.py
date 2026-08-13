"""推理后端必须实现的抽象检测器接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from scripts.inference.types import Detection


class DetectorBackend(ABC):
    """Python与后续C++推理实现共同遵循的最小接口。"""

    backend_name = "unknown"
    runtime_name = "unknown"
    task = "detect"
    device: str | int | None = None

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """对一帧BGR图像执行推理，返回原图像素坐标检测框。"""

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "runtime": self.runtime_name,
            "task": self.task,
            "device": self.device,
        }

    def close(self) -> None:
        """释放后端资源；无显式资源的实现可保持空操作。"""
