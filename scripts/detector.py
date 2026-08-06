"""旧检测模块的兼容入口。

新代码应从 ``scripts.inference`` 导入。保留本模块是为了兼容现有
SOP逻辑、测试以及外部调用方。
"""

from scripts.inference import Detection, DetectorBackend, build_detector
from scripts.inference.python_yolo import (
    PythonYoloBackend,
    resolve_inference_device,
)

YOLODetector = PythonYoloBackend
_resolve_inference_device = resolve_inference_device

__all__ = [
    "Detection",
    "DetectorBackend",
    "PythonYoloBackend",
    "YOLODetector",
    "build_detector",
    "_resolve_inference_device",
]
