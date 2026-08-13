"""统一的模型推理接口。

当前阶段只注册 Python 实现；后续 C++ 接口将作为并列后端接入，
不会替换或修改 Python 后端的对外协议。
"""

from scripts.inference.base import DetectorBackend
from scripts.inference.factory import build_detector
from scripts.inference.fusion import DetectionFusion
from scripts.inference.router import ModelRouter
from scripts.inference.types import Detection

__all__ = ["Detection", "DetectionFusion", "DetectorBackend", "ModelRouter", "build_detector"]
