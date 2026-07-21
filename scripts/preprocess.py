from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# ????????????????????????????????
DEFAULT_PREPROCESS_ALPHA = 1.0
DEFAULT_PREPROCESS_BETA = 0.0
DEFAULT_PREPROCESS_GAMMA = None
DEFAULT_PREPROCESS_CLAHE_ENABLED = False
DEFAULT_PREPROCESS_CLAHE_CLIP_LIMIT = 2.0
DEFAULT_PREPROCESS_CLAHE_TILE_GRID_SIZE = 8
DEFAULT_PREPROCESS_SATURATION = 1.0


# OpenCV 图像增强流水线：按配置对每个视频帧执行亮度、Gamma、CLAHE 和饱和度增强。
class FramePreprocessor:
    """视频帧预处理服务。

    当前交付配置只开放 enhancement 参数；空配置会原样返回输入帧。
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._validate_config()
        self._gamma_lookup = self._build_gamma_lookup()

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """按 enhancement 配置执行单帧预处理并返回处理后的 BGR 图像。"""

        return self._apply_enhancement(frame.copy())

    def to_dict(self) -> dict[str, Any]:
        """返回可写入 result.json 和运行日志的预处理配置。"""

        return self.config

    def _apply_enhancement(self, frame: np.ndarray) -> np.ndarray:
        """执行亮度/对比度、Gamma、CLAHE 和饱和度增强。"""

        enhancement = self.config.get("enhancement", {})
        alpha = float(enhancement.get("alpha", DEFAULT_PREPROCESS_ALPHA))
        beta = float(enhancement.get("beta", DEFAULT_PREPROCESS_BETA))
        if alpha != DEFAULT_PREPROCESS_ALPHA or beta != DEFAULT_PREPROCESS_BETA:
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        if self._gamma_lookup is not None:
            frame = cv2.LUT(frame, self._gamma_lookup)

        clahe = enhancement.get("clahe", {})
        if clahe.get("enabled", DEFAULT_PREPROCESS_CLAHE_ENABLED):
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            tile_grid_size = int(clahe.get("tile_grid_size", DEFAULT_PREPROCESS_CLAHE_TILE_GRID_SIZE))
            operator = cv2.createCLAHE(
                clipLimit=float(clahe.get("clip_limit", DEFAULT_PREPROCESS_CLAHE_CLIP_LIMIT)),
                tileGridSize=(tile_grid_size, tile_grid_size),
            )
            frame = cv2.cvtColor(cv2.merge((operator.apply(l_channel), a_channel, b_channel)), cv2.COLOR_LAB2BGR)

        saturation = float(enhancement.get("saturation", DEFAULT_PREPROCESS_SATURATION))
        if saturation != DEFAULT_PREPROCESS_SATURATION:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * saturation, 0, 255).astype(np.uint8)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return frame

    def _validate_config(self) -> None:
        """校验当前开放的 enhancement 配置。"""

        enhancement = self.config.get("enhancement", {})
        if enhancement.get("gamma") is not None and float(enhancement["gamma"]) <= 0:
            raise ValueError("gamma 必须大于 0")

        alpha = float(enhancement.get("alpha", DEFAULT_PREPROCESS_ALPHA))
        if alpha <= 0:
            raise ValueError("alpha 必须大于 0")

        saturation = float(enhancement.get("saturation", DEFAULT_PREPROCESS_SATURATION))
        if saturation < 0:
            raise ValueError("saturation 不能小于 0")

        clahe = enhancement.get("clahe", {})
        if clahe.get("enabled", DEFAULT_PREPROCESS_CLAHE_ENABLED):
            if float(clahe.get("clip_limit", DEFAULT_PREPROCESS_CLAHE_CLIP_LIMIT)) <= 0:
                raise ValueError("clahe.clip_limit 必须大于 0")
            if int(clahe.get("tile_grid_size", DEFAULT_PREPROCESS_CLAHE_TILE_GRID_SIZE)) < 2:
                raise ValueError("clahe.tile_grid_size 必须大于等于 2")

    def _build_gamma_lookup(self) -> np.ndarray | None:
        """预先构建 Gamma 查找表，避免在视频逐帧处理中重复计算。"""

        gamma = self.config.get("enhancement", {}).get("gamma", DEFAULT_PREPROCESS_GAMMA)
        if gamma is None or float(gamma) == 1.0:
            return None
        values = np.arange(256, dtype=np.float32) / 255.0
        return np.clip(values ** (1.0 / float(gamma)) * 255, 0, 255).astype(np.uint8)
