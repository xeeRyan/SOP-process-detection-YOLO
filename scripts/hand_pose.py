from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# 手部关键点连线定义。关键点编号采用 MediaPipe 标准 21 点手模型。
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


# 单只手的 21 点骨骼结果，坐标已经换算到原始视频帧像素系。
@dataclass(frozen=True)
class HandLandmarks:
    """手部骨骼检测输出对象。

    字段：
    - handedness: 左手/右手分类结果。
    - score: 左右手分类置信度。
    - points: 21 个关键点，格式为 (x, y, z)，其中 x/y 为原始视频帧像素坐标。
    """

    handedness: str
    score: float
    points: list[tuple[int, int, float]]

    def to_dict(self) -> dict:
        """转换为可写入 JSON 或接口返回的字典。"""
        return {
            "handedness": self.handedness,
            "score": round(self.score, 4),
            "points": [
                {"id": index, "x": x, "y": y, "z": round(z, 6)}
                for index, (x, y, z) in enumerate(self.points)
            ],
        }


# 手部骨骼检测器，供视频主流程复用。
class HandPoseDetector:
    """手部骨骼检测服务封装。

    对接用途：
    - 输入 OpenCV 视频帧和对应时间戳。
    - 输出每只手的 21 点骨骼坐标。
    - 输出结构可直接用于画面叠加、JSON 返回或后续手势规则判断。
    """

    def __init__(
        self,
        model_path: str | Path,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"未找到手部骨骼模型: {model_path}. "
                "请将 hand_landmarker.task 放到 models 目录。"
            )

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.model_path = model_path
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray, timestamp_ms: int) -> list[HandLandmarks]:
        """检测单帧中的手部骨骼。

        参数：
        - frame: OpenCV BGR 图像。
        - timestamp_ms: 当前帧时间戳，单位毫秒。

        返回：
        - HandLandmarks 列表，每个元素对应一只手。
        """

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        height, width = frame.shape[:2]

        hands: list[HandLandmarks] = []
        for index, landmarks in enumerate(result.hand_landmarks):
            handedness = "Unknown"
            score = 0.0

            # handedness 表示左手/右手分类，软件端可用于展示或动作规则。
            if index < len(result.handedness) and result.handedness[index]:
                category = result.handedness[index][0]
                handedness = category.category_name or "Unknown"
                score = float(category.score)

            # MediaPipe 输出归一化坐标，这里统一转成像素坐标。
            points = [
                (int(landmark.x * width), int(landmark.y * height), float(landmark.z))
                for landmark in landmarks
            ]
            hands.append(HandLandmarks(handedness=handedness, score=score, points=points))

        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> HandPoseDetector:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def build_hand_pose_detector(model_path: str | Path) -> HandPoseDetector:
    """创建手部骨骼检测器，供主流程和单独预览脚本复用。"""
    return HandPoseDetector(model_path)
