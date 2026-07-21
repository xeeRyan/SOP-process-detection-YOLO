from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

from scripts.config import BOX_COLOR, NG_COLOR, OK_COLOR, ROI_COLOR, SCREW_BIN_COLOR, TEXT_COLOR, TOOL_HOME_COLOR
from scripts.detector import Detection
from scripts.sop_logic import SOPStateMachine


# 手部关键点连线定义。放在可视化模块中，避免关闭手部骨骼时仍强制导入 MediaPipe。
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


# 输出视频中的 YOLO 检测框图层。
def draw_detections(frame: np.ndarray, detections: list[Detection]) -> None:
    """在输出视频帧上叠加目标检测结果。"""

    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection.bbox]
        label = f"{detection.class_name} {detection.conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            BOX_COLOR,
            2,
        )


# 输出视频中的 ROI 图层。
def draw_roi(frame: np.ndarray, roi: Sequence[int], label: str = "WORK ROI") -> None:
    """在输出视频帧上叠加业务 ROI。"""

    x1, y1, x2, y2 = roi
    if label == "SCREW BIN":
        color = SCREW_BIN_COLOR
    elif label == "TOOL HOME":
        color = TOOL_HOME_COLOR
    else:
        color = ROI_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


# 输出视频左上角的 SOP 状态图层。
def draw_sop_status(frame: np.ndarray, machine: SOPStateMachine) -> None:
    """在输出视频帧上叠加 SOP 状态。"""

    result_color = OK_COLOR if machine.final_result == "OK" else NG_COLOR
    cv2.putText(
        frame,
        f"State: {machine.state_text}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        TEXT_COLOR,
        3,
    )

    y = 85
    for step in machine.steps:
        color = OK_COLOR if step.status == "done" else TEXT_COLOR
        text = f"{step.step_id}. {step.key}: {step.status}"
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
        y += 40

    if machine.final_result == "OK" or machine.reason:
        result = machine.final_result if machine.final_result == "OK" else "NG"
        cv2.putText(frame, f"Result: {result}", (20, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, result_color, 3)


# 输出视频中的手部骨骼图层。
def draw_hand_landmarks(frame: np.ndarray, hands: list[Any]) -> None:
    """在输出视频帧上叠加手部骨骼结果。"""

    for hand in hands:
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, hand.points[start][:2], hand.points[end][:2], (0, 255, 0), 2)

        for index, point in enumerate(hand.points):
            cv2.circle(frame, point[:2], 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                str(index),
                (point[0] + 4, point[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        wrist = hand.points[0]
        cv2.putText(
            frame,
            f"{hand.handedness} {hand.score:.2f}",
            (wrist[0] + 8, wrist[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
