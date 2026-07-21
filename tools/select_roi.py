from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import ROOT


DEFAULT_IMAGE_PATH = ROOT / "datasets" / "sop" / "raw_frames" / "frame_000050.jpg"
DEFAULT_SCALE = 0.4


# 交互式 ROI 选择工具，用于生成 config.py 中的三个业务区域。
def select_roi(image_path: str | Path = DEFAULT_IMAGE_PATH, scale: float = DEFAULT_SCALE) -> None:
    """交互式选择 SOP 判定 ROI。

    输入：
    - image_path: 用于选区的参考图片。
    - scale: 预览窗口缩放比例。

    输出：
    - 在终端打印 WORK_ROI、SCREW_BIN_ROI、TOOL_HOME_ROI，可复制到 config.py。
    """

    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"图片读取失败: {image_path}")
    if scale <= 0:
        raise ValueError("scale 必须大于 0")

    display = image
    if scale != 1.0:
        display = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    print("操作说明:")
    print(f"原图尺寸: {image.shape[1]}x{image.shape[0]}")
    print(f"显示缩放: {scale}")
    print("1. 先框选装配区 WORK_ROI，按 Enter/Space 确认。")
    print("2. 再框选螺丝盘区 SCREW_BIN_ROI，按 Enter/Space 确认。")
    print("3. 最后框选工具原位区 TOOL_HOME_ROI，按 Enter/Space 确认。")
    print("4. 框选错误时按 c 取消当前选择。")

    work = cv2.selectROI("Select WORK_ROI", display, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select WORK_ROI")
    screw_bin = cv2.selectROI("Select SCREW_BIN_ROI", display, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select SCREW_BIN_ROI")
    tool_home = cv2.selectROI("Select TOOL_HOME_ROI", display, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select TOOL_HOME_ROI")

    # selectROI 在缩放图上取值，需要还原到原始视频坐标。
    work_roi = _scale_roi(_xywh_to_xyxy(work), scale)
    screw_bin_roi = _scale_roi(_xywh_to_xyxy(screw_bin), scale)
    tool_home_roi = _scale_roi(_xywh_to_xyxy(tool_home), scale)

    print()
    print("复制到 scripts/config.py:")
    print(f"WORK_ROI = {work_roi}")
    print(f"SCREW_BIN_ROI = {screw_bin_roi}")
    print(f"TOOL_HOME_ROI = {tool_home_roi}")


# OpenCV selectROI 返回 x/y/w/h，业务配置使用 x1/y1/x2/y2。
def _xywh_to_xyxy(rect: tuple[int, int, int, int]) -> list[int]:
    """将 OpenCV selectROI 的 x/y/w/h 转为 x1/y1/x2/y2。"""

    x, y, w, h = [int(value) for value in rect]
    return [x, y, x + w, y + h]


# 将预览图坐标还原成原图坐标。
def _scale_roi(roi: list[int], scale: float) -> list[int]:
    """将预览图坐标还原到原始图片坐标。"""

    return [int(round(value / scale)) for value in roi]


# select_roi.py 的命令行参数。
def build_parser() -> argparse.ArgumentParser:
    """构建 ROI 选择脚本的命令行参数。"""

    parser = argparse.ArgumentParser(description="交互式选择 SOP 判定 ROI")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE_PATH), help="用于选择 ROI 的图片")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="显示缩放比例，默认 0.4")
    return parser


# 命令行入口，打开 ROI 选择窗口。
def main() -> None:
    """命令行入口：打开选区窗口并打印 ROI 配置。"""

    args = build_parser().parse_args()
    select_roi(args.image, args.scale)


if __name__ == "__main__":
    main()
