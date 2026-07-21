from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import DEFAULT_VIDEO_PATH, ROOT
from scripts.utils import ensure_dir


DEFAULT_OUTPUT_DIR = ROOT / "datasets" / "sop" / "raw_frames"


# 数据集准备入口：从视频中按固定频率抽取标注图片。
def extract_frames(
    video_path: str | Path = DEFAULT_VIDEO_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    frames_per_second: float = 2.0,
    max_frames: int = 300,
    image_prefix: str = "frame",
) -> int:
    """从视频中抽取训练图片。

    输入：
    - video_path: 原始视频路径。
    - output_dir: 抽帧图片输出目录。
    - frames_per_second: 每秒抽取的图片数量。
    - max_frames: 最大保存数量，0 表示不限制。

    返回：
    - 实际保存的图片数量。
    """

    video_path = Path(video_path)
    output_dir = ensure_dir(Path(output_dir))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"视频打开失败: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    interval = max(1, int(round(fps / frames_per_second)))

    saved_count = 0
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_id % interval == 0:
            # 文件名使用连续编号，方便后续标注工具排序。
            output_path = output_dir / f"{image_prefix}_{saved_count:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
            saved_count += 1

            if max_frames > 0 and saved_count >= max_frames:
                break

        frame_id += 1

    cap.release()
    return saved_count


# extract_frames.py 的命令行参数。
def build_parser() -> argparse.ArgumentParser:
    """构建抽帧脚本的命令行参数。"""

    parser = argparse.ArgumentParser(description="从视频抽取 YOLO 标注图片")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO_PATH), help="输入视频路径，默认 videos/sk.mp4")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="输出图片目录")
    parser.add_argument("--fps", type=float, default=2.0, help="每秒抽取多少张图片")
    parser.add_argument("--max-frames", type=int, default=300, help="最多保存多少张，0 表示不限制")
    parser.add_argument("--prefix", default="frame", help="输出图片文件名前缀")
    return parser


# 命令行入口，输出保存图片数量和目录。
def main() -> None:
    """命令行入口：抽取标注图片并打印保存数量。"""

    args = build_parser().parse_args()
    count = extract_frames(
        video_path=args.video,
        output_dir=args.output,
        frames_per_second=args.fps,
        max_frames=args.max_frames,
        image_prefix=args.prefix,
    )
    print(f"抽帧完成: {count} 张")
    print(f"输出目录: {Path(args.output)}")


if __name__ == "__main__":
    main()
