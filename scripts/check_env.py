from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import DEFAULT_HAND_POSE_MODEL_PATH, DEFAULT_MODEL_PATH, DEFAULT_VIDEO_PATH


# 环境自检项：交付 demo 前确认模型、视频和关键依赖是否齐全。
REQUIRED_FILES = {
    "YOLO 训练权重": DEFAULT_MODEL_PATH,
    "手部骨骼模型": DEFAULT_HAND_POSE_MODEL_PATH,
    "默认测试视频": DEFAULT_VIDEO_PATH,
}

REQUIRED_PACKAGES = ["cv2", "numpy", "ultralytics", "mediapipe"]


def check_files() -> list[str]:
    """检查项目运行所需的模型和视频文件。"""

    errors: list[str] = []
    for label, path in REQUIRED_FILES.items():
        if not Path(path).exists():
            errors.append(f"缺少{label}: {path}")
    return errors


def check_packages() -> list[str]:
    """检查 Python 关键依赖是否可导入。"""

    errors: list[str] = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError as exc:
            errors.append(f"缺少依赖 {package}: {exc}")
    return errors


def collect_versions() -> dict[str, str]:
    """返回关键运行环境版本，方便软工记录问题现场。"""

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for package, module_name in {
        "opencv": "cv2",
        "numpy": "numpy",
        "ultralytics": "ultralytics",
        "mediapipe": "mediapipe",
    }.items():
        try:
            module = importlib.import_module(module_name)
            versions[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[package] = "not installed"
    return versions


def main() -> None:
    """命令行入口：输出环境自检结果。"""

    errors = check_files() + check_packages()
    versions = collect_versions()

    print("===== SOP Demo 环境自检 =====")
    for name, version in versions.items():
        print(f"{name}: {version}")

    if errors:
        print("\n自检结果: FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\n自检结果: OK")


if __name__ == "__main__":
    main()
