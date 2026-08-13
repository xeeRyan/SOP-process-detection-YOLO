"""检查运行依赖、默认模型和示例输入是否齐备。"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import DEFAULT_HAND_POSE_MODEL_PATH
from scripts.project import load_sop_project


REQUIRED_PACKAGES = ["cv2", "numpy", "ultralytics", "mediapipe"]


def check_files(project_dir: str | Path, video_path: str | Path | None = None) -> list[str]:
    """检查指定 SOP 项目及可选输入视频所需的文件。"""

    errors: list[str] = []
    project = load_sop_project(project_dir)
    required_files = {
        "项目活动模型": project.active_model_path,
        "手部骨骼模型": DEFAULT_HAND_POSE_MODEL_PATH,
    }
    if video_path not in (None, ""):
        required_files["输入视频"] = Path(video_path)
    for label, path in required_files.items():
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

    import argparse

    parser = argparse.ArgumentParser(description="检查 SOP 项目运行环境")
    parser.add_argument("--project", required=True, help="SOP 项目目录")
    parser.add_argument("--video", help="待检测视频路径")
    args = parser.parse_args()

    errors = check_files(args.project, args.video) + check_packages()
    versions = collect_versions()

    print("===== SOP 项目环境自检 =====")
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
