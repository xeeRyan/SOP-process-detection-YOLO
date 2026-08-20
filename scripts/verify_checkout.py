"""Verify that a fresh Git/LFS checkout contains the runnable project assets."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _is_materialized(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as stream:
        return not stream.read(len(LFS_POINTER_PREFIX)).startswith(LFS_POINTER_PREFIX)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    missing: list[str] = []
    required = [
        ROOT / "models" / "hand_landmarker.task",
        ROOT / "models" / "yolo26n.pt",
        ROOT / "models" / "yolo26s.pt",
        ROOT / "videos" / "smoke_60f.mp4",
    ]

    projects_root = ROOT / "projects"
    for project_dir in sorted(path for path in projects_root.iterdir() if path.is_dir()):
        project_file = project_dir / "project.json"
        if not project_file.is_file():
            continue
        project = _load_json(project_file)
        active_model = project.get("active_model")
        if active_model:
            required.append(project_dir / str(active_model))
        for profile in (project.get("model_profiles") or {}).values():
            if isinstance(profile, dict) and profile.get("path"):
                required.append(project_dir / str(profile["path"]))

        source_dir = project_dir / "source_videos"
        if not source_dir.is_dir() or not any(source_dir.iterdir()):
            missing.append(str(source_dir.relative_to(ROOT)) + " (no source video)")
        annotation_dir = project_dir / "annotations"
        if not annotation_dir.is_dir() or not any(annotation_dir.rglob("*.txt")):
            missing.append(str(annotation_dir.relative_to(ROOT)) + " (no YOLO labels)")

    for path in required:
        if not _is_materialized(path):
            missing.append(str(path.relative_to(ROOT)))

    if missing:
        print("Checkout verification failed. Run `git lfs pull` and check:")
        for value in missing:
            print(f"  - {value}")
        return 1

    print(f"Checkout verification passed: {len(required)} required assets are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
