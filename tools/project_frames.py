from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any, Sequence

import cv2

from scripts.project.schema_validator import validate_project
from scripts.project.storage import read_json_object, utc_now, write_json_atomic
from scripts.utils import ensure_dir


SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def extract_project_videos(
    project_dir: str | Path,
    video_paths: Sequence[str | Path],
    frames_per_second: float = 2.0,
    frame_interval: int | None = None,
    max_frames_per_video: int = 0,
    copy_videos: bool = True,
    jpeg_quality: int = 95,
) -> dict[str, Any]:
    """导入一个或多个流程视频并抽取项目级标注图片。"""

    # Frame extraction is a data-preparation operation.  It only needs the
    # project's identity and class definitions; ROI/workflow may still be
    # empty while the project is in draft state.
    project_root = Path(project_dir).expanduser().resolve()
    project = read_json_object(project_root / "project.json", description="SOP 项目配置文件")
    validate_project(project)
    project_id = str(project["project_id"])
    if not video_paths:
        raise ValueError("video_paths 至少需要包含一个视频")
    if frames_per_second <= 0:
        raise ValueError("frames_per_second 必须大于 0")
    if frame_interval is not None and frame_interval <= 0:
        raise ValueError("frame_interval 必须是正整数")
    if max_frames_per_video < 0:
        raise ValueError("max_frames_per_video 不能小于 0")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality 必须位于 1 到 100")

    source_dir = ensure_dir(project_root / "source_videos")
    frames_root = ensure_dir(project_root / "frames")
    ensure_dir(project_root / "annotations")
    ensure_dir(project_root / "dataset")
    manifest_path = project_root / "frames_manifest.json"
    manifest = _read_manifest(manifest_path, project_id)

    results: list[dict[str, Any]] = []
    for raw_video_path in video_paths:
        source_path = Path(raw_video_path).expanduser().resolve()
        _validate_video_path(source_path)
        video_id = _build_video_id(source_path)
        stored_path = source_dir / f"{video_id}{source_path.suffix.lower()}"
        if copy_videos:
            if source_path != stored_path.resolve() and (
                not stored_path.exists() or stored_path.stat().st_size != source_path.stat().st_size
            ):
                shutil.copy2(source_path, stored_path)
            input_path = stored_path
        else:
            input_path = source_path

        video_result = _extract_one_video(
            video_id=video_id,
            source_path=source_path,
            stored_path=stored_path if copy_videos else None,
            input_path=input_path,
            output_dir=ensure_dir(frames_root / video_id),
            frames_per_second=frames_per_second,
            frame_interval=frame_interval,
            max_frames=max_frames_per_video,
            jpeg_quality=jpeg_quality,
            project_root=project_root,
        )
        results.append(video_result)
        manifest["videos"][video_id] = video_result

    manifest["updated_at"] = utc_now()
    write_json_atomic(manifest_path, manifest)
    return {
        "project_id": project_id,
        "project_dir": str(project_root),
        "manifest_path": str(manifest_path),
        "video_count": len(results),
        "extracted_frame_count": sum(item["extracted_frame_count"] for item in results),
        "videos": results,
    }


def _extract_one_video(
    *,
    video_id: str,
    source_path: Path,
    stored_path: Path | None,
    input_path: Path,
    output_dir: Path,
    frames_per_second: float,
    frame_interval: int | None,
    max_frames: int,
    jpeg_quality: int,
    project_root: Path,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"视频打开失败: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = frame_interval or max(1, int(round(fps / frames_per_second)))
    frames: list[dict[str, Any]] = []
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % interval == 0:
                image_name = f"{video_id}_{frame_index:08d}.jpg"
                image_path = output_dir / image_name
                written = cv2.imwrite(
                    str(image_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                )
                if not written:
                    raise RuntimeError(f"抽帧图片写入失败: {image_path}")
                frames.append(
                    {
                        "frame_id": f"{video_id}:{frame_index}",
                        "frame_index": frame_index,
                        "time_sec": round(frame_index / fps, 6),
                        "image_path": _relative_path(image_path, project_root),
                        "annotation_path": f"annotations/{video_id}/{image_path.stem}.txt",
                        "annotation_status": "unlabeled",
                    }
                )
                if max_frames > 0 and len(frames) >= max_frames:
                    break
            frame_index += 1
    finally:
        cap.release()

    return {
        "video_id": video_id,
        "source_path": str(source_path),
        "stored_path": _relative_path(stored_path, project_root) if stored_path else None,
        "file_name": source_path.name,
        "fps": round(fps, 6),
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 6) if total_frames > 0 else None,
        "sampling": {
            "frame_interval": interval,
            "requested_frames_per_second": frames_per_second,
            "effective_frames_per_second": round(fps / interval, 6),
            "max_frames": max_frames,
        },
        "frames_dir": _relative_path(output_dir, project_root),
        "extracted_frame_count": len(frames),
        "frames": frames,
        "imported_at": utc_now(),
    }


def _validate_video_path(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"未找到流程视频: {path}")
    if path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        raise ValueError(f"不支持的视频格式: {path.suffix}")


def _build_video_id(path: Path) -> str:
    stat = path.stat()
    fingerprint = f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    suffix = hashlib.sha1(fingerprint).hexdigest()[:10]
    safe_stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in path.stem).strip("_")
    return f"{safe_stem or 'video'}_{suffix}"


def _read_manifest(path: Path, project_id: str) -> dict[str, Any]:
    if path.is_file():
        data = read_json_object(path, description="抽帧清单")
        if not isinstance(data, dict) or not isinstance(data.get("videos"), dict):
            raise ValueError(f"抽帧清单格式错误: {path}")
        return data
    return {
        "schema_version": "1.0",
        "project_id": project_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "videos": {},
    }


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
