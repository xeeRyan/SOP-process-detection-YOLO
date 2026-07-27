from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.project import load_sop_project
from scripts.project.storage import read_json_object, resolve_within, utc_now, write_json_atomic, write_text_atomic
from scripts.utils import ensure_dir


def import_yolo_annotations(
    project_dir: str | Path,
    labels_dir: str | Path,
    *,
    overwrite: bool = True,
    mark_missing_as_empty: bool = False,
) -> dict[str, Any]:
    """将外部工具生成的 YOLO txt 标签导入 SOP 项目。"""

    project = load_sop_project(project_dir)
    labels_root = Path(labels_dir).expanduser().resolve()
    if not labels_root.is_dir():
        raise FileNotFoundError(f"未找到外部标签目录: {labels_root}")

    manifest_path = project.root / "frames_manifest.json"
    manifest = _read_manifest(manifest_path)
    frames_by_stem = _index_manifest_frames(manifest)
    label_files = sorted(labels_root.rglob("*.txt"))
    labels_by_stem = _index_label_files(label_files)
    class_count = len(project.project["classes"])

    imported: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    skipped_existing: list[dict[str, Any]] = []
    missing_labels: list[dict[str, Any]] = []
    duplicate_labels = [
        {"image_stem": stem, "label_paths": [str(path) for path in paths]}
        for stem, paths in labels_by_stem.items()
        if len(paths) > 1
    ]
    unmatched_labels = [
        str(paths[0])
        for stem, paths in labels_by_stem.items()
        if stem not in frames_by_stem and len(paths) == 1
    ]

    for stem, frame_record in frames_by_stem.items():
        candidates = labels_by_stem.get(stem, [])
        if len(candidates) > 1:
            invalid.append({"image_stem": stem, "reason": "存在多个同名标签文件"})
            continue
        if not candidates:
            if mark_missing_as_empty:
                target = resolve_within(project.root, frame_record["annotation_path"], field="annotation_path")
                if target.exists() and not overwrite:
                    skipped_existing.append({"image_stem": stem, "annotation_path": str(target)})
                else:
                    ensure_dir(target.parent)
                    write_text_atomic(target, "")
                    frame_record["annotation_status"] = "reviewed_empty"
                    frame_record["annotation_count"] = 0
                    imported.append(
                        {
                            "image_stem": stem,
                            "annotation_path": frame_record["annotation_path"],
                            "annotation_count": 0,
                            "status": "reviewed_empty",
                        }
                    )
            else:
                missing_labels.append(
                    {"image_stem": stem, "image_path": frame_record["image_path"]}
                )
            continue

        label_path = candidates[0]
        validation = _validate_yolo_file(label_path, class_count)
        if validation["errors"]:
            invalid.append(
                {
                    "image_stem": stem,
                    "label_path": str(label_path),
                    "errors": validation["errors"],
                }
            )
            continue

        target = resolve_within(project.root, frame_record["annotation_path"], field="annotation_path")
        if target.exists() and not overwrite:
            skipped_existing.append({"image_stem": stem, "annotation_path": str(target)})
            continue
        ensure_dir(target.parent)
        normalized_text = "\n".join(validation["normalized_lines"])
        if normalized_text:
            normalized_text += "\n"
        write_text_atomic(target, normalized_text)
        annotation_count = len(validation["normalized_lines"])
        status = "labeled" if annotation_count else "reviewed_empty"
        frame_record["annotation_status"] = status
        frame_record["annotation_count"] = annotation_count
        imported.append(
            {
                "image_stem": stem,
                "source_label_path": str(label_path),
                "annotation_path": frame_record["annotation_path"],
                "annotation_count": annotation_count,
                "status": status,
            }
        )

    manifest["updated_at"] = utc_now()
    write_json_atomic(manifest_path, manifest)
    report = {
        "schema_version": "1.0",
        "project_id": project.project_id,
        "project_dir": str(project.root),
        "labels_dir": str(labels_root),
        "imported_at": utc_now(),
        "settings": {
            "overwrite": overwrite,
            "mark_missing_as_empty": mark_missing_as_empty,
        },
        "summary": {
            "frame_count": len(frames_by_stem),
            "external_label_file_count": len(label_files),
            "imported_count": len(imported),
            "labeled_count": sum(item["status"] == "labeled" for item in imported),
            "empty_count": sum(item["status"] == "reviewed_empty" for item in imported),
            "missing_label_count": len(missing_labels),
            "invalid_count": len(invalid),
            "unmatched_label_count": len(unmatched_labels),
            "duplicate_label_count": len(duplicate_labels),
            "skipped_existing_count": len(skipped_existing),
        },
        "imported": imported,
        "missing_labels": missing_labels,
        "invalid": invalid,
        "unmatched_labels": unmatched_labels,
        "duplicate_labels": duplicate_labels,
        "skipped_existing": skipped_existing,
    }
    report_path = project.root / "annotation_import_report.json"
    report["report_path"] = str(report_path)
    report["manifest_path"] = str(manifest_path)
    write_json_atomic(report_path, report)
    return report


def _validate_yolo_file(path: Path, class_count: int) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    normalized_lines: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        return {"errors": [{"line": 0, "message": f"标签不是 UTF-8 文本: {exc}"}], "normalized_lines": []}

    for line_number, raw_line in enumerate(lines, start=1):
        text = raw_line.strip()
        if not text:
            continue
        fields = text.split()
        if len(fields) != 5:
            errors.append({"line": line_number, "message": "YOLO 标签每行必须包含 5 个字段"})
            continue
        try:
            class_id = int(fields[0])
            center_x, center_y, width, height = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append({"line": line_number, "message": "类别和坐标必须是数值"})
            continue
        if not 0 <= class_id < class_count:
            errors.append(
                {"line": line_number, "message": f"类别 id {class_id} 超出项目范围 0..{class_count - 1}"}
            )
            continue
        coordinates = (center_x, center_y, width, height)
        if not all(0 <= value <= 1 for value in coordinates):
            errors.append({"line": line_number, "message": "归一化坐标必须位于 0 到 1"})
            continue
        if width <= 0 or height <= 0:
            errors.append({"line": line_number, "message": "标注框宽高必须大于 0"})
            continue
        if center_x - width / 2 < 0 or center_x + width / 2 > 1:
            errors.append({"line": line_number, "message": "标注框横向范围超出图片"})
            continue
        if center_y - height / 2 < 0 or center_y + height / 2 > 1:
            errors.append({"line": line_number, "message": "标注框纵向范围超出图片"})
            continue
        normalized_lines.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
        )
    return {"errors": errors, "normalized_lines": normalized_lines}


def _index_manifest_frames(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for video in manifest["videos"].values():
        for frame in video.get("frames", []):
            stem = Path(frame["image_path"]).stem
            if stem in result:
                duplicates.add(stem)
            result[stem] = frame
    if duplicates:
        raise ValueError(f"抽帧清单包含重复图片名: {sorted(duplicates)}")
    return result


def _index_label_files(paths: list[Path]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for path in paths:
        result.setdefault(path.stem, []).append(path)
    return result


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"项目尚未生成抽帧清单: {path}")
    data = read_json_object(path, description="抽帧清单")
    if not isinstance(data, dict) or not isinstance(data.get("videos"), dict):
        raise ValueError(f"抽帧清单格式错误: {path}")
    return data
