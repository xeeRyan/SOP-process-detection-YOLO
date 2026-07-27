from __future__ import annotations

import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.project import load_sop_project
from scripts.project.storage import read_json_object, resolve_within, utc_now, write_json_atomic
from scripts.utils import ensure_dir


SPLIT_NAMES = ("train", "val", "test")
ELIGIBLE_STATUSES = {"labeled", "reviewed_empty"}


def build_yolo_dataset(
    project_dir: str | Path,
    *,
    dataset_name: str | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    require_all_annotated: bool = True,
    require_all_splits: bool = True,
) -> dict[str, Any]:
    """按视频分组构建无相邻帧泄漏的 YOLO 数据集。"""

    project = load_sop_project(project_dir)
    ratios = _validate_ratios(train_ratio, val_ratio, test_ratio)
    manifest_path = project.root / "frames_manifest.json"
    manifest = _read_manifest(manifest_path)
    groups, excluded = _collect_video_groups(project.root, manifest)
    if not groups:
        raise ValueError("没有可用于构建数据集的已标注帧")
    if require_all_annotated and excluded:
        raise ValueError(
            f"仍有 {len(excluded)} 张图片未完成有效标注，请先查看 frames_manifest.json"
        )

    active_splits = [name for name, ratio in zip(SPLIT_NAMES, ratios) if ratio > 0]
    if require_all_splits and len(groups) < len(active_splits):
        raise ValueError(
            f"当前只有 {len(groups)} 个已标注视频，无法生成 {len(active_splits)} 个互斥集合；"
            "请增加独立视频，或将不需要集合的比例设为 0"
        )

    assignments = _assign_video_groups(groups, ratios, seed, require_all_splits)
    name = dataset_name or datetime.now().strftime("dataset_%Y%m%d_%H%M%S")
    if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in name):
        raise ValueError("dataset_name 只能包含字母、数字、横线和下划线")
    output_dir = project.root / "dataset" / name
    if output_dir.exists():
        raise FileExistsError(f"数据集目录已存在，请使用新的 dataset_name: {output_dir}")

    records: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        ensure_dir(output_dir / "images" / split)
        ensure_dir(output_dir / "labels" / split)
        for group in assignments[split]:
            for frame in group["frames"]:
                source_image = project.root / frame["image_path"]
                source_label = project.root / frame["annotation_path"]
                target_image = output_dir / "images" / split / source_image.name
                target_label = output_dir / "labels" / split / source_label.name
                shutil.copy2(source_image, target_image)
                shutil.copy2(source_label, target_label)
                records.append(
                    {
                        "video_id": group["video_id"],
                        "frame_id": frame.get("frame_id"),
                        "split": split,
                        "source_image": frame["image_path"],
                        "source_annotation": frame["annotation_path"],
                        "image": target_image.relative_to(output_dir).as_posix(),
                        "label": target_label.relative_to(output_dir).as_posix(),
                        "annotation_status": frame["annotation_status"],
                    }
                )

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(_build_data_yaml(output_dir, project.project["classes"]), encoding="utf-8")
    class_names = {int(item["id"]): str(item["name"]) for item in project.project["classes"]}
    summary = {}
    for split in SPLIT_NAMES:
        split_records = [record for record in records if record["split"] == split]
        class_counts = {name: 0 for name in class_names.values()}
        for record in split_records:
            label_path = output_dir / record["label"]
            for line in label_path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    class_counts[class_names[int(line.split()[0])]] += 1
        summary[split] = {
            "video_count": len(assignments[split]),
            "frame_count": len(split_records),
            "positive_frame_count": sum(record["annotation_status"] == "labeled" for record in split_records),
            "empty_frame_count": sum(record["annotation_status"] == "reviewed_empty" for record in split_records),
            "class_counts": class_counts,
            "video_ids": [group["video_id"] for group in assignments[split]],
        }
    dataset_manifest = {
        "schema_version": "1.0",
        "project_id": project.project_id,
        "dataset_name": name,
        "created_at": utc_now(),
        "seed": seed,
        "split_ratios": dict(zip(SPLIT_NAMES, ratios)),
        "grouping": "video_id",
        "summary": summary,
        "excluded_frames": excluded,
        "records": records,
    }
    dataset_manifest_path = output_dir / "dataset_manifest.json"
    write_json_atomic(dataset_manifest_path, dataset_manifest)
    return {
        "project_id": project.project_id,
        "dataset_name": name,
        "dataset_dir": str(output_dir),
        "data_yaml": str(yaml_path),
        "dataset_manifest": str(dataset_manifest_path),
        "summary": summary,
        "excluded_frame_count": len(excluded),
    }


def _collect_video_groups(project_root: Path, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for video_id, video in manifest["videos"].items():
        eligible: list[dict[str, Any]] = []
        for frame in video.get("frames", []):
            status = frame.get("annotation_status", "unlabeled")
            if status not in ELIGIBLE_STATUSES:
                excluded.append(
                    {"video_id": video_id, "frame_id": frame.get("frame_id"), "reason": status}
                )
                continue
            image_path = resolve_within(project_root, frame["image_path"], field="image_path")
            label_path = resolve_within(project_root, frame["annotation_path"], field="annotation_path")
            if not image_path.is_file() or not label_path.is_file():
                excluded.append(
                    {
                        "video_id": video_id,
                        "frame_id": frame.get("frame_id"),
                        "reason": "image_or_annotation_missing",
                    }
                )
                continue
            eligible.append(frame)
        if eligible:
            groups.append({"video_id": video_id, "frames": eligible})
    return groups, excluded


def _assign_video_groups(
    groups: list[dict[str, Any]],
    ratios: tuple[float, float, float],
    seed: int,
    require_all_splits: bool,
) -> dict[str, list[dict[str, Any]]]:
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    counts = _allocate_group_counts(len(shuffled), ratios, require_all_splits)
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    offset = 0
    for split, count in zip(SPLIT_NAMES, counts):
        result[split] = shuffled[offset : offset + count]
        offset += count
    return result


def _allocate_group_counts(total: int, ratios: tuple[float, float, float], require_all: bool) -> tuple[int, int, int]:
    counts = [0, 0, 0]
    active = [index for index, ratio in enumerate(ratios) if ratio > 0]
    remaining = total
    if require_all:
        for index in active:
            counts[index] = 1
            remaining -= 1
    if remaining > 0:
        raw = [remaining * ratio / sum(ratios) for ratio in ratios]
        floors = [int(value) for value in raw]
        for index, value in enumerate(floors):
            counts[index] += value
        left = remaining - sum(floors)
        order = sorted(range(3), key=lambda index: raw[index] - floors[index], reverse=True)
        for index in order[:left]:
            counts[index] += 1
    return tuple(counts)  # type: ignore[return-value]


def _validate_ratios(train: float, val: float, test: float) -> tuple[float, float, float]:
    ratios = (float(train), float(val), float(test))
    if any(value < 0 for value in ratios):
        raise ValueError("数据集划分比例不能小于 0")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("至少一个数据集划分比例必须大于 0")
    if train <= 0:
        raise ValueError("train_ratio 必须大于 0")
    return tuple(value / total for value in ratios)  # type: ignore[return-value]


def _build_data_yaml(output_dir: Path, classes: list[dict[str, Any]]) -> str:
    lines = [
        f"path: {json.dumps(output_dir.as_posix(), ensure_ascii=False)}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for item in sorted(classes, key=lambda value: value["id"]):
        lines.append(f"  {item['id']}: {json.dumps(item['name'], ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"项目尚未生成抽帧清单: {path}")
    data = read_json_object(path, description="抽帧清单")
    if not isinstance(data, dict) or not isinstance(data.get("videos"), dict):
        raise ValueError(f"抽帧清单格式错误: {path}")
    return data
