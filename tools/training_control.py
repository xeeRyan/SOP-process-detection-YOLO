"""基于原子JSON文件的训练进度和安全停止控制。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.project.storage import read_json_object, utc_now, write_json_atomic


STATUS_FILE_NAME = "training_status.json"
STOP_FILE_NAME = "stoptrain.json"


def initialize_training_control(
    control_dir: str | Path,
    *,
    task_id: str,
    model_id: str | None,
    model_version: str | None,
    total_epochs: int,
) -> tuple[Path, Path]:
    root = Path(control_dir)
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / STATUS_FILE_NAME
    stop_path = root / STOP_FILE_NAME
    write_json_atomic(stop_path, {"task_id": task_id, "is_stop": False})
    write_training_status(
        status_path,
        task_id=task_id,
        status="running",
        current_epoch=0,
        total_epochs=total_epochs,
        model_id=model_id,
        model_version=model_version,
        message=f"准备训练，共 {total_epochs} 轮",
    )
    return status_path, stop_path


def stop_requested(stop_path: str | Path, task_id: str) -> bool:
    try:
        control = read_json_object(stop_path, description="训练停止控制文件")
    except (FileNotFoundError, ValueError, OSError):
        return False
    return control.get("task_id") == task_id and control.get("is_stop") is True


def write_training_status(
    status_path: str | Path,
    *,
    task_id: str,
    status: str,
    current_epoch: int,
    total_epochs: int,
    model_id: str | None,
    model_version: str | None,
    message: str,
    latest_metrics: dict[str, Any] | None = None,
    stop_requested_value: bool = False,
    error: str | None = None,
) -> None:
    total = max(int(total_epochs), 0)
    current = max(int(current_epoch), 0)
    progress = round(current * 100.0 / total, 2) if total else 0.0
    write_json_atomic(
        status_path,
        {
            "schema_version": "1.0",
            "task_id": task_id,
            "status": status,
            "current_epoch": current,
            "total_epochs": total,
            "progress_percent": progress,
            "stop_requested": bool(stop_requested_value),
            "model_id": model_id,
            "model_version": model_version,
            "latest_metrics": latest_metrics or {},
            "message": message,
            "error": error,
            "updated_at": utc_now(),
        },
    )
