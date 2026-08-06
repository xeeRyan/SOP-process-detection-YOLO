"""项目级模型训练、版本登记与模型清单生成。"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.project import load_sop_project
from scripts.project.storage import read_json_object, utc_now, validate_safe_name, write_json_atomic
from tools.training import AUGMENTATION_PARAM_NAMES, train_yolo_model


def train_project_model(
    project_dir: str | Path,
    dataset_name: str,
    model_id: str,
    *,
    model_version: str | None = None,
    base_model_path: str | Path | None = None,
    set_active: bool = True,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """使用指定项目数据集训练模型并登记模型版本。"""

    project = load_sop_project(project_dir)
    validate_safe_name(dataset_name, "dataset_name")
    validate_safe_name(model_id, "model_id")
    version = model_version or datetime.now().strftime("%Y%m%d_%H%M%S")
    validate_safe_name(version, "model_version", allow_dot=True)
    options = params or {}

    dataset_dir = project.root / "dataset" / dataset_name
    data_yaml = dataset_dir / "data.yaml"
    dataset_manifest_path = dataset_dir / "dataset_manifest.json"
    if not data_yaml.is_file() or not dataset_manifest_path.is_file():
        raise FileNotFoundError(f"未找到完整项目数据集: {dataset_dir}")
    dataset_manifest = read_json_object(dataset_manifest_path, description="数据集清单")
    if dataset_manifest.get("project_id") != project.project_id:
        raise ValueError("数据集清单 project_id 与当前 SOP 项目不一致")

    base_model = _resolve_base_model(project.root, project.project, base_model_path)
    if not base_model.is_file():
        raise FileNotFoundError(f"未找到训练基础模型: {base_model}")

    model_dir = project.root / "models" / model_id / version
    if model_dir.exists():
        raise FileExistsError(f"模型版本已经存在，请使用新版本号: {model_dir}")
    run_name = str(options.get("run_name") or f"{model_id}_{version.replace('.', '_')}")
    validate_safe_name(run_name, "run_name")
    best_model_path = model_dir / "best.pt"
    augmentations = {
        name: options[name]
        for name in AUGMENTATION_PARAM_NAMES
        if name in options and options[name] is not None
    }

    training_result = train_yolo_model(
        base_model_path=base_model,
        data_yaml_path=data_yaml,
        project_dir=project.root / "runs",
        run_name=run_name,
        epochs=int(options.get("epochs", 100)),
        imgsz=int(options.get("imgsz", 640)),
        batch=int(options.get("batch", 8)),
        workers=int(options.get("workers", 0)),
        device=options.get("device"),
        optimizer=str(options.get("optimizer", "auto")),
        amp=bool(options.get("amp", True)),
        copy_best_to=best_model_path,
        exist_ok=False,
        log_dir=project.root / "outputs" / "logs",
        training_control_dir=project.root / "outputs" / "training",
        training_task_id=f"{model_id}_{version}",
        training_model_id=model_id,
        training_model_version=version,
        **augmentations,
    )
    model_manifest = register_project_model(
        project_dir=project.root,
        dataset_name=dataset_name,
        model_id=model_id,
        model_version=version,
        base_model_path=base_model,
        training_result=training_result,
        set_active=set_active,
    )
    return {
        "project_id": project.project_id,
        "dataset_name": dataset_name,
        "model_id": model_id,
        "model_version": version,
        "model_dir": str(model_dir),
        "model_manifest": str(model_dir / "model_manifest.json"),
        "active_model_updated": set_active,
        "training": training_result,
        "registration": model_manifest,
    }


def register_project_model(
    project_dir: str | Path,
    dataset_name: str,
    model_id: str,
    model_version: str,
    base_model_path: str | Path,
    training_result: dict[str, Any],
    *,
    set_active: bool,
) -> dict[str, Any]:
    """登记已完成的训练结果；独立函数便于校验和恢复训练产物。"""

    project = load_sop_project(project_dir)
    model_dir = project.root / "models" / model_id / model_version
    best_model = Path(training_result.get("copied_best_model") or "")
    if not best_model.is_file():
        raise FileNotFoundError(f"训练未生成可登记的 best 模型: {best_model}")
    results_csv = Path(training_result["results_csv"]) if training_result.get("results_csv") else None
    manifest = {
        "schema_version": "1.0",
        "project_id": project.project_id,
        "model_id": model_id,
        "model_version": model_version,
        "created_at": utc_now(),
        "dataset_name": dataset_name,
        "dataset_manifest": f"dataset/{dataset_name}/dataset_manifest.json",
        "base_model": str(Path(base_model_path).resolve()),
        "artifacts": {
            "best_pt": _relative(best_model, project.root),
            "last_pt": _relative_optional(training_result.get("last_model"), project.root),
            "torchscript": _relative_optional(training_result.get("exported_models", {}).get("torchscript"), project.root),
            "onnx": _relative_optional(training_result.get("exported_models", {}).get("onnx"), project.root),
            "engine": _relative_optional(training_result.get("exported_models", {}).get("engine"), project.root),
        },
        "training": {
            "run_dir": _relative_optional(training_result.get("run_dir"), project.root),
            "results_csv": _relative_optional(training_result.get("results_csv"), project.root),
            "args_yaml": _relative_optional(training_result.get("args_yaml"), project.root),
            "log_path": _relative_optional(training_result.get("log_path"), project.root),
            "parameters": training_result.get("train_params", {}),
            "status": "stopped" if training_result.get("stopped_by_user") else "completed",
            "requested_epochs": training_result.get("requested_epochs"),
            "completed_epochs": training_result.get("completed_epochs"),
            "stopped_by_user": bool(training_result.get("stopped_by_user", False)),
            "final_model_source": training_result.get("final_model_source"),
        },
        "validation_metrics": _read_final_metrics(results_csv),
        "export_errors": training_result.get("exported_models", {}).get("errors", []),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "model_manifest.json"
    write_json_atomic(manifest_path, manifest)
    if set_active:
        project_data = dict(project.project)
        project_data["active_model"] = _relative(best_model, project.root)
        project_data["active_model_manifest"] = _relative(manifest_path, project.root)
        write_json_atomic(project.root / "project.json", project_data)
    return manifest


def _resolve_base_model(project_root: Path, project_data: dict[str, Any], value: str | Path | None) -> Path:
    raw = value if value not in (None, "") else project_data.get("active_model")
    if raw in (None, ""):
        raise ValueError("项目未指定 active_model，训练请求也没有 base_model_path")
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _read_final_metrics(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {}
    result: dict[str, Any] = {}
    for key, raw_value in rows[-1].items():
        key = key.strip()
        value = (raw_value or "").strip()
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _relative_optional(value: str | Path | None, root: Path) -> str | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.exists():
        return None
    try:
        return _relative(path, root)
    except ValueError:
        return str(path.resolve())
