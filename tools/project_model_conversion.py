"""将已登记项目模型转换为TorchScript、ONNX或TensorRT Engine。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.project import load_sop_project
from scripts.project.storage import read_json_object, utc_now, validate_safe_name, write_json_atomic
from scripts.runtime_logging import close_operation_logger, create_operation_logger, log_event
from tools.training import export_model_formats


SUPPORTED_MODEL_FORMATS = ("torchscript", "onnx", "engine")


def convert_project_model(
    project_dir: str | Path,
    model_id: str,
    model_version: str,
    formats: list[str],
    *,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a registered best.pt into selected deployment formats."""

    project = load_sop_project(project_dir)
    validate_safe_name(model_id, "model_id")
    validate_safe_name(model_version, "model_version", allow_dot=True)
    requested = _validate_formats(formats)
    params = options or {}

    model_dir = project.root / "models" / model_id / model_version
    manifest_path = model_dir / "model_manifest.json"
    manifest = read_json_object(manifest_path, description="模型清单")
    if manifest.get("model_id") != model_id or manifest.get("model_version") != model_version:
        raise ValueError("模型清单与请求的 model_id/model_version 不一致")

    source_model = model_dir / "best.pt"
    if not source_model.is_file():
        raise FileNotFoundError(f"未找到模型源文件: {source_model}")

    logger, log_path = create_operation_logger("model_convert", project.root / "outputs" / "logs")
    log_event(
        logger,
        "model_conversion_started",
        source_model=str(source_model),
        formats=requested,
    )
    try:
        exported = export_model_formats(
            source_model_path=source_model,
            imgsz=int(params.get("export_imgsz", 640)),
            export_torchscript_enabled="torchscript" in requested,
            export_onnx_enabled="onnx" in requested,
            export_engine_enabled="engine" in requested,
            torchscript_output_path=model_dir / "best.torchscript",
            onnx_output_path=model_dir / "best.onnx",
            engine_output_path=model_dir / "best.engine",
            export_opset=int(params.get("export_opset", 12)),
            export_dynamic=bool(params.get("export_dynamic", False)),
            export_simplify=bool(params.get("export_simplify", True)),
            export_overwrite=bool(params.get("overwrite", False)),
            trtexec_path=str(params.get("trtexec_path", "trtexec")),
            engine_fp16=bool(params.get("engine_fp16", True)),
            engine_workspace_mb=params.get("engine_workspace_mb"),
            engine_verbose=bool(params.get("engine_verbose", False)),
            engine_dry_run=bool(params.get("engine_dry_run", False)),
            torchscript_optimize=bool(params.get("torchscript_optimize", False)),
            # Conversion outcomes are independent; one failed format must not
            # discard successful artifacts from the same request.
            export_strict=False,
            export_python_path=params.get("export_python_path"),
            logger=logger,
            reuse_existing_onnx="engine" in requested and "onnx" not in requested,
        )
        _update_manifest_artifacts(manifest, manifest_path, project.root, exported)
        conversions = _build_conversion_statuses(requested, exported)
        result = {
            "project_id": project.project_id,
            "model_id": model_id,
            "model_version": model_version,
            "source_model": str(source_model),
            "model_manifest": str(manifest_path),
            "conversions": conversions,
            "errors": exported.get("errors", []),
            "log_path": str(log_path),
        }
        log_event(logger, "model_conversion_finished", **result)
        return result
    finally:
        close_operation_logger(logger)


def _validate_formats(formats: list[str]) -> list[str]:
    if not isinstance(formats, list) or not formats:
        raise ValueError("formats 必须是非空数组")
    normalized = list(dict.fromkeys(str(item).strip().lower() for item in formats))
    unsupported = [item for item in normalized if item not in SUPPORTED_MODEL_FORMATS]
    if unsupported:
        raise ValueError(f"不支持的模型转换格式: {', '.join(unsupported)}")
    return normalized


def _update_manifest_artifacts(
    manifest: dict[str, Any],
    manifest_path: Path,
    project_root: Path,
    exported: dict[str, Any],
) -> None:
    artifacts = manifest.setdefault("artifacts", {})
    for name in SUPPORTED_MODEL_FORMATS:
        value = exported.get(name)
        if value and Path(value).is_file():
            artifacts[name] = Path(value).resolve().relative_to(project_root.resolve()).as_posix()
    manifest["converted_at"] = utc_now()
    write_json_atomic(manifest_path, manifest)


def _build_conversion_statuses(
    requested: list[str], exported: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    errors = list(exported.get("errors", []))
    statuses: dict[str, dict[str, Any]] = {}
    for name in requested:
        path = exported.get(name)
        matching_errors = [error for error in errors if error.startswith(f"{name} export failed:")]
        statuses[name] = {
            "status": "success" if path else "failed",
            "path": path,
            "error": matching_errors[0] if matching_errors else (errors[0] if not path and errors else None),
        }
    return statuses
