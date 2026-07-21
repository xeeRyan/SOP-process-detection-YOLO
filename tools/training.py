from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.config import ROOT
from scripts.runtime_logging import close_operation_logger, create_operation_logger, log_event
from deploy.export_onnx import export_onnx as export_onnx_model
from deploy.export_tensorrt import export_tensorrt as export_tensorrt_model
from deploy.export_torchscript import export_torchscript as export_torchscript_model


DEFAULT_BASE_MODEL_PATH = ROOT / "models" / "yolo26s.pt"
DEFAULT_DATA_YAML_PATH = ROOT / "datasets" / "sop" / "sop.yaml"
DEFAULT_TRAIN_PROJECT = ROOT / "runs"
DEFAULT_TRAIN_NAME = "sop_yolo26s"
DEFAULT_BEST_MODEL_PATH = ROOT / "models" / "best_yolo26s.pt"

# 绗竴鐗堝紑鏀剧粰杞欢绔殑 YOLO 璁粌澧炲己鍙傛暟銆?
AUGMENTATION_PARAM_NAMES = (
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "fliplr",
    "flipud",
    "mosaic",
    "mixup",
    "copy_paste",
)


def train_yolo_model(
    base_model_path: str | Path = DEFAULT_BASE_MODEL_PATH,
    data_yaml_path: str | Path = DEFAULT_DATA_YAML_PATH,
    project_dir: str | Path = DEFAULT_TRAIN_PROJECT,
    run_name: str = DEFAULT_TRAIN_NAME,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    workers: int = 0,
    device: str | int | None = None,
    optimizer: str = "auto",
    amp: bool = True,
    val_ratio: float | None = None,
    images_dir: str | Path | None = None,
    labels_dir: str | Path | None = None,
    dataset_output_dir: str | Path | None = None,
    copy_best_to: str | Path | None = DEFAULT_BEST_MODEL_PATH,
    exist_ok: bool = True,
    log_dir: str | Path | None = None,
    export_torchscript: bool = False,
    export_onnx: bool = False,
    export_engine: bool = False,
    torchscript_output_path: str | Path | None = None,
    onnx_output_path: str | Path | None = None,
    engine_output_path: str | Path | None = None,
    export_imgsz: int | None = None,
    export_opset: int = 12,
    export_dynamic: bool = False,
    export_simplify: bool = True,
    export_overwrite: bool = True,
    trtexec_path: str = "trtexec",
    engine_fp16: bool = True,
    engine_workspace_mb: int | None = None,
    engine_verbose: bool = False,
    engine_dry_run: bool = False,
    torchscript_optimize: bool = False,
    export_strict: bool = False,
    export_python_path: str | Path | None = None,
    **augmentation_params: Any,
) -> dict:
    """璁粌 YOLO26s SOP 妫€娴嬫ā鍨嬨€?

    闈㈠悜杞欢绔殑绗竴鐗堝彲璋冨弬鏁板寘鎷細epochs銆乥atch銆乮mgsz銆亀orkers銆乨evice銆乵odel銆乷ptimizer銆乤mp銆?
    val_ratio 浠ュ強甯哥敤鏁版嵁澧炲己鍙傛暟銆倂al_ratio 鍙湁鍦ㄥ悓鏃朵紶鍏?images_dir/labels_dir 鏃讹紝
    鎵嶄細瑙﹀彂鏁版嵁闆嗛噸鏂板垝鍒嗐€?
    """

    base_model_path = Path(base_model_path)
    data_yaml_path = Path(data_yaml_path)
    project_dir = Path(project_dir)
    logger, log_path = create_operation_logger("train", log_dir or project_dir.parent / "outputs" / "logs")

    selected_augmentations = _collect_augmentation_params(augmentation_params)
    train_parameters = _build_train_parameters(
        base_model_path=base_model_path,
        data_yaml_path=data_yaml_path,
        project_dir=project_dir,
        run_name=run_name,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        optimizer=optimizer,
        amp=amp,
        val_ratio=val_ratio,
        images_dir=images_dir,
        labels_dir=labels_dir,
        dataset_output_dir=dataset_output_dir,
        copy_best_to=copy_best_to,
        exist_ok=exist_ok,
        augmentations=selected_augmentations,
        export_options={
            "export_torchscript": bool(export_torchscript),
            "export_onnx": bool(export_onnx),
            "export_engine": bool(export_engine),
            "torchscript_output_path": str(torchscript_output_path) if torchscript_output_path is not None else None,
            "onnx_output_path": str(onnx_output_path) if onnx_output_path is not None else None,
            "engine_output_path": str(engine_output_path) if engine_output_path is not None else None,
            "export_imgsz": int(export_imgsz) if export_imgsz is not None else int(imgsz),
            "export_opset": int(export_opset),
            "export_dynamic": bool(export_dynamic),
            "export_simplify": bool(export_simplify),
            "export_overwrite": bool(export_overwrite),
            "trtexec_path": trtexec_path,
            "engine_fp16": bool(engine_fp16),
            "engine_workspace_mb": engine_workspace_mb,
            "engine_verbose": bool(engine_verbose),
            "engine_dry_run": bool(engine_dry_run),
            "torchscript_optimize": bool(torchscript_optimize),
            "export_strict": bool(export_strict),
            "export_python_path": str(export_python_path) if export_python_path is not None else None,
        },
    )
    log_event(logger, "train_started", parameters=train_parameters)
    log_event(logger, "train_parameters", parameters=train_parameters)

    if val_ratio is not None and images_dir is not None and labels_dir is not None:
        from tools.prepare_dataset import prepare_dataset

        dataset_output_path = Path(dataset_output_dir) if dataset_output_dir is not None else data_yaml_path.parent
        dataset_result = prepare_dataset(
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=dataset_output_path,
            val_ratio=float(val_ratio),
        )
        data_yaml_path = Path(dataset_result["yaml"])
        train_parameters["data"] = str(data_yaml_path)
        train_parameters["dataset_prepared"] = dataset_result
        log_event(logger, "dataset_prepared", **dataset_result)
        log_event(logger, "train_parameters", parameters=train_parameters)

    if not base_model_path.exists():
        log_event(logger, "train_failed", reason="base_model_not_found", base_model_path=base_model_path)
        close_operation_logger(logger)
        raise FileNotFoundError(f"鏈壘鍒板垵濮嬫ā鍨? {base_model_path}")
    if not data_yaml_path.exists():
        log_event(logger, "train_failed", reason="dataset_yaml_not_found", data_yaml_path=data_yaml_path)
        close_operation_logger(logger)
        raise FileNotFoundError(f"鏈壘鍒版暟鎹泦閰嶇疆: {data_yaml_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        logger.exception("train_failed | ultralytics_import_failed | %s", exc)
        close_operation_logger(logger)
        raise ImportError("Missing ultralytics. Install the dependencies from requirements.txt first.") from exc

    model = YOLO(str(base_model_path))
    model.add_callback("on_fit_epoch_end", _build_epoch_logger(logger))
    train_kwargs = {
        "data": str(data_yaml_path),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "workers": int(workers),
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": exist_ok,
        "optimizer": optimizer,
        "amp": bool(amp),
        **selected_augmentations,
    }
    if device is not None:
        train_kwargs["device"] = device

    try:
        model.train(**train_kwargs)
    except Exception as exc:
        logger.exception("train_failed | %s", exc)
        close_operation_logger(logger)
        raise

    run_dir = project_dir / run_name
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    args_yaml = run_dir / "args.yaml"
    copied_best_path = None
    if copy_best_to is not None and best_path.exists():
        copy_target = Path(copy_best_to)
        copy_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, copy_target)
        copied_best_path = str(copy_target)

    export_source = Path(copied_best_path) if copied_best_path else best_path
    exported_models = _export_requested_models(
        source_model_path=export_source,
        imgsz=int(export_imgsz) if export_imgsz is not None else int(imgsz),
        export_torchscript_enabled=bool(export_torchscript),
        export_onnx_enabled=bool(export_onnx),
        export_engine_enabled=bool(export_engine),
        torchscript_output_path=torchscript_output_path,
        onnx_output_path=onnx_output_path,
        engine_output_path=engine_output_path,
        export_opset=int(export_opset),
        export_dynamic=bool(export_dynamic),
        export_simplify=bool(export_simplify),
        export_overwrite=bool(export_overwrite),
        trtexec_path=trtexec_path,
        engine_fp16=bool(engine_fp16),
        engine_workspace_mb=engine_workspace_mb,
        engine_verbose=bool(engine_verbose),
        engine_dry_run=bool(engine_dry_run),
        torchscript_optimize=bool(torchscript_optimize),
        export_strict=bool(export_strict),
        export_python_path=export_python_path,
        logger=logger,
    )

    result = {
        "run_dir": str(run_dir),
        "best_model": str(best_path) if best_path.exists() else None,
        "last_model": str(last_path) if last_path.exists() else None,
        "results_csv": str(results_csv) if results_csv.exists() else None,
        "args_yaml": str(args_yaml) if args_yaml.exists() else None,
        "copied_best_model": copied_best_path,
        "exported_models": exported_models,
        "train_params": train_parameters,
        "log_path": str(log_path),
    }
    log_event(logger, "train_finished", **result)
    close_operation_logger(logger)
    return result


def _collect_augmentation_params(params: dict[str, Any]) -> dict[str, Any]:
    """Select supported Ultralytics augmentation parameters."""

    selected: dict[str, Any] = {}
    for name in AUGMENTATION_PARAM_NAMES:
        if name in params and params[name] is not None:
            selected[name] = params[name]
    return selected


def _build_train_parameters(
    base_model_path: Path,
    data_yaml_path: Path,
    project_dir: Path,
    run_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    device: str | int | None,
    optimizer: str,
    amp: bool,
    val_ratio: float | None,
    images_dir: str | Path | None,
    labels_dir: str | Path | None,
    dataset_output_dir: str | Path | None,
    copy_best_to: str | Path | None,
    exist_ok: bool,
    augmentations: dict[str, Any],
    export_options: dict[str, Any],
) -> dict[str, Any]:
    """Build the effective training parameters for logs and return values."""

    return {
        "model": str(base_model_path),
        "data": str(data_yaml_path),
        "project_dir": str(project_dir),
        "run_name": run_name,
        "epochs": int(epochs),
        "batch": int(batch),
        "imgsz": int(imgsz),
        "workers": int(workers),
        "device": device,
        "optimizer": optimizer,
        "amp": bool(amp),
        "val_ratio": val_ratio,
        "images_dir": str(images_dir) if images_dir is not None else None,
        "labels_dir": str(labels_dir) if labels_dir is not None else None,
        "dataset_output_dir": str(dataset_output_dir) if dataset_output_dir is not None else None,
        "copy_best_to": str(copy_best_to) if copy_best_to is not None else None,
        "exist_ok": bool(exist_ok),
        "augmentations": augmentations,
        "export": export_options,
    }


def _export_requested_models(
    source_model_path: Path,
    imgsz: int,
    export_torchscript_enabled: bool,
    export_onnx_enabled: bool,
    export_engine_enabled: bool,
    torchscript_output_path: str | Path | None,
    onnx_output_path: str | Path | None,
    engine_output_path: str | Path | None,
    export_opset: int,
    export_dynamic: bool,
    export_simplify: bool,
    export_overwrite: bool,
    trtexec_path: str,
    engine_fp16: bool,
    engine_workspace_mb: int | None,
    engine_verbose: bool,
    engine_dry_run: bool,
    torchscript_optimize: bool,
    export_strict: bool,
    export_python_path: str | Path | None,
    logger,
) -> dict[str, Any]:
    """Export trained model to ONNX and/or TensorRT when requested by software."""

    result: dict[str, Any] = {
        "requested": {
            "torchscript": bool(export_torchscript_enabled),
            "onnx": bool(export_onnx_enabled),
            "engine": bool(export_engine_enabled),
        },
        "torchscript": None,
        "onnx": None,
        "engine": None,
        "errors": [],
    }
    if not export_torchscript_enabled and not export_onnx_enabled and not export_engine_enabled:
        return result

    if not source_model_path.exists():
        message = f"export source model not found: {source_model_path}"
        result["errors"].append(message)
        log_event(logger, "model_export_failed", reason=message)
        if export_strict:
            raise FileNotFoundError(message)
        return result

    onnx_path = Path(onnx_output_path) if onnx_output_path is not None else source_model_path.with_suffix(".onnx")
    torchscript_path = (
        Path(torchscript_output_path)
        if torchscript_output_path is not None
        else source_model_path.with_suffix(".torchscript")
    )

    if export_torchscript_enabled:
        try:
            log_event(logger, "torchscript_export_started", source_model=str(source_model_path), output_path=str(torchscript_path))
            external_python = _find_export_python(export_python_path)
            export_script = _find_export_script("export_torchscript.py")
            if external_python is not None and export_script is not None:
                exported_torchscript = _export_torchscript_with_external_python(
                    python_path=external_python,
                    export_script=export_script,
                    model_path=source_model_path,
                    output_path=torchscript_path,
                    imgsz=int(imgsz),
                    optimize=bool(torchscript_optimize),
                    overwrite=bool(export_overwrite),
                )
            else:
                exported_torchscript = export_torchscript_model(
                    model_path=source_model_path,
                    output_path=torchscript_path,
                    imgsz=int(imgsz),
                    optimize=bool(torchscript_optimize),
                    overwrite=bool(export_overwrite),
                )
            result["torchscript"] = str(exported_torchscript)
            log_event(logger, "torchscript_export_finished", output_path=str(exported_torchscript))
        except Exception as exc:
            message = f"torchscript export failed: {exc}"
            result["errors"].append(message)
            logger.exception("torchscript_export_failed | %s", exc)
            if export_strict:
                raise
    engine_path = Path(engine_output_path) if engine_output_path is not None else source_model_path.with_suffix(".engine")

    try:
        log_event(logger, "onnx_export_started", source_model=str(source_model_path), output_path=str(onnx_path))
        external_python = _find_export_python(export_python_path)
        export_script = _find_export_script("export_onnx.py")
        if external_python is not None and export_script is not None:
            exported_onnx = _export_onnx_with_external_python(
                python_path=external_python,
                export_script=export_script,
                model_path=source_model_path,
                output_path=onnx_path,
                imgsz=int(imgsz),
                opset=int(export_opset),
                simplify=bool(export_simplify),
                dynamic=bool(export_dynamic),
                overwrite=bool(export_overwrite),
            )
            log_event(logger, "onnx_export_used_external_python", python_path=str(external_python), export_script=str(export_script))
        else:
            exported_onnx = export_onnx_model(
                model_path=source_model_path,
                output_path=onnx_path,
                imgsz=int(imgsz),
                opset=int(export_opset),
                simplify=bool(export_simplify),
                dynamic=bool(export_dynamic),
                overwrite=bool(export_overwrite),
            )
        result["onnx"] = str(exported_onnx)
        log_event(logger, "onnx_export_finished", output_path=str(exported_onnx))
    except Exception as exc:
        message = f"onnx export failed: {exc}"
        result["errors"].append(message)
        logger.exception("onnx_export_failed | %s", exc)
        if export_strict:
            raise
        return result

    if export_engine_enabled:
        try:
            log_event(logger, "engine_export_started", onnx_path=result["onnx"], output_path=str(engine_path))
            exported_engine = export_tensorrt_model(
                onnx_path=result["onnx"],
                engine_path=engine_path,
                fp16=bool(engine_fp16),
                workspace_mb=engine_workspace_mb,
                trtexec_path=trtexec_path,
                verbose=bool(engine_verbose),
                dry_run=bool(engine_dry_run),
            )
            result["engine"] = str(exported_engine)
            log_event(logger, "engine_export_finished", output_path=str(exported_engine), dry_run=bool(engine_dry_run))
        except Exception as exc:
            message = f"engine export failed: {exc}"
            result["errors"].append(message)
            logger.exception("engine_export_failed | %s", exc)
            if export_strict:
                raise

    return result


def _find_export_python(export_python_path: str | Path | None) -> Path | None:
    """Find an external Python interpreter for ONNX export outside PyInstaller."""

    candidates: list[Path] = []
    if export_python_path not in (None, ""):
        candidates.append(Path(export_python_path))

    env_python = os.environ.get("SOPAID_EXPORT_PYTHON")
    if env_python:
        candidates.append(Path(env_python))

    executable = Path(sys.executable).resolve()
    if getattr(sys, "frozen", False):
        for parent in executable.parents:
            candidates.append(parent / "sop_pack_env" / "python.exe")
    candidates.append(ROOT / "sop_pack_env" / "python.exe")

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved.exists() and resolved.name.lower() == "python.exe" and resolved != executable:
            return resolved
    return None


def _find_export_script(script_name: str) -> Path | None:
    """Find a deploy export script in the packaged or source tree."""

    executable = Path(sys.executable).resolve()
    candidates = []
    if getattr(sys, "frozen", False):
        candidates.append(executable.parent / "_internal" / "deploy" / script_name)
    candidates.append(ROOT / "deploy" / script_name)
    candidates.append(Path.cwd() / "deploy" / script_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _export_onnx_with_external_python(
    python_path: Path,
    export_script: Path,
    model_path: Path,
    output_path: Path,
    imgsz: int,
    opset: int,
    simplify: bool,
    dynamic: bool,
    overwrite: bool,
) -> Path:
    """Run deploy/export_onnx.py with external Python to avoid PyInstaller ONNX DLL issues."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_path),
        str(export_script),
        "--model",
        str(model_path),
        "--output",
        str(output_path),
        "--imgsz",
        str(int(imgsz)),
        "--opset",
        str(int(opset)),
    ]
    if dynamic:
        command.append("--dynamic")
    if not simplify:
        command.append("--no-simplify")
    if overwrite:
        command.append("--overwrite")

    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "external ONNX export failed: "
            f"stdout={completed.stdout.strip()} stderr={completed.stderr.strip()}"
        )
    if not output_path.exists():
        raise RuntimeError(f"external ONNX export finished, but output file was not found: {output_path}")
    return output_path

def _export_torchscript_with_external_python(
    python_path: Path,
    export_script: Path,
    model_path: Path,
    output_path: Path,
    imgsz: int,
    optimize: bool,
    overwrite: bool,
) -> Path:
    """Run TorchScript export outside PyInstaller."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_path),
        str(export_script),
        "--model",
        str(model_path),
        "--output",
        str(output_path),
        "--imgsz",
        str(int(imgsz)),
    ]
    if optimize:
        command.append("--optimize")
    if overwrite:
        command.append("--overwrite")

    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "external TorchScript export failed: "
            f"stdout={completed.stdout.strip()} stderr={completed.stderr.strip()}"
        )
    if not output_path.exists():
        raise RuntimeError(f"external TorchScript export finished, but output file was not found: {output_path}")
    return output_path

def _build_epoch_logger(logger):
    """Build an Ultralytics callback that logs epoch metrics."""

    def on_fit_epoch_end(trainer) -> None:
        loss_items = {}
        try:
            loss_items = trainer.label_loss_items(trainer.tloss)
        except Exception:
            pass
        log_event(
            logger,
            "train_epoch_finished",
            epoch=int(trainer.epoch) + 1,
            total_epochs=int(trainer.epochs),
            train_loss=loss_items,
            metrics=trainer.metrics or {},
            learning_rate=getattr(trainer, "lr", {}),
            fitness=getattr(trainer, "fitness", None),
        )

    return on_fit_epoch_end




