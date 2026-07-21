from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from scripts.config import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_ENABLE_YOLO,
    DEFAULT_HAND_POSE_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_NMS_THRESHOLD,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_JSON,
    DEFAULT_OUTPUT_VIDEO,
    DEFAULT_REALTIME_DISPLAY,
    DEFAULT_TCP_HOST,
    DEFAULT_TCP_PORT,
    DEFAULT_VIDEO_PATH,
    ENABLE_HAND_POSE,
    HAND_POSE_SAMPLE_INTERVAL,
    ROOT,
    SCREW_BIN_ROI,
    TARGET_CLASSES,
    TOOL_HOME_ROI,
    WORK_ROI,
)
from scripts.main_video import process_video
from tools.training import train_yolo_model


APP_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else ROOT
DEFAULT_CONFIG_PATH = APP_ROOT / "config" / "app_config.json"


def read_json(path: str | Path) -> dict[str, Any]:
    """读取 UTF-8 JSON 配置文件。"""

    with open(path, "r", encoding="utf-8-sig") as fp:
        return json.load(fp)


def save_json(json_data: dict[str, Any], path: str | Path) -> None:
    """保存 UTF-8 JSON 文件。"""

    with open(path, "w", encoding="utf-8") as fp:
        json.dump(json_data, fp, ensure_ascii=False, indent=2)


def load_default_config() -> dict[str, Any]:
    """读取项目默认配置文件。"""

    if DEFAULT_CONFIG_PATH.exists():
        return read_json(DEFAULT_CONFIG_PATH)
    return {}


def resolve_project_path(value: str | Path | None, default: str | Path | None = None) -> Path | None:
    """将相对路径解析到项目根目录，打包后解析到 exe 所在目录。"""

    raw_value = value if value not in (None, "") else default
    if raw_value in (None, ""):
        return None

    path = Path(raw_value)
    if not path.is_absolute():
        path = APP_ROOT / path
    return path


def get_tcp_config() -> tuple[str, int]:
    """读取 TCP 默认监听地址，优先使用 config/app_config.json。"""

    tcp_config = load_default_config().get("tcp", {})
    host = str(tcp_config.get("host") or DEFAULT_TCP_HOST)
    port = int(tcp_config.get("port") or DEFAULT_TCP_PORT)
    return host, port


def run_task(dict_data: dict[str, Any]) -> dict[str, Any]:
    """根据 JSON 中的 command 执行健康检查、模型训练或视频检测。"""

    command = dict_data.get("command") or dict_data.get("task") or dict_data.get("type") or "detect"
    if command == "health":
        return {
            "service": "SOP_PYD",
            "status": "ok",
            "commands": ["health", "detect", "train"],
        }
    if command == "train":
        return run_train(dict_data)
    if command == "detect":
        return run_detect(dict_data)
    raise ValueError(f"涓嶆敮鎸佺殑浠诲姟绫诲瀷: {command}")


def run_detect(dict_data: dict[str, Any]) -> dict[str, Any]:
    """执行视频级 SOP 工序检测。"""

    config = load_default_config()
    paths_config = config.get("paths", {})
    roi_config = config.get("roi", {})
    ai_config = config.get("ai", {})
    output_config = config.get("output", {})
    hand_pose_config = config.get("hand_pose", {})
    sop_config = config.get("sop", {})

    params = dict_data.get("params", dict_data)
    return process_video(
        video_path=resolve_project_path(params.get("video_path"), paths_config.get("video_path")) or DEFAULT_VIDEO_PATH,
        model_path=resolve_project_path(params.get("model_path"), paths_config.get("model_path")) or DEFAULT_MODEL_PATH,
        output_dir=resolve_project_path(params.get("output_dir"), paths_config.get("output_dir")) or DEFAULT_OUTPUT_DIR,
        roi=params.get("work_roi") or roi_config.get("work") or WORK_ROI,
        screw_bin_roi=params.get("screw_bin_roi") or roi_config.get("screw_bin") or SCREW_BIN_ROI,
        tool_home_roi=params.get("tool_home_roi") or roi_config.get("tool_home") or TOOL_HOME_ROI,
        enable_hand_pose=params.get("enable_hand_pose", hand_pose_config.get("enabled", ENABLE_HAND_POSE)),
        hand_pose_model_path=resolve_project_path(
            params.get("hand_pose_model_path"),
            paths_config.get("hand_pose_model_path"),
        )
        or DEFAULT_HAND_POSE_MODEL_PATH,
        hand_pose_sample_interval=int(
            params.get(
                "hand_pose_sample_interval",
                hand_pose_config.get("sample_interval", HAND_POSE_SAMPLE_INTERVAL),
            )
        ),
        enable_yolo=params.get("enable_yolo", ai_config.get("enable_yolo", DEFAULT_ENABLE_YOLO)),
        confidence_threshold=params.get("confidence_threshold", ai_config.get("confidence_threshold", CONFIDENCE_THRESHOLD)),
        nms_threshold=params.get("nms_threshold", ai_config.get("nms_threshold", DEFAULT_NMS_THRESHOLD)),
        target_classes=params.get("target_classes", ai_config.get("target_classes", TARGET_CLASSES)),
        sop_step_enabled=params.get("sop_step_enabled", sop_config.get("step_enabled")),
        sop_trigger_sources=params.get("sop_trigger_sources", sop_config.get("trigger_sources")),
        sop_step_timeouts_sec=params.get("sop_step_timeouts_sec", sop_config.get("step_timeouts_sec")),
        output_video=params.get("output_video", output_config.get("output_video", DEFAULT_OUTPUT_VIDEO)),
        output_json=params.get("output_json", output_config.get("output_json", DEFAULT_OUTPUT_JSON)),
        realtime_display=params.get("realtime_display", output_config.get("realtime_display", DEFAULT_REALTIME_DISPLAY)),
    )


def run_train(dict_data: dict[str, Any]) -> dict[str, Any]:
    """执行 YOLO 模型训练。"""

    config = load_default_config()
    train_config = config.get("train", {})
    params = dict_data.get("params", dict_data)
    return train_yolo_model(
        base_model_path=resolve_project_path(
            params.get("base_model_path") or params.get("model"),
            train_config.get("base_model_path") or train_config.get("model"),
        )
        or APP_ROOT / "models" / "yolo26s.pt",
        data_yaml_path=resolve_project_path(
            params.get("data_yaml_path") or params.get("data"),
            train_config.get("data_yaml_path") or train_config.get("data"),
        )
        or APP_ROOT / "datasets" / "sop" / "sop.yaml",
        project_dir=resolve_project_path(params.get("project_dir"), train_config.get("project_dir")) or APP_ROOT / "runs",
        run_name=params.get("run_name", train_config.get("run_name", "sop_yolo26s")),
        epochs=int(params.get("epochs", train_config.get("epochs", 100))),
        imgsz=int(params.get("imgsz", train_config.get("imgsz", 640))),
        batch=int(params.get("batch", train_config.get("batch", 8))),
        workers=int(params.get("workers", train_config.get("workers", 0))),
        device=params.get("device", train_config.get("device")),
        optimizer=params.get("optimizer", train_config.get("optimizer", "auto")),
        amp=bool(params.get("amp", train_config.get("amp", True))),
        val_ratio=params.get("val_ratio", train_config.get("val_ratio")),
        images_dir=resolve_project_path(params.get("images_dir"), train_config.get("images_dir")),
        labels_dir=resolve_project_path(params.get("labels_dir"), train_config.get("labels_dir")),
        dataset_output_dir=resolve_project_path(params.get("dataset_output_dir"), train_config.get("dataset_output_dir")),
        copy_best_to=resolve_project_path(params.get("copy_best_to"), train_config.get("copy_best_to")) or DEFAULT_MODEL_PATH,
        exist_ok=bool(params.get("exist_ok", train_config.get("exist_ok", True))),
        hsv_h=params.get("hsv_h", train_config.get("hsv_h")),
        hsv_s=params.get("hsv_s", train_config.get("hsv_s")),
        hsv_v=params.get("hsv_v", train_config.get("hsv_v")),
        degrees=params.get("degrees", train_config.get("degrees")),
        translate=params.get("translate", train_config.get("translate")),
        scale=params.get("scale", train_config.get("scale")),
        shear=params.get("shear", train_config.get("shear")),
        fliplr=params.get("fliplr", train_config.get("fliplr")),
        flipud=params.get("flipud", train_config.get("flipud")),
        mosaic=params.get("mosaic", train_config.get("mosaic")),
        mixup=params.get("mixup", train_config.get("mixup")),
        copy_paste=params.get("copy_paste", train_config.get("copy_paste")),
        export_torchscript=bool(
            params.get("export_torchscript", train_config.get("export_torchscript", False))
        ),
        export_onnx=bool(params.get("export_onnx", train_config.get("export_onnx", False))),
        export_engine=bool(params.get("export_engine", train_config.get("export_engine", False))),
        torchscript_output_path=resolve_project_path(
            params.get("torchscript_output_path"), train_config.get("torchscript_output_path")
        ),
        onnx_output_path=resolve_project_path(params.get("onnx_output_path"), train_config.get("onnx_output_path")),
        engine_output_path=resolve_project_path(params.get("engine_output_path"), train_config.get("engine_output_path")),
        export_imgsz=params.get("export_imgsz", train_config.get("export_imgsz")),
        export_opset=int(params.get("export_opset", train_config.get("export_opset", 12))),
        export_dynamic=bool(params.get("export_dynamic", train_config.get("export_dynamic", False))),
        export_simplify=bool(params.get("export_simplify", train_config.get("export_simplify", True))),
        export_overwrite=bool(params.get("export_overwrite", train_config.get("export_overwrite", True))),
        trtexec_path=params.get("trtexec_path", train_config.get("trtexec_path", "trtexec")),
        engine_fp16=bool(params.get("engine_fp16", train_config.get("engine_fp16", True))),
        engine_workspace_mb=params.get("engine_workspace_mb", train_config.get("engine_workspace_mb")),
        engine_verbose=bool(params.get("engine_verbose", train_config.get("engine_verbose", False))),
        engine_dry_run=bool(params.get("engine_dry_run", train_config.get("engine_dry_run", False))),
        torchscript_optimize=bool(
            params.get("torchscript_optimize", train_config.get("torchscript_optimize", False))
        ),
        export_strict=bool(params.get("export_strict", train_config.get("export_strict", False))),
        export_python_path=resolve_project_path(params.get("export_python_path"), train_config.get("export_python_path")),
    )



