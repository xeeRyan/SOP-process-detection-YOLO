"""JSON任务的中央调度器。

负责命令选择、默认配置和路径解析，具体业务委托给tools和scripts模块。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from scripts.config import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_ENABLE_YOLO,
    DEFAULT_HAND_POSE_MODEL_PATH,
    DEFAULT_NMS_THRESHOLD,
    DEFAULT_OUTPUT_JSON,
    DEFAULT_OUTPUT_VIDEO,
    DEFAULT_REALTIME_DISPLAY,
    DEFAULT_TCP_HOST,
    DEFAULT_TCP_PORT,
    ENABLE_HAND_POSE,
    HAND_POSE_SAMPLE_INTERVAL,
    ROOT,
)
from scripts.legacy_sk_config import DEFAULT_MODEL_PATH, DEFAULT_SOP_PROJECT_DIR, DEFAULT_VIDEO_PATH
from scripts.project.storage import read_json_object, resolve_within


APP_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else ROOT
DEFAULT_CONFIG_PATH = APP_ROOT / "config" / "app_config.json"
PROJECTS_ROOT = APP_ROOT / "projects"
SUPPORTED_COMMANDS = [
    "health",
    "list_projects",
    "create_project",
    "get_project",
    "update_project",
    "save_rois",
    "save_workflow",
    "activate_project",
    "extract_frames",
    "import_annotations",
    "build_dataset",
    "train_project",
    "convert_project_model",
    "detect",
    "train",
]


def read_json(path: str | Path) -> dict[str, Any]:
    """读取 UTF-8 JSON 配置文件。"""

    return read_json_object(path)


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
            "api_version": "1.2",
            "capabilities": [
                "project_model_conversion",
                "training_progress_file",
                "training_stop_after_epoch",
            ],
            "commands": SUPPORTED_COMMANDS,
        }
    handler = _command_handlers().get(str(command))
    if handler is None:
        raise ValueError(f"不支持的任务类型: {command}")
    return handler(dict_data)


def _command_handlers():
    return {
        "list_projects": run_list_projects,
        "create_project": run_create_project,
        "get_project": run_get_project,
        "update_project": run_update_project,
        "save_rois": run_save_rois,
        "save_workflow": run_save_workflow,
        "activate_project": run_activate_project,
        "extract_frames": run_extract_frames,
        "import_annotations": run_import_annotations,
        "build_dataset": run_build_dataset,
        "train_project": run_train_project,
        "convert_project_model": run_convert_project_model,
        "detect": run_detect,
        "train": run_train,
    }


def run_list_projects(_dict_data: dict[str, Any]) -> dict[str, Any]:
    from tools.project_management import list_sop_projects

    return list_sop_projects(PROJECTS_ROOT)


def run_get_project(dict_data: dict[str, Any]) -> dict[str, Any]:
    from tools.project_management import get_sop_project

    return get_sop_project(_required_project_dir(dict_data))


def run_activate_project(dict_data: dict[str, Any]) -> dict[str, Any]:
    from tools.project_management import activate_sop_project

    return activate_sop_project(_required_project_dir(dict_data))


def _required_project_dir(dict_data: dict[str, Any]) -> Path:
    params = dict_data.get("params", dict_data)
    project_dir = resolve_project_path(params.get("sop_project_dir"))
    if project_dir is None:
        raise ValueError("请求需要 sop_project_dir")
    return project_dir


def run_create_project(dict_data: dict[str, Any]) -> dict[str, Any]:
    from tools.project_management import create_sop_project

    params = dict_data.get("params", dict_data)
    return create_sop_project(
        projects_root=PROJECTS_ROOT,
        project_id=str(params.get("project_id") or ""),
        name=str(params.get("name") or ""),
        description=str(params.get("description") or ""),
        classes=params.get("classes") or [],
    )


def run_update_project(dict_data: dict[str, Any]) -> dict[str, Any]:
    from tools.project_management import update_project_metadata

    params = dict_data.get("params", dict_data)
    return update_project_metadata(
        _required_project_dir(dict_data),
        name=params.get("name"),
        description=params.get("description"),
        classes=params.get("classes"),
    )


def run_save_rois(dict_data: dict[str, Any]) -> dict[str, Any]:
    from tools.project_management import save_project_rois

    params = dict_data.get("params", dict_data)
    rois = params.get("rois")
    if not isinstance(rois, dict):
        raise ValueError("save_rois 需要 rois 对象")
    return save_project_rois(_required_project_dir(dict_data), rois)


def run_save_workflow(dict_data: dict[str, Any]) -> dict[str, Any]:
    from tools.project_management import save_project_workflow

    params = dict_data.get("params", dict_data)
    workflow = params.get("workflow")
    if not isinstance(workflow, dict):
        raise ValueError("save_workflow 需要 workflow 对象")
    return save_project_workflow(_required_project_dir(dict_data), workflow)


def run_extract_frames(dict_data: dict[str, Any]) -> dict[str, Any]:
    """导入 SOP 流程视频并生成待标注图片及清单。"""

    from tools.project_frames import extract_project_videos

    params = dict_data.get("params", dict_data)
    raw_paths = params.get("video_paths")
    if raw_paths is None and params.get("video_path") is not None:
        raw_paths = [params["video_path"]]
    if isinstance(raw_paths, (str, Path)):
        raw_paths = [raw_paths]
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError("extract_frames 需要 video_paths 或 video_path")

    project_dir = resolve_project_path(params.get("sop_project_dir")) or DEFAULT_SOP_PROJECT_DIR
    video_paths = [resolve_project_path(path) for path in raw_paths]
    return extract_project_videos(
        project_dir=project_dir,
        video_paths=[path for path in video_paths if path is not None],
        frames_per_second=float(params.get("frames_per_second", 2.0)),
        frame_interval=int(params["frame_interval"]) if params.get("frame_interval") is not None else None,
        max_frames_per_video=int(params.get("max_frames_per_video", 0)),
        copy_videos=bool(params.get("copy_videos", True)),
        jpeg_quality=int(params.get("jpeg_quality", 95)),
    )


def run_import_annotations(dict_data: dict[str, Any]) -> dict[str, Any]:
    """导入并校验外部工具生成的 YOLO 标签。"""

    from tools.annotation_import import import_yolo_annotations

    params = dict_data.get("params", dict_data)
    labels_dir = resolve_project_path(params.get("labels_dir"))
    if labels_dir is None:
        raise ValueError("import_annotations 需要 labels_dir")
    project_dir = resolve_project_path(params.get("sop_project_dir")) or DEFAULT_SOP_PROJECT_DIR
    return import_yolo_annotations(
        project_dir=project_dir,
        labels_dir=labels_dir,
        overwrite=bool(params.get("overwrite", True)),
        mark_missing_as_empty=bool(params.get("mark_missing_as_empty", False)),
    )


def run_build_dataset(dict_data: dict[str, Any]) -> dict[str, Any]:
    """按视频分组生成标准 YOLO 训练数据集。"""

    from tools.dataset_builder import build_yolo_dataset

    params = dict_data.get("params", dict_data)
    project_dir = resolve_project_path(params.get("sop_project_dir")) or DEFAULT_SOP_PROJECT_DIR
    return build_yolo_dataset(
        project_dir=project_dir,
        dataset_name=params.get("dataset_name"),
        train_ratio=float(params.get("train_ratio", 0.7)),
        val_ratio=float(params.get("val_ratio", 0.2)),
        test_ratio=float(params.get("test_ratio", 0.1)),
        seed=int(params.get("seed", 42)),
        require_all_annotated=bool(params.get("require_all_annotated", True)),
        require_all_splits=bool(params.get("require_all_splits", True)),
        allow_single_video_frame_split=bool(
            params.get("allow_single_video_frame_split", False)
        ),
    )


def run_train_project(dict_data: dict[str, Any]) -> dict[str, Any]:
    """使用项目内指定版本数据集训练并登记模型。"""

    from tools.project_training import train_project_model

    params = dict_data.get("params", dict_data)
    dataset_name = str(params.get("dataset_name") or "")
    model_id = str(params.get("model_id") or "")
    if not dataset_name or not model_id:
        raise ValueError("train_project 需要 dataset_name 和 model_id")
    project_dir = resolve_project_path(params.get("sop_project_dir")) or DEFAULT_SOP_PROJECT_DIR
    base_model = resolve_project_path(params.get("base_model_path"))
    return train_project_model(
        project_dir=project_dir,
        dataset_name=dataset_name,
        model_id=model_id,
        model_version=params.get("model_version"),
        base_model_path=base_model,
        set_active=bool(params.get("set_active", True)),
        params=params,
    )


def run_convert_project_model(dict_data: dict[str, Any]) -> dict[str, Any]:
    """将已登记模型的 best.pt 转换为部署格式。"""

    from tools.project_model_conversion import convert_project_model

    params = dict_data.get("params", dict_data)
    model_id = str(params.get("model_id") or "")
    model_version = str(params.get("model_version") or "")
    if not model_id or not model_version:
        raise ValueError("convert_project_model 需要 model_id 和 model_version")
    project_dir = resolve_project_path(params.get("sop_project_dir")) or DEFAULT_SOP_PROJECT_DIR
    return convert_project_model(
        project_dir=project_dir,
        model_id=model_id,
        model_version=model_version,
        formats=params.get("formats"),
        options=params,
    )


def run_detect(dict_data: dict[str, Any]) -> dict[str, Any]:
    """执行视频级 SOP 工序检测。"""

    from scripts.main_video import process_video

    config = load_default_config()
    paths_config = config.get("paths", {})
    ai_config = config.get("ai", {})
    output_config = config.get("output", {})
    hand_pose_config = config.get("hand_pose", {})
    tracking_config = config.get("tracking", {})
    sop_config = config.get("sop", {})

    params = dict_data.get("params", dict_data)
    sop_project_dir = resolve_project_path(
        params.get("sop_project_dir"),
        paths_config.get("sop_project_dir"),
    ) or DEFAULT_SOP_PROJECT_DIR
    requested_model = params.get("model_path")
    model_path = resolve_project_path(requested_model) if requested_model not in (None, "") else None
    output_dir = resolve_detect_output_dir(
        sop_project_dir,
        params.get("output_dir", paths_config.get("output_dir", "outputs/frontend_detect")),
    )
    return process_video(
        video_path=resolve_project_path(params.get("video_path"), paths_config.get("video_path")) or DEFAULT_VIDEO_PATH,
        model_path=model_path,
        output_dir=output_dir,
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
        tracking_iou_threshold=float(
            params.get("tracking_iou_threshold", tracking_config.get("iou_threshold", 0.2))
        ),
        tracking_max_missing_frames=int(
            params.get("tracking_max_missing_frames", tracking_config.get("max_missing_frames", 8))
        ),
        event_lost_tolerance_frames=int(
            params.get(
                "event_lost_tolerance_frames",
                tracking_config.get("event_lost_tolerance_frames", 8),
            )
        ),
        enable_yolo=params.get("enable_yolo", ai_config.get("enable_yolo", DEFAULT_ENABLE_YOLO)),
        confidence_threshold=params.get("confidence_threshold", ai_config.get("confidence_threshold", CONFIDENCE_THRESHOLD)),
        nms_threshold=params.get("nms_threshold", ai_config.get("nms_threshold", DEFAULT_NMS_THRESHOLD)),
        inference_device=params.get("inference_device"),
        inference_backend=params.get("inference_backend", ai_config.get("inference_backend", "python")),
        target_classes=params.get("target_classes"),
        sop_step_enabled=params.get("sop_step_enabled", sop_config.get("step_enabled")),
        sop_trigger_sources=params.get("sop_trigger_sources", sop_config.get("trigger_sources")),
        sop_step_timeouts_sec=params.get("sop_step_timeouts_sec", sop_config.get("step_timeouts_sec")),
        output_video=params.get("output_video", output_config.get("output_video", DEFAULT_OUTPUT_VIDEO)),
        output_json=params.get("output_json", output_config.get("output_json", DEFAULT_OUTPUT_JSON)),
        realtime_display=params.get("realtime_display", output_config.get("realtime_display", DEFAULT_REALTIME_DISPLAY)),
        sop_project_dir=sop_project_dir,
    )


def resolve_detect_output_dir(project_dir: str | Path, value: str | Path | None) -> Path:
    """将检测输出目录解析到当前 SOP 项目内部。"""

    relative_or_absolute = value if value not in (None, "") else "outputs/frontend_detect"
    return resolve_within(project_dir, relative_or_absolute, field="output_dir")


def run_train(dict_data: dict[str, Any]) -> dict[str, Any]:
    """执行 YOLO 模型训练。"""

    from tools.training import train_yolo_model

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
    )



