"""视频检测主流程：加载项目、推理、跟踪、事件判断并写出结果。"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2

from scripts.config import (
    DEFAULT_HAND_POSE_MODEL_PATH,
    DEFAULT_ENABLE_YOLO,
    DEFAULT_NMS_THRESHOLD,
    DEFAULT_OUTPUT_JSON,
    DEFAULT_OUTPUT_VIDEO,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REALTIME_DISPLAY,
    CONFIDENCE_THRESHOLD,
    ENABLE_HAND_POSE,
    HAND_POSE_SAMPLE_INTERVAL,
    HAND_POSE_ACTIVE_STEP_ONLY,
    VISION_INFERENCE_INTERVAL_FRAMES,
    TRACKING_IOU_THRESHOLD,
    TRACKING_MAX_MISSING_FRAMES,
    TRACKING_MIN_CONFIRMED_HITS,
    TRACKING_LOW_CONFIDENCE,
    TRACKING_NEW_TRACK_CONFIDENCE,
    TRACKING_SECOND_MATCH_IOU_THRESHOLD,
    EVENT_LOST_TOLERANCE_FRAMES,
    EVENT_ENTER_STABLE_FRAMES,
    EVENT_EXIT_STABLE_FRAMES,
    RESULT_JSON_NAME,
    RESULT_VIDEO_NAME,
)
from scripts.inference import DetectionFusion, ModelRouter, build_detector
from scripts.events import SopEventEngine
from scripts.project import load_sop_project
from scripts.sop_logic import SOPStateMachine
from scripts.tracking import ByteTrackTracker
from scripts.project.storage import write_json_atomic
from scripts.utils import ensure_dir
from scripts.visualizer import draw_detections, draw_hand_landmarks, draw_roi, draw_sop_status
from scripts.runtime_logging import close_operation_logger, create_operation_logger, log_event


# 主流程入口：视频输入 -> YOLO 检测 -> 手部骨骼 -> SOP 判定 -> 视频/JSON 输出。
def process_video(
    video_path: str | Path,
    *,
    sop_project_dir: str | Path,
    model_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    enable_hand_pose: bool = ENABLE_HAND_POSE,
    hand_pose_model_path: str | Path = DEFAULT_HAND_POSE_MODEL_PATH,
    hand_pose_sample_interval: int = HAND_POSE_SAMPLE_INTERVAL,
    hand_pose_active_step_only: bool = HAND_POSE_ACTIVE_STEP_ONLY,
    vision_inference_interval_frames: int = VISION_INFERENCE_INTERVAL_FRAMES,
    tracking_iou_threshold: float = TRACKING_IOU_THRESHOLD,
    tracking_max_missing_frames: int = TRACKING_MAX_MISSING_FRAMES,
    tracking_min_confirmed_hits: int = TRACKING_MIN_CONFIRMED_HITS,
    tracking_low_confidence: float = TRACKING_LOW_CONFIDENCE,
    tracking_new_track_confidence: float = TRACKING_NEW_TRACK_CONFIDENCE,
    tracking_second_match_iou_threshold: float = TRACKING_SECOND_MATCH_IOU_THRESHOLD,
    event_lost_tolerance_frames: int = EVENT_LOST_TOLERANCE_FRAMES,
    event_enter_stable_frames: int = EVENT_ENTER_STABLE_FRAMES,
    event_exit_stable_frames: int = EVENT_EXIT_STABLE_FRAMES,
    enable_yolo: bool = DEFAULT_ENABLE_YOLO,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    inference_device: str | int | None = None,
    inference_backend: str = "python",
    target_classes: list[str] | None = None,
    sop_step_enabled: dict[str, bool] | None = None,
    sop_trigger_sources: dict[str, str] | None = None,
    sop_step_timeouts_sec: dict[str, float] | None = None,
    output_video: bool = DEFAULT_OUTPUT_VIDEO,
    output_json: bool = DEFAULT_OUTPUT_JSON,
    realtime_display: bool = DEFAULT_REALTIME_DISPLAY,
) -> dict:
    """视频级 SOP 检测入口。

    对接用途：
    - 输入视频路径、YOLO 权重路径、输出目录和 ROI 配置。
    - 输出带标注的视频 result.mp4 和结构化结果 result.json。
    - 返回值与 result.json 内容一致，可直接给软件端消费。
    """

    video_path = Path(video_path).expanduser().resolve()
    sop_project = load_sop_project(sop_project_dir)
    output_dir = ensure_dir(Path(output_dir))
    if hand_pose_sample_interval <= 0:
        raise ValueError("hand_pose_sample_interval 必须是正整数")
    if vision_inference_interval_frames <= 0:
        raise ValueError("vision_inference_interval_frames 必须是正整数")
    if not 0 <= tracking_iou_threshold <= 1:
        raise ValueError("tracking_iou_threshold 必须在 0 到 1 之间")
    if tracking_max_missing_frames < 0 or event_lost_tolerance_frames < 0:
        raise ValueError("跟踪和事件丢失容忍帧数不能小于 0")
    if event_enter_stable_frames < 1 or event_exit_stable_frames < 1:
        raise ValueError("事件进入和退出确认帧数必须大于 0")
    if tracking_min_confirmed_hits < 1:
        raise ValueError("tracking_min_confirmed_hits 必须大于等于 1")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold必须在0到1之间")
    if not 0 <= nms_threshold <= 1:
        raise ValueError("nms_threshold必须在0到1之间")
    if not 0 <= tracking_low_confidence <= confidence_threshold <= tracking_new_track_confidence <= 1:
        raise ValueError(
            "跟踪置信度必须满足 0 <= low <= confidence_threshold <= new_track <= 1"
        )
    model_paths: dict[str, Path] = {}
    if enable_yolo:
        for task in sorted(sop_project.required_model_tasks):
            if task == "detect" and model_path not in (None, ""):
                model_paths[task] = Path(model_path).expanduser().resolve()
            else:
                model_paths[task] = sop_project.resolve_model_path(task)
    model_path = model_paths.get("detect")
    if model_path is None and model_paths:
        model_path = next(iter(model_paths.values()))
    active_classes = list(sop_project.class_names if target_classes is None else target_classes)
    if not active_classes:
        raise ValueError("target_classes不能为空")
    unknown_classes = sorted(set(active_classes) - set(sop_project.class_names))
    if unknown_classes:
        raise ValueError(
            "target_classes包含当前SOP项目未定义的类别: " + ", ".join(unknown_classes)
        )
    pipeline_config = _build_pipeline_config(
        enable_yolo=enable_yolo,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        inference_device=inference_device,
        inference_backend=inference_backend,
        vision_task=sop_project.active_model_task,
        vision_tasks=sorted(model_paths),
        target_classes=active_classes,
        sop_step_enabled=sop_step_enabled,
        sop_trigger_sources=sop_trigger_sources,
        sop_step_timeouts_sec=sop_step_timeouts_sec,
        output_video=output_video,
        output_json=output_json,
        realtime_display=realtime_display,
        tracking_iou_threshold=tracking_iou_threshold,
        tracking_max_missing_frames=tracking_max_missing_frames,
        tracking_min_confirmed_hits=tracking_min_confirmed_hits,
        tracking_low_confidence=tracking_low_confidence,
        tracking_new_track_confidence=tracking_new_track_confidence,
        tracking_second_match_iou_threshold=tracking_second_match_iou_threshold,
        event_lost_tolerance_frames=event_lost_tolerance_frames,
        event_enter_stable_frames=event_enter_stable_frames,
        event_exit_stable_frames=event_exit_stable_frames,
    )
    pipeline_config["ai"]["inference_interval_frames"] = vision_inference_interval_frames
    pipeline_config["ai"]["hand_pose_active_step_only"] = bool(hand_pose_active_step_only)
    logger, log_path = create_operation_logger("detect", output_dir.parent / "logs")
    log_event(
        logger,
        "detect_started",
        video_path=video_path,
        model_path=model_path,
        output_dir=output_dir,
        project_id=sop_project.project_id,
        enable_hand_pose=enable_hand_pose,
        hand_pose_model_path=hand_pose_model_path,
        hand_pose_sample_interval=hand_pose_sample_interval,
        hand_pose_active_step_only=hand_pose_active_step_only,
        vision_inference_interval_frames=vision_inference_interval_frames,
        pipeline=pipeline_config,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log_event(logger, "detect_failed", reason="video_open_failed", video_path=video_path)
        close_operation_logger(logger)
        raise RuntimeError(f"视频打开失败: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log_event(logger, "video_opened", fps=fps, width=width, height=height)

    project_rois = sop_project.resolve_rois(width, height)
    log_event(
        logger,
        "project_configuration_resolved",
        project_id=sop_project.project_id,
        workflow_id=sop_project.workflow["workflow_id"],
        model_paths=model_paths,
        rois=project_rois,
    )

    # 按工作流需要懒加载视觉任务后端；只有检测任务的结果进入跟踪链。
    detector_factories = {
        task: _make_detector_factory(
            task_model_path,
            task=task,
            conf_threshold=tracking_low_confidence,
            target_classes=active_classes,
            nms_threshold=nms_threshold,
            device=inference_device,
            backend=inference_backend,
        )
        for task, task_model_path in model_paths.items()
    }
    model_router = ModelRouter(
        detector_factories=detector_factories,
        default_interval=vision_inference_interval_frames,
        task_intervals=_model_task_intervals(sop_project),
    )
    detection_fusion = DetectionFusion(iou_threshold=tracking_iou_threshold)
    pipeline_config["ai"]["router"] = model_router.metadata()
    pipeline_config["ai"]["backends"] = {}
    hand_pose_detector = None
    if enable_hand_pose:
        try:
            from scripts.hand_pose import build_hand_pose_detector
        except ModuleNotFoundError as exc:
            if exc.name == "mediapipe":
                model_router.close()
                cap.release()
                close_operation_logger(logger)
                raise ModuleNotFoundError(
                    "缺少 mediapipe。请先运行 `pip install -r requirements.txt`，"
                    "或使用 `--no-hand-pose` 关闭手部骨骼检测。"
                ) from exc
            cap.release()
            model_router.close()
            close_operation_logger(logger)
            raise

        try:
            hand_pose_detector = build_hand_pose_detector(hand_pose_model_path)
        except Exception as exc:
            logger.exception("detect_failed | hand_pose_init_failed | %s", exc)
            model_router.close()
            cap.release()
            close_operation_logger(logger)
            raise

    # SOP 状态机只依赖检测结果和 ROI，不直接依赖模型对象。
    machine = SOPStateMachine(
        workflow=sop_project.workflow,
        rois=project_rois,
        step_enabled=sop_step_enabled,
        trigger_sources=sop_trigger_sources,
        step_timeouts_sec=sop_step_timeouts_sec,
    )
    # 唯一生产跟踪链：ByteTrack两阶段匹配 + 卡尔曼运动预测。
    object_tracker = (
        ByteTrackTracker(
            iou_threshold=tracking_iou_threshold,
            max_missing_frames=tracking_max_missing_frames,
            min_confirmed_hits=tracking_min_confirmed_hits,
            high_confidence=confidence_threshold,
            low_confidence=tracking_low_confidence,
            new_track_confidence=tracking_new_track_confidence,
            second_match_iou_threshold=tracking_second_match_iou_threshold,
        )
        if model_router.available_tasks
        else None
    )
    event_engine = SopEventEngine(
        machine.rois,
        lost_tolerance_frames=event_lost_tolerance_frames,
        enter_stable_frames=event_enter_stable_frames,
        exit_stable_frames=event_exit_stable_frames,
    )
    event_history = []

    result_video_path = output_dir / RESULT_VIDEO_NAME
    result_json_path = output_dir / RESULT_JSON_NAME
    # 每个检测任务使用独立临时文件。固定的 result.inprogress.mp4 在重复点击、
    # 多服务进程或上次任务尚未释放 VideoWriter 时会产生 WinError 32。
    temporary_video_path = _build_temporary_video_path(result_video_path)
    video_written = False

    # 输出视频在读取首帧预处理结果后创建，确保裁剪/Resize 后尺寸正确。
    writer = None

    # hand_pose 字段面向接口返回：统计信息用于快速判断，samples 用于抽样复核关键点。
    hand_pose_summary = {
        "enabled": bool(enable_hand_pose),
        "model": str(hand_pose_model_path) if enable_hand_pose else None,
        "detected_frame_count": 0,
        "inference_count": 0,
        "detection_interval_frames": hand_pose_sample_interval,
        "active_step_only": bool(hand_pose_active_step_only),
        "max_hands_in_frame": 0,
        "samples": [],
    }

    frame_id = 0
    processing_failed = False
    cached_hands = []
    previous_step_key: str | None = None
    hand_pose_interval = max(1, int(hand_pose_sample_interval))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # ??????????????????????
            if writer is None and output_video:
                writer = cv2.VideoWriter(
                    str(temporary_video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    log_event(logger, "detect_failed", reason="video_writer_open_failed", output_video=result_video_path)
                    raise RuntimeError(f"输出视频创建失败: {result_video_path}")
                log_event(logger, "output_video_opened", width=frame.shape[1], height=frame.shape[0])

            # 单帧检测结果会同时用于 SOP 判定和可视化叠加。
            current_trigger = (
                machine.current_step.trigger_config
                if machine.current_step is not None
                else {}
            )
            current_step_key = machine.current_step.key if machine.current_step is not None else None
            step_changed = current_step_key != previous_step_key
            if step_changed:
                model_router.clear_cache()
                previous_step_key = current_step_key
            raw_by_task, fresh_tasks = model_router.infer(
                frame,
                current_trigger,
                frame_id,
                force=step_changed,
            )
            tracking_task = "detect" if "detect" in raw_by_task else next(iter(raw_by_task), None)
            raw_tracking = raw_by_task.get(tracking_task, []) if tracking_task else []
            if object_tracker and tracking_task:
                tracked_detections = (
                    object_tracker.update(raw_tracking)
                    if tracking_task in fresh_tasks
                    else object_tracker.predict()
                )
            else:
                tracked_detections = raw_tracking
            detections = detection_fusion.merge(
                tracked_detections,
                {
                    task: task_detections
                    for task, task_detections in raw_by_task.items()
                    if task != tracking_task
                },
            )
            time_sec = frame_id / fps
            timestamp_ms = int(time_sec * 1000)
            hands = []
            hand_pose_inferred = False
            if hand_pose_detector:
                hand_pose_active = _trigger_requires_hand_pose(current_trigger)
                should_infer_hand = (
                    _should_run_hand_pose(frame_id, hand_pose_interval)
                    and (not hand_pose_active_step_only or hand_pose_active)
                )
                if should_infer_hand:
                    cached_hands = hand_pose_detector.detect(frame, timestamp_ms)
                    hand_pose_inferred = True
                    hand_pose_summary["inference_count"] += 1
                    if cached_hands:
                        hand_pose_summary["detected_frame_count"] += 1
                elif hand_pose_active_step_only and not hand_pose_active:
                    cached_hands = []
                hands = cached_hands
            if hands:
                hand_pose_summary["max_hands_in_frame"] = max(hand_pose_summary["max_hands_in_frame"], len(hands))
                if hand_pose_inferred:
                    hand_pose_summary["samples"].append(
                        {
                            "frame": frame_id,
                            "time": round(time_sec, 3),
                            "hands": [hand.to_dict() for hand in hands],
                        }
                    )

            previous_index = machine.current_index
            frame_events = event_engine.update(detections, frame_id, time_sec)
            event_history.extend(frame_events)
            machine.update(detections, frame_id, time_sec, hands=hands, events=frame_events)
            if machine.current_index != previous_index:
                for changed_step in machine.steps[previous_index : machine.current_index]:
                    log_event(
                        logger,
                        "sop_step_completed" if changed_step.status == "done" else "sop_step_skipped",
                        step_id=changed_step.step_id,
                        key=changed_step.key,
                        frame=changed_step.frame,
                        time_sec=changed_step.time,
                        trigger_source=changed_step.trigger_source,
                    )

            # 所有可视化信息直接叠加到输出视频帧。
            draw_detections(frame, detections)
            draw_hand_landmarks(frame, hands)
            for roi_id, roi_value in machine.rois.items():
                draw_roi(frame, roi_value, roi_id.upper().replace("_", " "))
            draw_sop_status(frame, machine)
            if writer is not None:
                writer.write(frame)
            if realtime_display:
                cv2.imshow("SOP Process Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    log_event(logger, "realtime_display_stopped", frame=frame_id)
                    break

            frame_id += 1
    except Exception as exc:
        processing_failed = True
        logger.exception("detect_failed | %s", exc)
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if hand_pose_detector:
            hand_pose_detector.close()
        model_router.close()
        if realtime_display:
            cv2.destroyAllWindows()
        if processing_failed:
            # Windows 必须在 writer.release() 之后才能删除视频文件。
            try:
                temporary_video_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                log_event(logger, "temporary_video_cleanup_failed", path=temporary_video_path, error=str(cleanup_error))
            close_operation_logger(logger)

    if output_video and temporary_video_path.exists():
        # Path.replace 使用原子替换，无需先删除目标文件，缩短并发竞争窗口。
        temporary_video_path.replace(result_video_path)
        video_written = True

    result = machine.finalize(video_path.name)
    result["sop_project"] = {
        "project_id": sop_project.project_id,
        "project_dir": str(sop_project.root),
        "workflow_id": sop_project.workflow["workflow_id"],
        "workflow_version": sop_project.workflow.get("version"),
        "model_path": str(model_path) if model_path is not None else None,
        "model_paths": {task: str(path) for task, path in model_paths.items()},
        "model_manifest": sop_project.project.get("active_model_manifest"),
    }
    result["hand_pose"] = hand_pose_summary
    result["pipeline"] = pipeline_config
    result["pipeline"]["inference_counts"] = dict(model_router.inference_counts)
    result["pipeline"]["router"] = model_router.metadata()
    result["events"] = [event.to_dict() for event in event_history]
    result["output_video"] = str(result_video_path) if video_written else None
    result["output_json"] = str(result_json_path) if output_json else None
    result["log_path"] = str(log_path)
    if output_json:
        write_json_atomic(result_json_path, result)
    log_event(
        logger,
        "detect_finished",
        final_result=result["final_result"],
        reason=result["reason"],
        frame_count=frame_id,
        output_video=result["output_video"],
        output_json=result["output_json"],
        hand_pose_detected_frame_count=hand_pose_summary["detected_frame_count"],
    )
    close_operation_logger(logger)
    return result


def _build_temporary_video_path(result_video_path: Path) -> Path:
    """生成任务级唯一临时视频路径，避免检测任务之间互相抢占文件。"""

    task_id = uuid4().hex
    return result_video_path.with_name(
        f"{result_video_path.stem}.{task_id}.inprogress{result_video_path.suffix}"
    )


def _should_run_hand_pose(frame_id: int, interval: int) -> bool:
    """判断当前帧是否执行手部模型推理。"""

    if interval <= 0:
        raise ValueError("手部推理间隔必须是正整数")
    return frame_id % interval == 0


def _should_run_vision(frame_id: int, interval: int) -> bool:
    """判断当前帧是否执行视觉模型推理。"""

    if interval <= 0:
        raise ValueError("视觉模型推理间隔必须是正整数")
    return frame_id % interval == 0


def _trigger_requires_hand_pose(trigger: dict) -> bool:
    """递归判断当前步骤是否需要手部关键点结果。"""

    if not isinstance(trigger, dict):
        return False
    if trigger.get("type") == "hand_in_roi" or trigger.get("allow_hand_pose") is True:
        return True
    if any(_trigger_requires_hand_pose(item) for item in trigger.get("conditions", [])):
        return True
    nested = trigger.get("condition")
    return isinstance(nested, dict) and _trigger_requires_hand_pose(nested)


def _vision_tasks_for_trigger(trigger: dict) -> set[str]:
    """Return model tasks required by one active workflow trigger."""

    if not isinstance(trigger, dict):
        return {"detect"}
    evidence = trigger.get("evidence")
    evidence_type = str(evidence.get("type", "bbox")) if isinstance(evidence, dict) else "bbox"
    trigger_type = str(trigger.get("type", "")).strip().lower()
    if evidence_type == "mask":
        tasks = {"segment"}
    elif evidence_type == "keypoints":
        tasks = {"pose"}
    elif trigger_type in {"composite", "duration", "hand_in_roi"}:
        tasks = set()
    else:
        tasks = {"detect"}
    for condition in trigger.get("conditions", []):
        tasks.update(_vision_tasks_for_trigger(condition))
    nested = trigger.get("condition")
    if isinstance(nested, dict):
        tasks.update(_vision_tasks_for_trigger(nested))
    if trigger.get("type") == "hand_in_roi":
        tasks.discard("detect")
    return tasks


def _model_task_intervals(sop_project) -> dict[str, int]:
    """读取项目级任务推理间隔；未声明时由全局间隔控制。"""

    intervals: dict[str, int] = {}
    for task, profile in sop_project.model_profiles.items():
        if not isinstance(profile, dict):
            continue
        value = profile.get("inference_interval_frames")
        if isinstance(value, int) and value > 0:
            intervals[str(task)] = value
    return intervals


def _make_detector_factory(model_path: Path, **kwargs):
    """创建延迟构造单个视觉后端的工厂。"""

    def create_detector():
        return build_detector(model_path, **kwargs)

    return create_detector


def _build_pipeline_config(
    enable_yolo: bool,
    confidence_threshold: float,
    nms_threshold: float,
    inference_device: str | int | None,
    inference_backend: str,
    vision_task: str,
    vision_tasks: list[str],
    target_classes: list[str],
    sop_step_enabled: dict[str, bool] | None,
    sop_trigger_sources: dict[str, str] | None,
    sop_step_timeouts_sec: dict[str, float] | None,
    output_video: bool,
    output_json: bool,
    realtime_display: bool,
    tracking_iou_threshold: float,
    tracking_max_missing_frames: int,
    tracking_min_confirmed_hits: int,
    tracking_low_confidence: float,
    tracking_new_track_confidence: float,
    tracking_second_match_iou_threshold: float,
    event_lost_tolerance_frames: int,
    event_enter_stable_frames: int,
    event_exit_stable_frames: int,
) -> dict:
    """生成本次任务的统一流水线配置，供日志和结果 JSON 共同使用。"""

    return {
        "ai": {
            "enable_yolo": enable_yolo,
            "confidence_threshold": confidence_threshold,
            "nms_threshold": nms_threshold,
            "inference_device": inference_device,
            "backend": inference_backend,
            "task": vision_task,
            "tasks": vision_tasks,
            "target_classes": target_classes,
        },
        "sop": {
            "step_enabled": sop_step_enabled or {},
            "trigger_sources": sop_trigger_sources or {},
            "step_timeouts_sec": sop_step_timeouts_sec or {},
        },
        "tracking": {
            "backend": "bytetrack",
            "iou_threshold": tracking_iou_threshold,
            "max_missing_frames": tracking_max_missing_frames,
            "min_confirmed_hits": tracking_min_confirmed_hits,
            "high_confidence": confidence_threshold,
            "low_confidence": tracking_low_confidence,
            "new_track_confidence": tracking_new_track_confidence,
            "second_match_iou_threshold": tracking_second_match_iou_threshold,
            "motion_prediction": True,
            "event_lost_tolerance_frames": event_lost_tolerance_frames,
            "event_enter_stable_frames": event_enter_stable_frames,
            "event_exit_stable_frames": event_exit_stable_frames,
        },
        "output": {
            "output_video": output_video,
            "output_json": output_json,
            "realtime_display": realtime_display,
        },
    }
