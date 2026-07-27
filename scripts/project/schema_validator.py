from __future__ import annotations

from typing import Any


SUPPORTED_TRIGGER_TYPES = {"object_in_roi", "object_present", "hand_in_roi"}


class ProjectValidationError(ValueError):
    """SOP 项目配置不合法。"""

    def __init__(self, message: str, field: str = "") -> None:
        super().__init__(f"{message}（字段: {field}）" if field else message)
        self.field = field


def validate_project(project: dict[str, Any]) -> None:
    _require_text(project, "project_id", "project_id")
    _require_text(project, "name", "name")
    classes = project.get("classes")
    if not isinstance(classes, list) or not classes:
        _fail("项目至少需要定义一个识别类别", "classes")

    ids: set[int] = set()
    names: set[str] = set()
    for index, item in enumerate(classes):
        field = f"classes[{index}]"
        if not isinstance(item, dict):
            _fail("类别必须是对象", field)
        class_id = item.get("id")
        if not isinstance(class_id, int) or class_id < 0:
            _fail("类别 id 必须是非负整数", f"{field}.id")
        name = _require_text(item, "name", f"{field}.name")
        if class_id in ids:
            _fail(f"类别 id 重复: {class_id}", f"{field}.id")
        if name in names:
            _fail(f"类别名称重复: {name}", f"{field}.name")
        ids.add(class_id)
        names.add(name)


def validate_rois(rois: dict[str, Any]) -> None:
    coordinate_type = rois.get("coordinate_type", "normalized")
    if coordinate_type not in {"normalized", "pixel"}:
        _fail("coordinate_type 只支持 normalized 或 pixel", "coordinate_type")
    regions = rois.get("regions")
    if not isinstance(regions, list) or not regions:
        _fail("至少需要定义一个 ROI", "regions")

    ids: set[str] = set()
    for index, region in enumerate(regions):
        field = f"regions[{index}]"
        if not isinstance(region, dict):
            _fail("ROI 必须是对象", field)
        roi_id = _require_text(region, "id", f"{field}.id")
        if roi_id in ids:
            _fail(f"ROI id 重复: {roi_id}", f"{field}.id")
        if region.get("shape", "rectangle") != "rectangle":
            _fail("第一版只支持 rectangle ROI", f"{field}.shape")
        points = region.get("points")
        if not isinstance(points, list) or len(points) != 2:
            _fail("矩形 ROI 必须包含左上、右下两个点", f"{field}.points")
        for point_index, point in enumerate(points):
            if not isinstance(point, list) or len(point) != 2 or not all(isinstance(value, (int, float)) for value in point):
                _fail("ROI 点必须是 [x, y] 数值数组", f"{field}.points[{point_index}]")
            if coordinate_type == "normalized" and not all(0 <= float(value) <= 1 for value in point):
                _fail("归一化 ROI 坐标必须位于 0 到 1", f"{field}.points[{point_index}]")
        if points[0][0] >= points[1][0] or points[0][1] >= points[1][1]:
            _fail("ROI 左上角必须位于右下角之前", f"{field}.points")
        ids.add(roi_id)


def validate_workflow(workflow: dict[str, Any], class_names: set[str], roi_ids: set[str]) -> None:
    _require_text(workflow, "workflow_id", "workflow_id")
    steps = workflow.get("steps")
    if not isinstance(steps, list) or not steps:
        _fail("流程至少需要定义一个步骤", "steps")

    step_ids: set[str] = set()
    orders: set[int] = set()
    for index, step in enumerate(steps):
        field = f"steps[{index}]"
        if not isinstance(step, dict):
            _fail("步骤必须是对象", field)
        step_id = str(step.get("id", "")).strip()
        if not step_id:
            _fail("步骤 id 不能为空", f"{field}.id")
        order = step.get("order")
        if not isinstance(order, int) or order <= 0:
            _fail("步骤 order 必须是正整数", f"{field}.order")
        if step_id in step_ids:
            _fail(f"步骤 id 重复: {step_id}", f"{field}.id")
        if order in orders:
            _fail(f"步骤顺序重复: {order}", f"{field}.order")
        _require_text(step, "name", f"{field}.name")
        trigger = step.get("trigger")
        if not isinstance(trigger, dict):
            _fail("步骤必须定义 trigger", f"{field}.trigger")
        trigger_type = trigger.get("type")
        if trigger_type not in SUPPORTED_TRIGGER_TYPES:
            _fail(f"不支持的触发类型: {trigger_type}", f"{field}.trigger.type")
        if trigger_type != "hand_in_roi":
            class_name = _require_text(trigger, "class_name", f"{field}.trigger.class_name")
            if class_name not in class_names:
                _fail(f"步骤引用了未定义类别: {class_name}", f"{field}.trigger.class_name")
        if trigger_type in {"object_in_roi", "hand_in_roi"}:
            roi_id = _require_text(trigger, "roi_id", f"{field}.trigger.roi_id")
            if roi_id not in roi_ids:
                _fail(f"步骤引用了不存在的 ROI: {roi_id}", f"{field}.trigger.roi_id")
        confidence = trigger.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            _fail("confidence 必须位于 0 到 1", f"{field}.trigger.confidence")
        stable_frames = trigger.get("stable_frames", 3)
        if not isinstance(stable_frames, int) or stable_frames <= 0:
            _fail("stable_frames 必须是正整数", f"{field}.trigger.stable_frames")
        timeout = step.get("timeout_sec")
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
            _fail("timeout_sec 必须大于 0", f"{field}.timeout_sec")
        step_ids.add(step_id)
        orders.add(order)


def _require_text(data: dict[str, Any], key: str, field: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        _fail(f"{key} 不能为空", field)
    return value.strip()


def _fail(message: str, field: str) -> None:
    raise ProjectValidationError(message, field)
