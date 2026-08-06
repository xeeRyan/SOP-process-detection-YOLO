"""项目、ROI与工作流配置的结构和交叉引用校验。"""

from __future__ import annotations

from typing import Any


SUPPORTED_TRIGGER_TYPES = {
    "object_in_roi",
    "object_present",
    "hand_in_roi",
    "object_event",
    "composite",
    "object_count",
    "object_transition",
    "duration",
}
SUPPORTED_OBJECT_EVENTS = {
    "object_appear",
    "object_disappear",
    "object_enter_roi",
    "object_exit_roi",
    "object_move_roi",
}


class ProjectValidationError(ValueError):
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
            _fail("类别id必须是非负整数", f"{field}.id")
        name = _require_text(item, "name", f"{field}.name")
        if class_id in ids:
            _fail(f"类别id重复: {class_id}", f"{field}.id")
        if name in names:
            _fail(f"类别名称重复: {name}", f"{field}.name")
        ids.add(class_id)
        names.add(name)


def validate_rois(rois: dict[str, Any]) -> None:
    coordinate_type = rois.get("coordinate_type", "normalized")
    if coordinate_type not in {"normalized", "pixel"}:
        _fail("coordinate_type只支持normalized或pixel", "coordinate_type")
    regions = rois.get("regions")
    if not isinstance(regions, list) or not regions:
        _fail("至少需要定义一个ROI", "regions")

    ids: set[str] = set()
    for index, region in enumerate(regions):
        field = f"regions[{index}]"
        if not isinstance(region, dict):
            _fail("ROI必须是对象", field)
        roi_id = _require_text(region, "id", f"{field}.id")
        if roi_id in ids:
            _fail(f"ROI id重复: {roi_id}", f"{field}.id")
        if region.get("shape", "rectangle") != "rectangle":
            _fail("当前只支持rectangle ROI", f"{field}.shape")
        points = region.get("points")
        if not isinstance(points, list) or len(points) != 2:
            _fail("矩形ROI必须包含左上、右下两个点", f"{field}.points")
        for point_index, point in enumerate(points):
            if (
                not isinstance(point, list)
                or len(point) != 2
                or not all(isinstance(value, (int, float)) for value in point)
            ):
                _fail("ROI点必须是[x, y]数值数组", f"{field}.points[{point_index}]")
            if coordinate_type == "normalized" and not all(
                0 <= float(value) <= 1 for value in point
            ):
                _fail("归一化ROI坐标必须位于0到1", f"{field}.points[{point_index}]")
        if points[0][0] >= points[1][0] or points[0][1] >= points[1][1]:
            _fail("ROI左上角必须位于右下角之前", f"{field}.points")
        ids.add(roi_id)


def validate_workflow(
    workflow: dict[str, Any],
    class_names: set[str],
    roi_ids: set[str],
) -> None:
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
            _fail("步骤id不能为空", f"{field}.id")
        order = step.get("order")
        if not isinstance(order, int) or order <= 0:
            _fail("步骤order必须是正整数", f"{field}.order")
        if step_id in step_ids:
            _fail(f"步骤id重复: {step_id}", f"{field}.id")
        if order in orders:
            _fail(f"步骤顺序重复: {order}", f"{field}.order")
        _require_text(step, "name", f"{field}.name")
        trigger = step.get("trigger")
        if not isinstance(trigger, dict):
            _fail("步骤必须定义trigger", f"{field}.trigger")
        _validate_trigger(trigger, class_names, roi_ids, f"{field}.trigger")
        required = step.get("required", True)
        if not isinstance(required, bool):
            _fail("required必须是布尔值", f"{field}.required")
        timeout = step.get("timeout_sec")
        if timeout is not None and (
            not isinstance(timeout, (int, float)) or timeout <= 0
        ):
            _fail("timeout_sec必须大于0", f"{field}.timeout_sec")
        step_ids.add(step_id)
        orders.add(order)


def _validate_trigger(
    trigger: dict[str, Any],
    class_names: set[str],
    roi_ids: set[str],
    field: str,
) -> None:
    trigger_type = trigger.get("type")
    if trigger_type not in SUPPORTED_TRIGGER_TYPES:
        _fail(f"不支持的触发类型: {trigger_type}", f"{field}.type")

    confidence = trigger.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        _fail("confidence必须位于0到1", f"{field}.confidence")
    stable_frames = trigger.get("stable_frames", 3)
    if not isinstance(stable_frames, int) or stable_frames <= 0:
        _fail("stable_frames必须是正整数", f"{field}.stable_frames")

    if trigger_type == "composite":
        if trigger.get("operator", "all") not in {"all", "any"}:
            _fail("组合条件operator只支持all或any", f"{field}.operator")
        conditions = trigger.get("conditions")
        if not isinstance(conditions, list) or len(conditions) < 2:
            _fail("组合条件至少需要两个子条件", f"{field}.conditions")
        for index, condition in enumerate(conditions):
            if not isinstance(condition, dict):
                _fail("子条件必须是对象", f"{field}.conditions[{index}]")
            _validate_trigger(
                condition,
                class_names,
                roi_ids,
                f"{field}.conditions[{index}]",
            )
        return

    if trigger_type == "duration":
        duration_sec = trigger.get("duration_sec")
        if not isinstance(duration_sec, (int, float)) or duration_sec <= 0:
            _fail("duration_sec必须大于0", f"{field}.duration_sec")
        condition = trigger.get("condition")
        if not isinstance(condition, dict):
            _fail("持续时间触发器必须定义condition", f"{field}.condition")
        _validate_trigger(condition, class_names, roi_ids, f"{field}.condition")
        return

    if trigger_type == "object_transition":
        _validate_class_reference(trigger, class_names, field)
        from_roi = _require_text(trigger, "from_roi_id", f"{field}.from_roi_id")
        to_roi = _require_text(trigger, "to_roi_id", f"{field}.to_roi_id")
        if from_roi not in roi_ids or to_roi not in roi_ids:
            _fail("跨ROI移动引用了不存在的ROI", field)
        if from_roi == to_roi:
            _fail("from_roi_id与to_roi_id不能相同", field)
        return

    if trigger_type == "object_count":
        _validate_class_reference(trigger, class_names, field)
        if trigger.get("roi_id"):
            _validate_roi_reference(trigger, roi_ids, field)
        minimum = trigger.get("min_count", trigger.get("count", 1))
        maximum = trigger.get("max_count")
        if not isinstance(minimum, int) or minimum < 0:
            _fail("min_count必须是非负整数", f"{field}.min_count")
        if maximum is not None and (
            not isinstance(maximum, int) or maximum < minimum
        ):
            _fail("max_count必须是不小于min_count的整数", f"{field}.max_count")
        return

    if trigger_type == "object_event":
        event_type = trigger.get("event")
        if event_type not in SUPPORTED_OBJECT_EVENTS:
            _fail(f"不支持的对象事件: {event_type}", f"{field}.event")
        if trigger.get("class_name"):
            _validate_class_reference(trigger, class_names, field)
        if trigger.get("roi_id"):
            _validate_roi_reference(trigger, roi_ids, field)
        return

    event_type = trigger.get("event")
    if event_type is not None and event_type not in SUPPORTED_OBJECT_EVENTS:
        _fail(f"不支持的对象事件: {event_type}", f"{field}.event")
    require_transition = trigger.get("require_transition", False)
    if not isinstance(require_transition, bool):
        _fail("require_transition必须是布尔值", f"{field}.require_transition")
    if trigger_type != "hand_in_roi":
        _validate_class_reference(trigger, class_names, field)
    if trigger_type in {"object_in_roi", "hand_in_roi"}:
        _validate_roi_reference(trigger, roi_ids, field)


def _validate_class_reference(
    trigger: dict[str, Any],
    class_names: set[str],
    field: str,
) -> None:
    class_name = _require_text(trigger, "class_name", f"{field}.class_name")
    if class_name not in class_names:
        _fail(f"步骤引用了未定义类别: {class_name}", f"{field}.class_name")


def _validate_roi_reference(
    trigger: dict[str, Any],
    roi_ids: set[str],
    field: str,
) -> None:
    roi_id = _require_text(trigger, "roi_id", f"{field}.roi_id")
    if roi_id not in roi_ids:
        _fail(f"步骤引用了不存在的ROI: {roi_id}", f"{field}.roi_id")


def _require_text(data: dict[str, Any], key: str, field: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        _fail(f"{key}不能为空", field)
    return value.strip()


def _fail(message: str, field: str) -> None:
    raise ProjectValidationError(message, field)
