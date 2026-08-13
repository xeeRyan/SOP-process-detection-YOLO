"""加载并规范化SOP项目配置，向运行流程提供项目对象。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.project.schema_validator import (
    ProjectValidationError,
    validate_project,
    validate_rois,
    validate_workflow,
)
from scripts.project.storage import read_json_object, resolve_within


@dataclass(frozen=True)
class SopProject:
    root: Path
    project: dict[str, Any]
    workflow: dict[str, Any]
    roi_config: dict[str, Any]

    @property
    def project_id(self) -> str:
        return str(self.project["project_id"])

    @property
    def class_names(self) -> list[str]:
        return [str(item["name"]) for item in self.project["classes"]]

    @property
    def active_model_task(self) -> str:
        """Return the task implemented by the active project model."""

        return str(self.project.get("active_model_task", "detect")).strip().lower()

    @property
    def model_profiles(self) -> dict[str, dict[str, Any]]:
        """Return task-specific model profiles declared by the project."""

        profiles = self.project.get("model_profiles", {})
        return profiles if isinstance(profiles, dict) else {}

    @property
    def required_model_tasks(self) -> set[str]:
        """Infer model tasks required by workflow evidence declarations."""

        tasks: set[str] = set()

        def visit(trigger: dict[str, Any]) -> None:
            evidence = trigger.get("evidence")
            evidence_type = str(evidence.get("type", "bbox")) if isinstance(evidence, dict) else "bbox"
            if evidence_type == "mask":
                tasks.add("segment")
            elif evidence_type == "keypoints":
                tasks.add("pose")
            elif trigger.get("type") != "hand_in_roi":
                tasks.add("detect")
            for condition in trigger.get("conditions", []):
                if isinstance(condition, dict):
                    visit(condition)
            nested = trigger.get("condition")
            if isinstance(nested, dict):
                visit(nested)

        for step in self.workflow.get("steps", []):
            trigger = step.get("trigger", {})
            if isinstance(trigger, dict):
                visit(trigger)
        return tasks or {"detect"}

    def resolve_model_path(self, task: str) -> Path:
        """Resolve the model used for a concrete vision task."""

        normalized_task = str(task).strip().lower()
        profile = self.model_profiles.get(normalized_task)
        if isinstance(profile, dict) and profile.get("path"):
            return resolve_within(
                self.root,
                str(profile["path"]),
                field=f"model_profiles.{normalized_task}.path",
            )
        if normalized_task == self.active_model_task:
            return self.active_model_path
        raise ValueError(
            f"SOP项目 {self.project_id} 未配置 {normalized_task} 模型，"
            "请在 project.json 的 model_profiles 中补充路径"
        )

    @property
    def active_model_path(self) -> Path:
        """Return the active model path declared by this project.

        Model paths are resolved through the same project-local seam as the
        workflow and ROI files.  This prevents a project from silently using
        a model belonging to another project or the old global demo config.
        """

        active_model = self.project.get("active_model")
        if not isinstance(active_model, str) or not active_model.strip():
            raise ValueError(
                f"SOP项目未配置活动模型: {self.project_id}; "
                "请先训练并激活项目模型，或在请求中显式传入 model_path"
            )
        return resolve_within(self.root, active_model.strip(), field="active_model")

    def resolve_rois(self, frame_width: int, frame_height: int) -> dict[str, list[int]]:
        """将项目 ROI 转换为当前视频的像素坐标。"""

        normalized = self.roi_config.get("coordinate_type", "normalized") == "normalized"
        result: dict[str, list[int]] = {}
        for region in self.roi_config["regions"]:
            (x1, y1), (x2, y2) = region["points"]
            if normalized:
                result[region["id"]] = [
                    round(float(x1) * frame_width),
                    round(float(y1) * frame_height),
                    round(float(x2) * frame_width),
                    round(float(y2) * frame_height),
                ]
            else:
                result[region["id"]] = [round(x1), round(y1), round(x2), round(y2)]
        return result


def load_sop_project(project_dir: str | Path) -> SopProject:
    root = Path(project_dir).expanduser().resolve()
    project = read_json_object(root / "project.json", description="SOP 项目配置文件")
    workflow_path = resolve_within(
        root, str(project.get("active_workflow", "workflow.json")), field="active_workflow"
    )
    roi_path = resolve_within(
        root, str(project.get("active_roi_profile", "rois.json")), field="active_roi_profile"
    )
    workflow = read_json_object(workflow_path, description="SOP 流程配置文件")
    roi_config = read_json_object(roi_path, description="SOP ROI 配置文件")

    validate_project(project)
    active_model = project.get("active_model")
    if active_model not in (None, ""):
        if not isinstance(active_model, str):
            raise ValueError("active_model必须是项目内相对路径或为空")
        resolve_within(root, active_model.strip(), field="active_model")
    for task, profile in (project.get("model_profiles") or {}).items():
        if not isinstance(profile, dict) or not isinstance(profile.get("path"), str):
            raise ValueError(f"model_profiles.{task} 必须包含 path 字符串")
        resolve_within(root, profile["path"], field=f"model_profiles.{task}.path")
    validate_rois(roi_config)
    class_names = {str(item["name"]) for item in project["classes"]}
    roi_ids = {str(item["id"]) for item in roi_config["regions"]}
    validate_workflow(workflow, class_names, roi_ids)
    available_tasks = {str(project.get("active_model_task", "detect"))}
    available_tasks.update(str(task) for task in (project.get("model_profiles") or {}))
    _validate_workflow_evidence(workflow, available_tasks)
    return SopProject(root=root, project=project, workflow=workflow, roi_config=roi_config)


def _validate_workflow_evidence(workflow: dict[str, Any], available_tasks: set[str]) -> None:
    """Reject evidence that the active model cannot produce."""

    required_task_by_evidence = {"mask": "segment", "keypoints": "pose"}

    def visit(trigger: dict[str, Any], field: str) -> None:
        evidence = trigger.get("evidence")
        if isinstance(evidence, dict):
            evidence_type = str(evidence.get("type", "bbox"))
            required_task = required_task_by_evidence.get(evidence_type)
            if required_task is not None and required_task not in available_tasks:
                raise ProjectValidationError(
                    f"evidence.type={evidence_type}需要{required_task}模型，项目未配置该任务模型",
                    field,
                )
        for index, condition in enumerate(trigger.get("conditions", [])):
            if isinstance(condition, dict):
                visit(condition, f"{field}.conditions[{index}]")
        nested = trigger.get("condition")
        if isinstance(nested, dict):
            visit(nested, f"{field}.condition")

    for index, step in enumerate(workflow.get("steps", [])):
        trigger = step.get("trigger", {})
        if isinstance(trigger, dict):
            visit(trigger, f"steps[{index}].trigger")
