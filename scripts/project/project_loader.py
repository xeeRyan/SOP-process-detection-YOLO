"""加载并规范化SOP项目配置，向运行流程提供项目对象。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.project.schema_validator import validate_project, validate_rois, validate_workflow
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
    validate_rois(roi_config)
    class_names = {str(item["name"]) for item in project["classes"]}
    roi_ids = {str(item["id"]) for item in roi_config["regions"]}
    validate_workflow(workflow, class_names, roi_ids)
    return SopProject(root=root, project=project, workflow=workflow, roi_config=roi_config)
