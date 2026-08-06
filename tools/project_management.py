"""SOP项目的创建、读取、修改、激活和完整性检查。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.project import load_sop_project
from scripts.project.schema_validator import validate_project, validate_rois, validate_workflow
from scripts.project.storage import read_json_object, resolve_within, utc_now, validate_safe_name, write_json_atomic


def create_sop_project(
    projects_root: str | Path,
    project_id: str,
    name: str,
    classes: list[dict[str, Any]],
    *,
    description: str = "",
) -> dict[str, Any]:
    """创建可继续配置的 SOP 草稿项目。"""

    validate_safe_name(project_id, "project_id")
    root = Path(projects_root).expanduser().resolve()
    project_dir = root / project_id
    if project_dir.exists():
        raise FileExistsError(f"SOP 项目已经存在: {project_dir}")
    project = {
        "schema_version": "1.0",
        "project_id": project_id,
        "name": name,
        "description": description,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "draft",
        "classes": classes,
        "active_model": None,
        "active_workflow": "workflow.json",
        "active_roi_profile": "rois.json",
    }
    validate_project(project)
    project_dir.mkdir(parents=True)
    write_json_atomic(project_dir / "project.json", project)
    write_json_atomic(
        project_dir / "rois.json",
        {"schema_version": "1.0", "coordinate_type": "normalized", "regions": []},
    )
    write_json_atomic(
        project_dir / "workflow.json",
        {
            "schema_version": "1.0",
            "workflow_id": f"{project_id}_V1",
            "name": f"{name}流程",
            "version": "1.0.0",
            "steps": [],
        },
    )
    for directory in ("source_videos", "frames", "annotations", "dataset", "models", "runs", "outputs"):
        (project_dir / directory).mkdir()
    return get_sop_project(project_dir)


def list_sop_projects(projects_root: str | Path) -> dict[str, Any]:
    root = Path(projects_root).expanduser().resolve()
    projects: list[dict[str, Any]] = []
    if root.is_dir():
        for project_file in sorted(root.glob("*/project.json")):
            try:
                data = read_json_object(project_file, description="项目配置文件")
                readiness = _project_readiness(project_file.parent)
                projects.append(
                    {
                        "project_id": data.get("project_id"),
                        "name": data.get("name"),
                        "description": data.get("description", ""),
                        "status": data.get("status", "draft"),
                        "project_dir": str(project_file.parent.resolve()),
                        "class_count": len(data.get("classes", [])),
                        "ready": readiness["ready"],
                        "errors": readiness["errors"],
                    }
                )
            except Exception as exc:
                projects.append(
                    {
                        "project_id": project_file.parent.name,
                        "project_dir": str(project_file.parent.resolve()),
                        "status": "invalid",
                        "ready": False,
                        "errors": [str(exc)],
                    }
                )
    return {"projects_root": str(root), "project_count": len(projects), "projects": projects}


def get_sop_project(project_dir: str | Path) -> dict[str, Any]:
    root = Path(project_dir).expanduser().resolve()
    project = read_json_object(root / "project.json", description="项目配置文件")
    rois = read_json_object(
        resolve_within(root, str(project.get("active_roi_profile", "rois.json")), field="active_roi_profile"),
        description="ROI 配置文件",
    )
    workflow = read_json_object(
        resolve_within(root, str(project.get("active_workflow", "workflow.json")), field="active_workflow"),
        description="流程配置文件",
    )
    readiness = _project_readiness(root)
    return {
        "project_dir": str(root),
        "project": project,
        "rois": rois,
        "workflow": workflow,
        "ready": readiness["ready"],
        "validation_errors": readiness["errors"],
    }


def update_project_metadata(
    project_dir: str | Path,
    *,
    name: str | None = None,
    description: str | None = None,
    classes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(project_dir).expanduser().resolve()
    project_path = root / "project.json"
    project = read_json_object(project_path, description="项目配置文件")
    if name is not None:
        project["name"] = name
    if description is not None:
        project["description"] = description
    if classes is not None:
        if (root / "frames_manifest.json").exists() or _directory_has_files(root / "dataset"):
            raise ValueError("项目已经包含抽帧或数据集，不能再修改类别定义")
        project["classes"] = classes
    project["updated_at"] = utc_now()
    project["status"] = "draft"
    validate_project(project)
    write_json_atomic(project_path, project)
    return get_sop_project(root)


def save_project_rois(project_dir: str | Path, roi_config: dict[str, Any]) -> dict[str, Any]:
    root = Path(project_dir).expanduser().resolve()
    project = read_json_object(root / "project.json", description="项目配置文件")
    validate_rois(roi_config)
    write_json_atomic(
        resolve_within(root, str(project.get("active_roi_profile", "rois.json")), field="active_roi_profile"),
        roi_config,
    )
    _mark_draft(root, project)
    return get_sop_project(root)


def save_project_workflow(project_dir: str | Path, workflow: dict[str, Any]) -> dict[str, Any]:
    root = Path(project_dir).expanduser().resolve()
    project = read_json_object(root / "project.json", description="项目配置文件")
    rois = read_json_object(
        resolve_within(root, str(project.get("active_roi_profile", "rois.json")), field="active_roi_profile"),
        description="ROI 配置文件",
    )
    validate_project(project)
    validate_rois(rois)
    validate_workflow(
        workflow,
        {str(item["name"]) for item in project["classes"]},
        {str(item["id"]) for item in rois["regions"]},
    )
    write_json_atomic(
        resolve_within(root, str(project.get("active_workflow", "workflow.json")), field="active_workflow"),
        workflow,
    )
    _mark_draft(root, project)
    return get_sop_project(root)


def activate_sop_project(project_dir: str | Path) -> dict[str, Any]:
    project = load_sop_project(project_dir)
    project_data = dict(project.project)
    project_data["status"] = "active"
    project_data["updated_at"] = utc_now()
    write_json_atomic(project.root / "project.json", project_data)
    return get_sop_project(project.root)


def _project_readiness(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        load_sop_project(root)
    except Exception as exc:
        errors.append(str(exc))
    return {"ready": not errors, "errors": errors}


def _mark_draft(root: Path, project: dict[str, Any]) -> None:
    updated = dict(project)
    updated["status"] = "draft"
    updated["updated_at"] = utc_now()
    write_json_atomic(root / "project.json", updated)


def _directory_has_files(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))
