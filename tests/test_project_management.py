from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from task_dispatcher import run_task
from tools.project_management import create_sop_project, list_sop_projects


class ProjectManagementTests(unittest.TestCase):
    CLASSES = [{"id": 0, "name": "part", "display_name": "零件"}]

    def test_create_configure_and_activate_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            created = create_sop_project(
                temp_dir,
                "SOP_TEST",
                "测试流程",
                self.CLASSES,
                description="项目管理测试",
            )
            project_dir = Path(created["project_dir"])
            self.assertFalse(created["ready"])
            self.assertEqual(created["project"]["status"], "draft")

            roi_result = run_task(
                {
                    "command": "save_rois",
                    "params": {
                        "sop_project_dir": str(project_dir),
                        "rois": {
                            "schema_version": "1.0",
                            "coordinate_type": "normalized",
                            "regions": [
                                {
                                    "id": "work",
                                    "name": "工作区",
                                    "shape": "rectangle",
                                    "points": [[0.1, 0.1], [0.9, 0.9]],
                                }
                            ],
                        },
                    },
                }
            )
            self.assertFalse(roi_result["ready"])

            workflow_result = run_task(
                {
                    "command": "save_workflow",
                    "params": {
                        "sop_project_dir": str(project_dir),
                        "workflow": {
                            "schema_version": "1.0",
                            "workflow_id": "SOP_TEST_V1",
                            "name": "测试流程",
                            "version": "1.0.0",
                            "steps": [
                                {
                                    "id": "put_part",
                                    "order": 1,
                                    "name": "放置零件",
                                    "required": True,
                                    "trigger": {
                                        "type": "object_in_roi",
                                        "class_name": "part",
                                        "roi_id": "work",
                                        "confidence": 0.4,
                                        "stable_frames": 3,
                                    },
                                }
                            ],
                        },
                    },
                }
            )
            self.assertTrue(workflow_result["ready"])
            self.assertEqual(workflow_result["project"]["status"], "draft")

            activated = run_task(
                {
                    "command": "activate_project",
                    "params": {"sop_project_dir": str(project_dir)},
                }
            )
            self.assertTrue(activated["ready"])
            self.assertEqual(activated["project"]["status"], "active")

            listed = list_sop_projects(temp_dir)
            self.assertEqual(listed["project_count"], 1)
            self.assertTrue(listed["projects"][0]["ready"])

    def test_workflow_rejects_unknown_class_and_roi(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            created = create_sop_project(temp_dir, "SOP_TEST", "测试流程", self.CLASSES)
            project_dir = Path(created["project_dir"])
            run_task(
                {
                    "command": "save_rois",
                    "params": {
                        "sop_project_dir": str(project_dir),
                        "rois": {
                            "coordinate_type": "normalized",
                            "regions": [
                                {"id": "work", "shape": "rectangle", "points": [[0.1, 0.1], [0.9, 0.9]]}
                            ],
                        },
                    },
                }
            )
            with self.assertRaisesRegex(ValueError, "未定义类别"):
                run_task(
                    {
                        "command": "save_workflow",
                        "params": {
                            "sop_project_dir": str(project_dir),
                            "workflow": {
                                "workflow_id": "SOP_TEST_V1",
                                "steps": [
                                    {
                                        "id": "bad",
                                        "order": 1,
                                        "name": "错误步骤",
                                        "trigger": {
                                            "type": "object_in_roi",
                                            "class_name": "unknown",
                                            "roi_id": "missing",
                                        },
                                    }
                                ],
                            },
                        },
                    }
                )

    def test_classes_lock_after_frames_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            created = create_sop_project(temp_dir, "SOP_TEST", "测试流程", self.CLASSES)
            project_dir = Path(created["project_dir"])
            (project_dir / "frames_manifest.json").write_text(
                json.dumps({"videos": {}}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "不能再修改类别"):
                run_task(
                    {
                        "command": "update_project",
                        "params": {
                            "sop_project_dir": str(project_dir),
                            "classes": [{"id": 0, "name": "changed"}],
                        },
                    }
                )
