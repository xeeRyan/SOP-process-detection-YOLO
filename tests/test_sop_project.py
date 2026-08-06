from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.legacy_sk_config import DEFAULT_SOP_PROJECT_DIR
from scripts.detector import Detection
from scripts.project import ProjectValidationError, load_sop_project
from scripts.sop_logic import SOPStateMachine


class SopProjectTests(unittest.TestCase):
    def test_load_sk_demo_and_resolve_normalized_rois(self) -> None:
        project = load_sop_project(DEFAULT_SOP_PROJECT_DIR)

        self.assertEqual(project.project_id, "SK_DEMO")
        self.assertEqual(project.class_names, ["bearing", "cover", "tool"])
        self.assertEqual(len(project.workflow["steps"]), 4)
        self.assertEqual(project.resolve_rois(2448, 2048)["work"], [1115, 1132, 1648, 1522])

    def test_workflow_drives_state_machine(self) -> None:
        project = load_sop_project(DEFAULT_SOP_PROJECT_DIR)
        machine = SOPStateMachine(
            workflow=project.workflow,
            rois=project.resolve_rois(2448, 2048),
        )

        samples = [
            Detection("bearing", 0.9, [1200, 1200, 1300, 1300]),
            Detection("cover", 0.9, [1200, 1200, 1300, 1300]),
            Detection("tool", 0.9, [1150, 600, 1250, 700]),
            Detection("tool", 0.9, [850, 600, 950, 700]),
        ]
        frame = 0
        for step_index, detection in enumerate(samples):
            stable_frames = 10 if step_index == 3 else 3
            for _ in range(stable_frames):
                frame += 1
                machine.update([detection], frame, frame / 25)

        self.assertEqual(machine.final_result, "OK")
        self.assertTrue(all(step.status == "done" for step in machine.steps))

    def test_optional_step_does_not_block_later_required_step(self) -> None:
        workflow = {
            "steps": [
                {
                    "id": "optional_tool",
                    "order": 1,
                    "name": "可选工具检查",
                    "required": False,
                    "trigger": {
                        "type": "object_in_roi",
                        "class_name": "tool",
                        "roi_id": "work",
                        "stable_frames": 1,
                    },
                },
                {
                    "id": "bearing",
                    "order": 2,
                    "name": "放入轴承",
                    "required": True,
                    "trigger": {
                        "type": "object_in_roi",
                        "class_name": "bearing",
                        "roi_id": "work",
                        "stable_frames": 1,
                    },
                },
            ]
        }
        machine = SOPStateMachine(workflow=workflow, rois={"work": [0, 0, 100, 100]})

        machine.update(
            [Detection(class_name="bearing", conf=0.9, bbox=[10, 10, 30, 30])],
            frame_id=1,
            time_sec=0.1,
        )
        result = machine.finalize("optional.mp4")

        self.assertEqual(result["final_result"], "OK")
        self.assertEqual(result["steps"][0]["status"], "skipped")
        self.assertEqual(result["steps"][1]["status"], "done")
        self.assertFalse(result["steps"][0]["required"])

    def test_invalid_class_reference_has_field(self) -> None:
        source = Path(DEFAULT_SOP_PROJECT_DIR)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("project.json", "rois.json", "workflow.json"):
                data = json.loads((source / name).read_text(encoding="utf-8"))
                if name == "workflow.json":
                    data["steps"][0]["trigger"]["class_name"] = "missing"
                (root / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            with self.assertRaises(ProjectValidationError) as context:
                load_sop_project(root)

            self.assertEqual(context.exception.field, "steps[0].trigger.class_name")

    def test_project_config_paths_cannot_escape_project_directory(self) -> None:
        source = Path(DEFAULT_SOP_PROJECT_DIR)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            root.mkdir()
            for name in ("project.json", "rois.json", "workflow.json"):
                data = json.loads((source / name).read_text(encoding="utf-8"))
                if name == "project.json":
                    data["active_workflow"] = "../outside.json"
                (root / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "必须位于 SOP 项目目录内"):
                load_sop_project(root)


if __name__ == "__main__":
    unittest.main()
