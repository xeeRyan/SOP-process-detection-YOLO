from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


DEFAULT_SOP_PROJECT_DIR = Path(__file__).resolve().parents[1] / "projects" / "SK_DEMO"
from scripts.inference import Detection
from scripts.project import ProjectValidationError, load_sop_project
from scripts.sop_logic import SOPStateMachine
from scripts.events import SopEvent


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

    def test_active_model_path_cannot_escape_project_directory(self) -> None:
        source = Path(DEFAULT_SOP_PROJECT_DIR)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            root.mkdir()
            for name in ("project.json", "rois.json", "workflow.json"):
                data = json.loads((source / name).read_text(encoding="utf-8"))
                if name == "project.json":
                    data["active_model"] = "../outside.pt"
                (root / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "必须位于 SOP 项目目录内"):
                load_sop_project(root)

    def test_workflow_evidence_must_match_active_model_task(self) -> None:
        source = Path(DEFAULT_SOP_PROJECT_DIR)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            root.mkdir()
            for name in ("project.json", "rois.json", "workflow.json"):
                data = json.loads((source / name).read_text(encoding="utf-8"))
                if name == "project.json":
                    data["active_model_task"] = "segment"
                if name == "workflow.json":
                    data["steps"][0]["trigger"]["evidence"] = {
                        "type": "keypoints",
                        "indices": [0],
                    }
                (root / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            with self.assertRaisesRegex(ProjectValidationError, "需要pose模型"):
                load_sop_project(root)

    def test_model_profiles_enable_mixed_vision_tasks(self) -> None:
        source = Path(DEFAULT_SOP_PROJECT_DIR)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            root.mkdir()
            for name in ("project.json", "rois.json", "workflow.json"):
                data = json.loads((source / name).read_text(encoding="utf-8"))
                if name == "project.json":
                    data["model_profiles"] = {
                        "segment": {
                            "path": "models/sop_segment/1.0.0/best.pt",
                            "task": "segment",
                        }
                    }
                if name == "workflow.json":
                    data["steps"][0]["trigger"]["evidence"] = {
                        "type": "mask",
                        "min_roi_overlap": 0.3,
                    }
                (root / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            project = load_sop_project(root)
            self.assertEqual(project.required_model_tasks, {"detect", "segment"})
            self.assertEqual(
                project.resolve_model_path("segment"),
                root / "models" / "sop_segment" / "1.0.0" / "best.pt",
            )

    def test_attribute_sequence_accepts_random_initial_value_and_alternates_axes(self) -> None:
        workflow = {
            "workflow_id": "sequence",
            "constraints": {
                "attribute_sequences": [
                    {
                        "id": "orientation",
                        "class_name": "item",
                        "roi_id": "stack",
                        "axes": [
                            {"attribute": "surface", "values": ["a", "b"]},
                            {"attribute": "direction", "values": ["left", "right"]},
                        ],
                    }
                ]
            },
            "steps": [
                {
                    "id": "first",
                    "order": 1,
                    "name": "第一件",
                    "trigger": {"type": "object_event", "event": "object_enter_roi", "class_name": "item", "roi_id": "stack", "stable_frames": 1},
                },
                {
                    "id": "second",
                    "order": 2,
                    "name": "第二件",
                    "trigger": {"type": "object_event", "event": "object_enter_roi", "class_name": "item", "roi_id": "stack", "stable_frames": 1},
                },
            ],
        }
        machine = SOPStateMachine(workflow=workflow, rois={"stack": [0, 0, 100, 100]})
        first = Detection("item", 0.9, [10, 10, 20, 20], track_id=1, attributes={"surface": "b", "direction": "right"})
        second = Detection("item", 0.9, [10, 10, 20, 20], track_id=2, attributes={"surface": "a", "direction": "left"})

        machine.update([first], 1, 0.1, events=[SopEvent("object_enter_roi", 1, 0.1, "item", 1, "stack")])
        machine.update([second], 2, 0.2, events=[SopEvent("object_enter_roi", 2, 0.2, "item", 2, "stack")])

        self.assertEqual(machine.final_result, "OK")
        self.assertEqual(machine.finalize("sequence.mp4")["constraints"][0]["count"], 2)

    def test_attribute_sequence_mismatch_fails_without_business_specific_logic(self) -> None:
        workflow = {
            "workflow_id": "sequence",
            "constraints": {
                "attribute_sequences": [
                    {"id": "orientation", "axes": [{"attribute": "state", "values": ["a", "b"]}]}
                ]
            },
            "steps": [
                {
                    "id": "first",
                    "order": 1,
                    "name": "第一件",
                    "trigger": {"type": "object_event", "event": "object_enter_roi", "class_name": "item", "stable_frames": 1},
                },
                {
                    "id": "second",
                    "order": 2,
                    "name": "第二件",
                    "trigger": {"type": "object_event", "event": "object_enter_roi", "class_name": "item", "stable_frames": 1},
                },
            ],
        }
        machine = SOPStateMachine(workflow=workflow, rois={"stack": [0, 0, 100, 100]})
        detection = Detection("item", 0.9, [10, 10, 20, 20], track_id=1, attributes={"state": "a"})
        event = SopEvent("object_enter_roi", 1, 0.1, "item", 1, "stack")

        machine.update([detection], 1, 0.1, events=[event])
        wrong = Detection("item", 0.9, [10, 10, 20, 20], track_id=2, attributes={"state": "a"})
        machine.update(
            [wrong],
            2,
            0.2,
            events=[SopEvent("object_enter_roi", 2, 0.2, "item", 2, "stack")],
        )

        self.assertEqual(machine.final_result, "NG")
        self.assertIn("属性序列约束 orientation", machine.reason)

    def test_batch_step_resets_attribute_sequence(self) -> None:
        workflow = {
            "workflow_id": "batch-sequence",
            "constraints": {
                "attribute_sequences": [
                    {"id": "orientation", "axes": [{"attribute": "state", "values": ["a", "b"]}]}
                ],
                "batch_policy": {"reset_sequence_on_step_ids": ["batch_2_start"]},
            },
            "steps": [
                {"id": "batch_1_item", "order": 1, "name": "第一批", "trigger": {"type": "object_event", "event": "object_enter_roi", "class_name": "item", "stable_frames": 1}},
                {"id": "batch_2_start", "order": 2, "name": "第二批开始", "trigger": {"type": "object_present", "class_name": "item", "stable_frames": 1}},
                {"id": "batch_2_item", "order": 3, "name": "第二批", "trigger": {"type": "object_event", "event": "object_enter_roi", "class_name": "item", "stable_frames": 1}},
            ],
        }
        machine = SOPStateMachine(workflow=workflow, rois={"stack": [0, 0, 100, 100]})
        first = Detection("item", 0.9, [10, 10, 20, 20], track_id=1, attributes={"state": "a"})
        event = SopEvent("object_enter_roi", 1, 0.1, "item", 1, "stack")
        machine.update([first], 1, 0.1, events=[event])
        machine.update([first], 2, 0.2)

        second = Detection("item", 0.9, [10, 10, 20, 20], track_id=2, attributes={"state": "a"})
        machine.update(
            [second],
            3,
            0.3,
            events=[SopEvent("object_enter_roi", 3, 0.3, "item", 2, "stack")],
        )

        self.assertEqual(machine.final_result, "OK")
        self.assertEqual(machine.finalize("batch.mp4")["constraints"][0]["count"], 1)


if __name__ == "__main__":
    unittest.main()
