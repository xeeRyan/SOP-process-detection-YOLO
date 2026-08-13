from __future__ import annotations

import unittest

from scripts.inference import Detection
from scripts.main_video import _vision_tasks_for_trigger
from scripts.project.schema_validator import ProjectValidationError, validate_workflow
from scripts.sop_logic import SOPStateMachine
from scripts.utils import mask_roi_overlap


class VisionEvidenceTests(unittest.TestCase):
    def test_auxiliary_tasks_are_scheduled_only_for_active_evidence(self) -> None:
        self.assertEqual(
            _vision_tasks_for_trigger({"type": "object_in_roi"}),
            {"detect"},
        )
        self.assertEqual(
            _vision_tasks_for_trigger(
                {"type": "object_in_roi", "evidence": {"type": "mask"}}
            ),
            {"segment"},
        )
        self.assertEqual(
            _vision_tasks_for_trigger(
                {
                    "type": "composite",
                    "conditions": [
                        {"type": "object_in_roi"},
                        {"type": "object_in_roi", "evidence": {"type": "keypoints"}},
                    ],
                }
            ),
            {"detect", "pose"},
        )

    def test_detection_serializes_optional_mask_and_keypoints(self) -> None:
        value = Detection(
            class_name="part",
            conf=0.9,
            bbox=[10, 10, 90, 90],
            mask=[[10, 10], [90, 10], [90, 90]],
            keypoints=[[20, 30, 0.8]],
        ).to_dict()

        self.assertEqual(value["mask"], [[10, 10], [90, 10], [90, 90]])
        self.assertEqual(value["keypoints"], [[20, 30, 0.8]])

    def test_mask_overlap_is_used_by_sop_trigger(self) -> None:
        workflow = {
            "steps": [
                {
                    "id": "install",
                    "order": 1,
                    "name": "安装到位",
                    "trigger": {
                        "type": "object_in_roi",
                        "class_name": "part",
                        "roi_id": "work",
                        "evidence": {"type": "mask", "min_roi_overlap": 0.4},
                        "stable_frames": 1,
                    },
                }
            ]
        }
        machine = SOPStateMachine(workflow=workflow, rois={"work": [0, 0, 50, 100]})
        detection = Detection(
            class_name="part",
            conf=0.9,
            bbox=[10, 10, 90, 90],
            mask=[[10, 10], [90, 10], [90, 90], [10, 90]],
        )

        machine.update([detection], frame_id=1, time_sec=0.04)

        self.assertEqual(machine.final_result, "OK")
        self.assertGreater(mask_roi_overlap(detection.mask, [0, 0, 50, 100]), 0.45)

    def test_keypoint_evidence_is_used_by_sop_trigger(self) -> None:
        workflow = {
            "steps": [
                {
                    "id": "press",
                    "order": 1,
                    "name": "按压",
                    "trigger": {
                        "type": "object_in_roi",
                        "class_name": "hand",
                        "roi_id": "work",
                        "evidence": {
                            "type": "keypoints",
                            "indices": [0],
                            "min_keypoint_score": 0.7,
                        },
                        "stable_frames": 1,
                    },
                }
            ]
        }
        machine = SOPStateMachine(workflow=workflow, rois={"work": [0, 0, 50, 50]})
        machine.update(
            [Detection("hand", 0.9, [60, 60, 90, 90], keypoints=[[20, 20, 0.9]])],
            frame_id=1,
            time_sec=0.04,
        )

        self.assertEqual(machine.final_result, "OK")

    def test_invalid_evidence_is_rejected(self) -> None:
        workflow = {
            "workflow_id": "invalid",
            "steps": [
                {
                    "id": "step",
                    "order": 1,
                    "name": "步骤",
                    "trigger": {
                        "type": "object_in_roi",
                        "class_name": "part",
                        "roi_id": "work",
                        "evidence": {"type": "mask", "min_roi_overlap": 2},
                    },
                }
            ],
        }
        with self.assertRaises(ProjectValidationError):
            validate_workflow(workflow, {"part"}, {"work"})


if __name__ == "__main__":
    unittest.main()
