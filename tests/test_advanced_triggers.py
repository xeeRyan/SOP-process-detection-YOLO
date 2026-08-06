from __future__ import annotations

import unittest

from scripts.detector import Detection
from scripts.events import SopEventEngine
from scripts.project.schema_validator import ProjectValidationError, validate_workflow
from scripts.sop_logic import SOPStateMachine


ROIS = {
    "source": [0, 0, 40, 100],
    "work": [40, 0, 80, 100],
    "target": [80, 0, 120, 100],
}


def workflow_with(trigger: dict) -> dict:
    return {
        "workflow_id": "advanced",
        "steps": [
            {
                "id": "step",
                "order": 1,
                "name": "高级触发步骤",
                "required": True,
                "trigger": trigger,
            }
        ],
    }


class AdvancedTriggerTests(unittest.TestCase):
    def test_composite_all_requires_every_condition(self) -> None:
        machine = SOPStateMachine(
            workflow=workflow_with(
                {
                    "type": "composite",
                    "operator": "all",
                    "stable_frames": 1,
                    "conditions": [
                        {
                            "type": "object_in_roi",
                            "class_name": "part",
                            "roi_id": "work",
                        },
                        {
                            "type": "object_present",
                            "class_name": "tool",
                        },
                    ],
                }
            ),
            rois=ROIS,
        )
        part = Detection("part", 0.9, [50, 10, 60, 20])
        tool = Detection("tool", 0.9, [5, 10, 15, 20])

        machine.update([part], 0, 0.0)
        self.assertEqual(machine.steps[0].status, "pending")
        machine.update([part, tool], 1, 0.04)
        self.assertEqual(machine.final_result, "OK")

    def test_object_count_supports_minimum_and_maximum(self) -> None:
        machine = SOPStateMachine(
            workflow=workflow_with(
                {
                    "type": "object_count",
                    "class_name": "screw",
                    "roi_id": "work",
                    "min_count": 2,
                    "max_count": 2,
                    "stable_frames": 1,
                }
            ),
            rois=ROIS,
        )
        first = Detection("screw", 0.9, [45, 10, 50, 15], track_id=1)
        second = Detection("screw", 0.9, [55, 10, 60, 15], track_id=2)

        machine.update([first], 0, 0.0)
        self.assertEqual(machine.steps[0].status, "pending")
        machine.update([first, second], 1, 0.04)
        self.assertEqual(machine.final_result, "OK")

    def test_cross_roi_transition_uses_same_track(self) -> None:
        machine = SOPStateMachine(
            workflow=workflow_with(
                {
                    "type": "object_transition",
                    "class_name": "part",
                    "from_roi_id": "source",
                    "to_roi_id": "target",
                }
            ),
            rois=ROIS,
        )
        events = SopEventEngine(ROIS)
        source = Detection("part", 0.9, [5, 10, 15, 20], track_id=1)
        target = Detection("part", 0.9, [90, 10, 100, 20], track_id=1)

        machine.update([source], 0, 0.0, events=events.update([source], 0, 0.0))
        move_events = events.update([target], 1, 0.04)
        machine.update([target], 1, 0.04, events=move_events)

        self.assertIn("object_move_roi", [event.event_type for event in move_events])
        self.assertEqual(machine.final_result, "OK")

    def test_duration_requires_continuous_condition(self) -> None:
        machine = SOPStateMachine(
            workflow=workflow_with(
                {
                    "type": "duration",
                    "duration_sec": 1.0,
                    "condition": {
                        "type": "object_in_roi",
                        "class_name": "tool",
                        "roi_id": "work",
                    },
                }
            ),
            rois=ROIS,
        )
        tool = Detection("tool", 0.9, [50, 10, 60, 20])

        machine.update([tool], 0, 0.0)
        machine.update([], 1, 0.5)
        machine.update([tool], 2, 0.6)
        machine.update([tool], 3, 1.5)
        self.assertEqual(machine.steps[0].status, "pending")
        machine.update([tool], 4, 1.6)
        self.assertEqual(machine.final_result, "OK")

    def test_schema_validates_nested_triggers(self) -> None:
        workflow = workflow_with(
            {
                "type": "composite",
                "operator": "any",
                "conditions": [
                    {
                        "type": "object_count",
                        "class_name": "part",
                        "roi_id": "work",
                        "min_count": 1,
                    },
                    {
                        "type": "duration",
                        "duration_sec": 0.5,
                        "condition": {
                            "type": "object_present",
                            "class_name": "tool",
                        },
                    },
                ],
            }
        )
        validate_workflow(workflow, {"part", "tool"}, set(ROIS))

        workflow["steps"][0]["trigger"]["operator"] = "invalid"
        with self.assertRaises(ProjectValidationError):
            validate_workflow(workflow, {"part", "tool"}, set(ROIS))


if __name__ == "__main__":
    unittest.main()
