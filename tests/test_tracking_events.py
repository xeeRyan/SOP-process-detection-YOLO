from __future__ import annotations

import unittest

from scripts.inference import Detection
from scripts.events import SopEventEngine
from scripts.sop_logic import SOPStateMachine
from scripts.tracking import ByteTrackTracker


class TrackingEventTests(unittest.TestCase):
    def test_tracker_preserves_id_and_event_engine_emits_roi_transition(self) -> None:
        tracker = ByteTrackTracker(
            high_confidence=0.5,
            low_confidence=0.1,
            new_track_confidence=0.6,
            iou_threshold=0.1,
        )
        events = SopEventEngine({"work": [50, 0, 100, 100]})

        first = tracker.update([Detection("part", 0.9, [0, 10, 40, 30])])
        first_events = events.update(first, 0, 0.0)
        second = tracker.update([Detection("part", 0.9, [20, 10, 60, 30])])
        events.update(second, 1, 0.04)
        third = tracker.update([Detection("part", 0.9, [40, 10, 80, 30])])
        third_events = events.update(third, 2, 0.08)

        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(second[0].track_id, third[0].track_id)
        self.assertIn("object_appear", [event.event_type for event in first_events])
        self.assertIn("object_enter_roi", [event.event_type for event in third_events])

    def test_transition_step_does_not_complete_from_object_already_inside(self) -> None:
        workflow = {
            "steps": [
                {
                    "id": "place",
                    "order": 1,
                    "name": "放入零件",
                    "required": True,
                    "trigger": {
                        "type": "object_in_roi",
                        "event": "object_enter_roi",
                        "class_name": "part",
                        "roi_id": "work",
                        "stable_frames": 1,
                    },
                }
            ]
        }
        machine = SOPStateMachine(workflow=workflow, rois={"work": [50, 0, 100, 100]})
        event_engine = SopEventEngine(machine.rois)

        already_inside = Detection("part", 0.9, [60, 10, 80, 30], track_id=1)
        initial_events = event_engine.update([already_inside], 0, 0.0)
        machine.update([already_inside], 0, 0.0, events=initial_events)

        self.assertEqual(machine.steps[0].status, "pending")

        outside = Detection("part", 0.9, [0, 10, 20, 30], track_id=1)
        event_engine.update([outside], 1, 0.04)
        entered = Detection("part", 0.9, [55, 10, 75, 30], track_id=1)
        enter_events = event_engine.update([entered], 2, 0.08)
        machine.update([entered], 2, 0.08, events=enter_events)

        self.assertEqual(machine.final_result, "OK")

    def test_short_detection_gap_does_not_emit_false_disappear_or_exit(self) -> None:
        engine = SopEventEngine({"work": [0, 0, 100, 100]}, lost_tolerance_frames=2)
        detection = Detection("part", 0.9, [10, 10, 30, 30], track_id=1)

        engine.update([detection], 0, 0.0)
        first_gap = engine.update([], 1, 0.04)
        second_gap = engine.update([], 2, 0.08)
        recovered = engine.update([detection], 3, 0.12)

        self.assertEqual(first_gap, [])
        self.assertEqual(second_gap, [])
        self.assertNotIn("object_appear", [event.event_type for event in recovered])

    def test_disappear_is_emitted_after_tolerance_is_exceeded(self) -> None:
        engine = SopEventEngine({"work": [0, 0, 100, 100]}, lost_tolerance_frames=1)
        detection = Detection("part", 0.9, [10, 10, 30, 30], track_id=1)

        engine.update([detection], 0, 0.0)
        self.assertEqual(engine.update([], 1, 0.04), [])
        events = engine.update([], 2, 0.08)

        event_types = [event.event_type for event in events]
        self.assertIn("object_exit_roi", event_types)
        self.assertIn("object_disappear", event_types)

    def test_negative_lost_tolerance_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SopEventEngine({}, lost_tolerance_frames=-1)

    def test_enter_event_requires_consecutive_confirmations(self) -> None:
        engine = SopEventEngine(
            {"work": [50, 0, 100, 100]},
            enter_stable_frames=2,
        )
        outside = Detection("part", 0.9, [0, 10, 20, 30], track_id=1)
        inside = Detection("part", 0.9, [60, 10, 80, 30], track_id=1)

        engine.update([outside], 0, 0.0)
        first_inside = engine.update([inside], 1, 0.04)
        interrupted = engine.update([outside], 2, 0.08)
        second_inside = engine.update([inside], 3, 0.12)
        confirmed = engine.update([inside], 4, 0.16)

        self.assertNotIn("object_enter_roi", [event.event_type for event in first_inside])
        self.assertNotIn("object_enter_roi", [event.event_type for event in interrupted])
        self.assertNotIn("object_enter_roi", [event.event_type for event in second_inside])
        self.assertIn("object_enter_roi", [event.event_type for event in confirmed])

    def test_exit_event_requires_consecutive_confirmations(self) -> None:
        engine = SopEventEngine(
            {"work": [50, 0, 100, 100]},
            exit_stable_frames=2,
        )
        inside = Detection("part", 0.9, [60, 10, 80, 30], track_id=1)
        outside = Detection("part", 0.9, [0, 10, 20, 30], track_id=1)

        engine.update([inside], 0, 0.0)
        first_exit = engine.update([outside], 1, 0.04)
        recovered = engine.update([inside], 2, 0.08)
        second_exit = engine.update([outside], 3, 0.12)
        confirmed = engine.update([outside], 4, 0.16)

        self.assertNotIn("object_exit_roi", [event.event_type for event in first_exit])
        self.assertNotIn("object_exit_roi", [event.event_type for event in recovered])
        self.assertNotIn("object_exit_roi", [event.event_type for event in second_exit])
        self.assertIn("object_exit_roi", [event.event_type for event in confirmed])

    def test_invalid_event_stability_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SopEventEngine({}, enter_stable_frames=0)
