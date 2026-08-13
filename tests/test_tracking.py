from __future__ import annotations

import unittest

from scripts.inference import Detection
from scripts.tracking import ByteTrackTracker


def detection(class_name: str, bbox: list[float]) -> Detection:
    return Detection(class_name, 0.9, bbox)


class ByteTrackTrackerTests(unittest.TestCase):
    def create_tracker(self) -> ByteTrackTracker:
        return ByteTrackTracker(
            high_confidence=0.5,
            low_confidence=0.1,
            new_track_confidence=0.6,
            iou_threshold=0.2,
            second_match_iou_threshold=0.1,
        )

    def test_low_confidence_detection_recovers_existing_track(self) -> None:
        tracker = self.create_tracker()
        first = tracker.update([Detection("part", 0.8, [0, 0, 10, 10])])

        recovered = tracker.update([Detection("part", 0.2, [1, 0, 11, 10])])

        self.assertEqual(recovered[0].track_id, first[0].track_id)

    def test_low_confidence_detection_cannot_create_track(self) -> None:
        tracker = self.create_tracker()

        self.assertEqual(tracker.update([Detection("part", 0.2, [0, 0, 10, 10])]), [])

    def test_high_detection_below_new_track_threshold_cannot_create_track(self) -> None:
        tracker = self.create_tracker()

        self.assertEqual(tracker.update([Detection("part", 0.55, [0, 0, 10, 10])]), [])

    def test_first_stage_high_detection_wins_over_low_detection(self) -> None:
        tracker = self.create_tracker()
        first = tracker.update([Detection("part", 0.8, [0, 0, 10, 10])])

        tracked = tracker.update(
            [
                Detection("part", 0.7, [1, 0, 11, 10]),
                Detection("part", 0.2, [1, 0, 11, 10]),
            ]
        )

        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0].track_id, first[0].track_id)
        self.assertEqual(tracked[0].conf, 0.7)

    def test_invalid_confidence_order_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "置信度阈值"):
            ByteTrackTracker(
                low_confidence=0.5,
                high_confidence=0.4,
                new_track_confidence=0.6,
            )

    def test_motion_prediction_recovers_fast_target_after_missing_frame(self) -> None:
        tracker = self.create_tracker()
        first = tracker.update([Detection("part", 0.8, [0, 0, 10, 10])])
        tracker.update([Detection("part", 0.8, [4, 0, 14, 10])])

        tracker.update([])
        recovered = tracker.update([Detection("part", 0.8, [12, 0, 22, 10])])

        self.assertEqual(recovered[0].track_id, first[0].track_id)

    def test_motion_filter_handles_size_change(self) -> None:
        tracker = self.create_tracker()
        first = tracker.update([Detection("part", 0.8, [0, 0, 10, 10])])
        second = tracker.update([Detection("part", 0.8, [1, 0, 13, 12])])
        third = tracker.update([Detection("part", 0.8, [2, 0, 16, 14])])

        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(second[0].track_id, third[0].track_id)

    def test_prediction_keeps_confirmed_track_without_counting_a_miss(self) -> None:
        tracker = self.create_tracker()
        first = tracker.update([Detection("part", 0.8, [0, 0, 10, 10])])

        predicted = tracker.predict()

        self.assertEqual(len(predicted), 1)
        self.assertEqual(predicted[0].track_id, first[0].track_id)
        self.assertGreater(predicted[0].bbox[0], first[0].bbox[0] - 1)
        self.assertEqual(tracker.update([Detection("part", 0.8, [1, 0, 11, 10])])[0].track_id, first[0].track_id)

    def test_prediction_preserves_generic_attributes(self) -> None:
        tracker = self.create_tracker()
        first = tracker.update(
            [
                Detection(
                    "part",
                    0.8,
                    [0, 0, 10, 10],
                    attributes={"surface": "front", "direction": "left"},
                )
            ]
        )

        predicted = tracker.predict()

        self.assertEqual(predicted[0].track_id, first[0].track_id)
        self.assertEqual(
            predicted[0].attributes,
            {"surface": "front", "direction": "left"},
        )


if __name__ == "__main__":
    unittest.main()
