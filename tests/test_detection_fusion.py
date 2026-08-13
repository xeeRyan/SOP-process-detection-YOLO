from __future__ import annotations

import unittest

from scripts.inference import Detection, DetectionFusion


class DetectionFusionTests(unittest.TestCase):
    def test_auxiliary_evidence_is_attached_to_tracked_detection(self) -> None:
        primary = Detection(
            "part",
            0.9,
            [10, 10, 50, 50],
            track_id=7,
            attributes={"state": "visible"},
        )
        auxiliary = Detection(
            "part",
            0.8,
            [12, 12, 48, 48],
            mask=[[12, 12], [48, 12], [48, 48]],
            keypoints=[[20, 20, 0.95]],
            attributes={"surface": "front"},
        )

        result = DetectionFusion(iou_threshold=0.3).merge(
            [primary], {"segment": [auxiliary]}
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].track_id, 7)
        self.assertIsNotNone(result[0].mask)
        self.assertEqual(result[0].keypoints, [[20, 20, 0.95]])
        self.assertEqual(
            result[0].attributes,
            {"state": "visible", "surface": "front"},
        )

    def test_unmatched_auxiliary_detection_is_retained(self) -> None:
        primary = Detection("part", 0.9, [0, 0, 10, 10], track_id=1)
        auxiliary = Detection("part", 0.8, [50, 50, 70, 70], mask=[])

        result = DetectionFusion(iou_threshold=0.5).merge(
            [primary], {"segment": [auxiliary]}
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[1].track_id, None)

    def test_different_classes_are_not_fused(self) -> None:
        primary = Detection("part", 0.9, [0, 0, 20, 20], track_id=1)
        auxiliary = Detection("tool", 0.8, [0, 0, 20, 20], keypoints=[])

        result = DetectionFusion().merge([primary], {"pose": [auxiliary]})

        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
