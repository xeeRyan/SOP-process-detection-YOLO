from __future__ import annotations

import unittest
from pathlib import Path

from scripts.main_video import _build_temporary_video_path, _should_run_hand_pose


class VideoOutputPathTests(unittest.TestCase):
    def test_temporary_video_paths_are_unique_and_stay_in_output_directory(self) -> None:
        result = Path("outputs/frontend_detect/result.mp4")

        first = _build_temporary_video_path(result)
        second = _build_temporary_video_path(result)

        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, result.parent)
        self.assertEqual(first.suffix, ".mp4")
        self.assertIn(".inprogress", first.name)

    def test_hand_pose_inference_interval_is_applied(self) -> None:
        sampled = [frame for frame in range(10) if _should_run_hand_pose(frame, 3)]

        self.assertEqual(sampled, [0, 3, 6, 9])
        with self.assertRaises(ValueError):
            _should_run_hand_pose(0, 0)
