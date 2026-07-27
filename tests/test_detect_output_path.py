from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from task_dispatcher import resolve_detect_output_dir


class DetectOutputPathTests(unittest.TestCase):
    def test_relative_output_is_resolved_inside_sop_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "DEMO"
            project.mkdir()

            output = resolve_detect_output_dir(project, "outputs/run_001")

            self.assertEqual(output, (project / "outputs" / "run_001").resolve())

    def test_output_cannot_escape_sop_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "DEMO"
            project.mkdir()

            with self.assertRaisesRegex(ValueError, "必须位于 SOP 项目目录内"):
                resolve_detect_output_dir(project, "../outside")
