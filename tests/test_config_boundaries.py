from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import scripts.config as runtime_config
from scripts.legacy_sk_config import DEFAULT_MODEL_PATH, DEFAULT_SOP_PROJECT_DIR
from scripts.main_video import process_video


class ConfigBoundaryTests(unittest.TestCase):
    def test_runtime_config_does_not_expose_sk_business_defaults(self) -> None:
        for name in (
            "TARGET_CLASSES",
            "STEP_DEFINITIONS",
            "WORK_ROI",
            "SCREW_BIN_ROI",
            "TOOL_HOME_ROI",
            "DEFAULT_MODEL_PATH",
            "DEFAULT_VIDEO_PATH",
            "DEFAULT_SOP_PROJECT_DIR",
        ):
            self.assertFalse(hasattr(runtime_config, name), name)

    def test_project_without_active_model_does_not_fall_back_to_sk_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "PROJECT"
            project_dir.mkdir()
            for name in ("project.json", "rois.json", "workflow.json"):
                shutil.copy2(Path(DEFAULT_SOP_PROJECT_DIR) / name, project_dir / name)
            project = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
            project["project_id"] = "NO_MODEL"
            project["active_model"] = None
            (project_dir / "project.json").write_text(
                json.dumps(project, ensure_ascii=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "未配置活动模型"):
                process_video(
                    video_path=project_dir / "missing.mp4",
                    output_dir=project_dir / "outputs",
                    sop_project_dir=project_dir,
                )

    def test_legacy_default_model_remains_available(self) -> None:
        self.assertEqual(DEFAULT_MODEL_PATH.name, "best_yolo26s.pt")


if __name__ == "__main__":
    unittest.main()
