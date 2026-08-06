from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.config import ROOT
from scripts.legacy_sk_config import DEFAULT_SOP_PROJECT_DIR
from task_dispatcher import run_task
from tools.project_frames import extract_project_videos


class ProjectFrameTests(unittest.TestCase):
    def _create_project(self, root: Path) -> Path:
        project_dir = root / "TEST_SOP"
        project_dir.mkdir()
        for name in ("project.json", "rois.json", "workflow.json"):
            shutil.copy2(Path(DEFAULT_SOP_PROJECT_DIR) / name, project_dir / name)
        project_data = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
        project_data["project_id"] = "TEST_SOP"
        (project_dir / "project.json").write_text(
            json.dumps(project_data, ensure_ascii=False), encoding="utf-8"
        )
        return project_dir

    def test_extract_project_video_builds_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = self._create_project(Path(temp_dir))
            result = extract_project_videos(
                project_dir=project_dir,
                video_paths=[ROOT / "videos" / "smoke_60f.mp4"],
                frame_interval=10,
                max_frames_per_video=3,
                copy_videos=False,
            )

            self.assertEqual(result["project_id"], "TEST_SOP")
            self.assertEqual(result["extracted_frame_count"], 3)
            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            video = next(iter(manifest["videos"].values()))
            self.assertEqual(video["extracted_frame_count"], 3)
            self.assertEqual(video["frames"][0]["annotation_status"], "unlabeled")
            for frame in video["frames"]:
                self.assertTrue((project_dir / frame["image_path"]).is_file())

    def test_dispatcher_accepts_single_video_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = self._create_project(Path(temp_dir))
            result = run_task(
                {
                    "command": "extract_frames",
                    "params": {
                        "sop_project_dir": str(project_dir),
                        "video_path": str(ROOT / "videos" / "smoke_60f.mp4"),
                        "frame_interval": 20,
                        "max_frames_per_video": 2,
                        "copy_videos": True,
                    },
                }
            )

            self.assertEqual(result["video_count"], 1)
            self.assertEqual(result["extracted_frame_count"], 2)
            video = result["videos"][0]
            self.assertIsNotNone(video["stored_path"])
            self.assertTrue((project_dir / video["stored_path"]).is_file())

    def test_extract_does_not_require_roi_or_workflow_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = self._create_project(Path(temp_dir))
            (project_dir / "rois.json").write_text(
                json.dumps(
                    {"schema_version": "1.0", "coordinate_type": "normalized", "regions": []}
                ),
                encoding="utf-8",
            )
            (project_dir / "workflow.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "workflow_id": "TEST_SOP_V1",
                        "name": "待配置流程",
                        "version": "1.0.0",
                        "steps": [],
                    }
                ),
                encoding="utf-8",
            )

            result = extract_project_videos(
                project_dir=project_dir,
                video_paths=[ROOT / "videos" / "smoke_60f.mp4"],
                frame_interval=30,
                max_frames_per_video=1,
                copy_videos=False,
            )

            self.assertEqual(result["extracted_frame_count"], 1)
