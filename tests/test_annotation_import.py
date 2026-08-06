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


class AnnotationImportTests(unittest.TestCase):
    def _prepare_project(self, root: Path) -> tuple[Path, list[str]]:
        project_dir = root / "TEST_SOP"
        project_dir.mkdir()
        for name in ("project.json", "rois.json", "workflow.json"):
            shutil.copy2(Path(DEFAULT_SOP_PROJECT_DIR) / name, project_dir / name)
        project_data = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
        project_data["project_id"] = "TEST_SOP"
        (project_dir / "project.json").write_text(
            json.dumps(project_data, ensure_ascii=False), encoding="utf-8"
        )
        extraction = extract_project_videos(
            project_dir,
            [ROOT / "videos" / "smoke_60f.mp4"],
            frame_interval=10,
            max_frames_per_video=3,
            copy_videos=False,
        )
        stems = [Path(item["image_path"]).stem for item in extraction["videos"][0]["frames"]]
        return project_dir, stems

    def test_import_reports_valid_invalid_missing_and_unmatched(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_dir, stems = self._prepare_project(root)
            labels_dir = root / "external_labels"
            labels_dir.mkdir()
            (labels_dir / f"{stems[0]}.txt").write_text(
                "0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8"
            )
            (labels_dir / f"{stems[1]}.txt").write_text(
                "9 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8"
            )
            (labels_dir / "unknown_frame.txt").write_text(
                "0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8"
            )

            result = run_task(
                {
                    "command": "import_annotations",
                    "params": {
                        "sop_project_dir": str(project_dir),
                        "labels_dir": str(labels_dir),
                    },
                }
            )

            summary = result["summary"]
            self.assertEqual(summary["imported_count"], 1)
            self.assertEqual(summary["invalid_count"], 1)
            self.assertEqual(summary["missing_label_count"], 1)
            self.assertEqual(summary["unmatched_label_count"], 1)
            self.assertTrue(Path(result["report_path"]).is_file())
            imported_path = project_dir / result["imported"][0]["annotation_path"]
            self.assertEqual(
                imported_path.read_text(encoding="utf-8"),
                "0 0.500000 0.500000 0.200000 0.200000\n",
            )

            manifest = json.loads((project_dir / "frames_manifest.json").read_text(encoding="utf-8"))
            frames = next(iter(manifest["videos"].values()))["frames"]
            statuses = {Path(item["image_path"]).stem: item["annotation_status"] for item in frames}
            self.assertEqual(statuses[stems[0]], "labeled")
            self.assertEqual(statuses[stems[1]], "unlabeled")
            self.assertEqual(statuses[stems[2]], "unlabeled")

    def test_mark_missing_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_dir, _ = self._prepare_project(root)
            labels_dir = root / "empty_labels"
            labels_dir.mkdir()

            result = run_task(
                {
                    "command": "import_annotations",
                    "params": {
                        "sop_project_dir": str(project_dir),
                        "labels_dir": str(labels_dir),
                        "mark_missing_as_empty": True,
                    },
                }
            )

            self.assertEqual(result["summary"]["empty_count"], 3)
            self.assertEqual(result["summary"]["missing_label_count"], 0)

    def test_import_does_not_require_completed_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_dir, stems = self._prepare_project(root)
            (project_dir / "workflow.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "workflow_id": "TEST_SOP_V1",
                        "name": "待配置流程",
                        "version": "1.0.0",
                        "steps": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            labels_dir = root / "external_labels"
            labels_dir.mkdir()
            (labels_dir / f"{stems[0]}.txt").write_text(
                "0 0.500000 0.500000 0.200000 0.200000\n",
                encoding="utf-8",
            )

            result = run_task(
                {
                    "command": "import_annotations",
                    "params": {
                        "sop_project_dir": str(project_dir),
                        "labels_dir": str(labels_dir),
                    },
                }
            )

            self.assertEqual(result["summary"]["imported_count"], 1)
