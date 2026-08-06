from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.legacy_sk_config import DEFAULT_SOP_PROJECT_DIR
from task_dispatcher import run_task


class DatasetBuilderTests(unittest.TestCase):
    def _create_project(self, root: Path, video_count: int, annotated: bool = True) -> Path:
        project_dir = root / "TEST_SOP"
        project_dir.mkdir()
        for name in ("project.json", "rois.json", "workflow.json"):
            shutil.copy2(Path(DEFAULT_SOP_PROJECT_DIR) / name, project_dir / name)
        project_data = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
        project_data["project_id"] = "TEST_SOP"
        (project_dir / "project.json").write_text(
            json.dumps(project_data, ensure_ascii=False), encoding="utf-8"
        )

        videos = {}
        for video_index in range(video_count):
            video_id = f"video_{video_index}"
            frames = []
            for frame_index in range(2):
                stem = f"{video_id}_{frame_index:04d}"
                image_path = project_dir / "frames" / video_id / f"{stem}.jpg"
                label_path = project_dir / "annotations" / video_id / f"{stem}.txt"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(b"fake-jpeg")
                if annotated:
                    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
                frames.append(
                    {
                        "frame_id": f"{video_id}:{frame_index}",
                        "image_path": image_path.relative_to(project_dir).as_posix(),
                        "annotation_path": label_path.relative_to(project_dir).as_posix(),
                        "annotation_status": "labeled" if annotated else "unlabeled",
                    }
                )
            videos[video_id] = {"video_id": video_id, "frames": frames}
        manifest = {"schema_version": "1.0", "project_id": "TEST_SOP", "videos": videos}
        (project_dir / "frames_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
        return project_dir

    def test_build_dataset_keeps_each_video_in_one_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = self._create_project(Path(temp_dir), video_count=4)
            result = run_task(
                {
                    "command": "build_dataset",
                    "params": {
                        "sop_project_dir": str(project_dir),
                        "dataset_name": "dataset_v1",
                        "train_ratio": 0.5,
                        "val_ratio": 0.25,
                        "test_ratio": 0.25,
                        "seed": 7,
                    },
                }
            )

            dataset_dir = Path(result["dataset_dir"])
            manifest = json.loads(Path(result["dataset_manifest"]).read_text(encoding="utf-8"))
            split_by_video: dict[str, set[str]] = {}
            for record in manifest["records"]:
                split_by_video.setdefault(record["video_id"], set()).add(record["split"])
            self.assertTrue(all(len(splits) == 1 for splits in split_by_video.values()))
            self.assertEqual(sum(item["frame_count"] for item in result["summary"].values()), 8)
            self.assertTrue((dataset_dir / "data.yaml").is_file())
            self.assertEqual(len(list((dataset_dir / "images" / "train").glob("*.jpg"))), 4)
            self.assertEqual(len(list((dataset_dir / "images" / "val").glob("*.jpg"))), 2)
            self.assertEqual(len(list((dataset_dir / "images" / "test").glob("*.jpg"))), 2)

    def test_rejects_too_few_independent_videos(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = self._create_project(Path(temp_dir), video_count=1)
            with self.assertRaisesRegex(ValueError, "独立视频"):
                run_task(
                    {
                        "command": "build_dataset",
                        "params": {"sop_project_dir": str(project_dir), "dataset_name": "dataset_v1"},
                    }
                )

    def test_rejects_unannotated_frames_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = self._create_project(Path(temp_dir), video_count=3, annotated=False)
            with self.assertRaisesRegex(ValueError, "没有可用于"):
                run_task(
                    {
                        "command": "build_dataset",
                        "params": {"sop_project_dir": str(project_dir), "dataset_name": "dataset_v1"},
                    }
                )

    def test_explicit_smoke_mode_can_split_frames_from_one_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = self._create_project(Path(temp_dir), video_count=1)
            result = run_task(
                {
                    "command": "build_dataset",
                    "params": {
                        "sop_project_dir": str(project_dir),
                        "dataset_name": "smoke_v1",
                        "train_ratio": 0.5,
                        "val_ratio": 0.5,
                        "test_ratio": 0.0,
                        "allow_single_video_frame_split": True,
                    },
                }
            )

            manifest = json.loads(
                Path(result["dataset_manifest"]).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["grouping"], "frame_smoke_test")
            self.assertIn("不能用于正式检出率评估", manifest["leakage_warning"])
            self.assertEqual(result["summary"]["train"]["frame_count"], 1)
            self.assertEqual(result["summary"]["val"]["frame_count"], 1)
