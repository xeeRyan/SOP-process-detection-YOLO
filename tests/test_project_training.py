from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.config import DEFAULT_SOP_PROJECT_DIR
from task_dispatcher import run_task


class ProjectTrainingTests(unittest.TestCase):
    def _create_project(self, root: Path) -> tuple[Path, Path]:
        project_dir = root / "TEST_SOP"
        project_dir.mkdir()
        for name in ("project.json", "rois.json", "workflow.json"):
            shutil.copy2(Path(DEFAULT_SOP_PROJECT_DIR) / name, project_dir / name)
        project_data = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
        project_data["project_id"] = "TEST_SOP"
        project_data["active_model"] = "base.pt"
        (project_dir / "project.json").write_text(
            json.dumps(project_data, ensure_ascii=False), encoding="utf-8"
        )
        base_model = project_dir / "base.pt"
        base_model.write_bytes(b"base-model")
        dataset_dir = project_dir / "dataset" / "dataset_v1"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "data.yaml").write_text("train: images/train\nval: images/val\n", encoding="utf-8")
        (dataset_dir / "dataset_manifest.json").write_text(
            json.dumps({"project_id": "TEST_SOP"}), encoding="utf-8"
        )
        return project_dir, base_model

    @staticmethod
    def _fake_training(**kwargs):
        copy_best = Path(kwargs["copy_best_to"])
        copy_best.parent.mkdir(parents=True, exist_ok=True)
        copy_best.write_bytes(b"trained-model")
        run_dir = Path(kwargs["project_dir"]) / kwargs["run_name"]
        weights = run_dir / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        last_model = weights / "last.pt"
        last_model.write_bytes(b"last-model")
        results_csv = run_dir / "results.csv"
        results_csv.write_text(
            "epoch,metrics/mAP50(B),metrics/recall(B)\n0,0.91,0.87\n", encoding="utf-8"
        )
        args_yaml = run_dir / "args.yaml"
        args_yaml.write_text("epochs: 1\n", encoding="utf-8")
        log_path = Path(kwargs["log_dir"]) / "train_test.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok", encoding="utf-8")
        return {
            "run_dir": str(run_dir),
            "best_model": str(copy_best),
            "last_model": str(last_model),
            "results_csv": str(results_csv),
            "args_yaml": str(args_yaml),
            "copied_best_model": str(copy_best),
            "exported_models": {"torchscript": None, "onnx": None, "engine": None, "errors": []},
            "train_params": {"epochs": kwargs["epochs"], "data": str(kwargs["data_yaml_path"])},
            "log_path": str(log_path),
        }

    def test_train_project_registers_version_and_active_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir, base_model = self._create_project(Path(temp_dir))
            with patch("tools.project_training.train_yolo_model", side_effect=self._fake_training):
                result = run_task(
                    {
                        "command": "train_project",
                        "params": {
                            "sop_project_dir": str(project_dir),
                            "dataset_name": "dataset_v1",
                            "model_id": "detector",
                            "model_version": "1.0.0",
                            "base_model_path": str(base_model),
                            "epochs": 1,
                            "set_active": True,
                        },
                    }
                )

            model_manifest = json.loads(Path(result["model_manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(model_manifest["dataset_name"], "dataset_v1")
            self.assertEqual(model_manifest["validation_metrics"]["metrics/mAP50(B)"], 0.91)
            self.assertEqual(model_manifest["validation_metrics"]["metrics/recall(B)"], 0.87)
            project_data = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
            self.assertEqual(project_data["active_model"], "models/detector/1.0.0/best.pt")
            self.assertEqual(
                project_data["active_model_manifest"],
                "models/detector/1.0.0/model_manifest.json",
            )

    def test_rejects_existing_model_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir, base_model = self._create_project(Path(temp_dir))
            (project_dir / "models" / "detector" / "1.0.0").mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "模型版本已经存在"):
                run_task(
                    {
                        "command": "train_project",
                        "params": {
                            "sop_project_dir": str(project_dir),
                            "dataset_name": "dataset_v1",
                            "model_id": "detector",
                            "model_version": "1.0.0",
                            "base_model_path": str(base_model),
                        },
                    }
                )
