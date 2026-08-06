from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from task_dispatcher import run_task
from scripts.legacy_sk_config import DEFAULT_SOP_PROJECT_DIR


class ProjectModelConversionTests(unittest.TestCase):
    def _create_registered_model(self, root: Path) -> tuple[Path, Path]:
        project_dir = root / "TEST_SOP"
        model_dir = project_dir / "models" / "detector" / "1.0.0"
        model_dir.mkdir(parents=True)
        for name in ("project.json", "rois.json", "workflow.json"):
            shutil.copy2(Path(DEFAULT_SOP_PROJECT_DIR) / name, project_dir / name)
        project_data = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
        project_data["project_id"] = "TEST_SOP"
        (project_dir / "project.json").write_text(
            json.dumps(project_data, ensure_ascii=False), encoding="utf-8"
        )
        (model_dir / "best.pt").write_bytes(b"checkpoint")
        (model_dir / "model_manifest.json").write_text(
            json.dumps(
                {
                    "model_id": "detector",
                    "model_version": "1.0.0",
                    "artifacts": {"best_pt": "models/detector/1.0.0/best.pt"},
                }
            ),
            encoding="utf-8",
        )
        return project_dir, model_dir

    @staticmethod
    def _fake_export(**kwargs):
        result = {
            "requested": {
                "torchscript": kwargs["export_torchscript_enabled"],
                "onnx": kwargs["export_onnx_enabled"],
                "engine": kwargs["export_engine_enabled"],
            },
            "torchscript": None,
            "onnx": None,
            "engine": None,
            "errors": [],
        }
        if kwargs["export_torchscript_enabled"]:
            path = Path(kwargs["torchscript_output_path"])
            path.write_bytes(b"torchscript")
            result["torchscript"] = str(path)
        if kwargs["export_onnx_enabled"] or kwargs["export_engine_enabled"]:
            path = Path(kwargs["onnx_output_path"])
            path.write_bytes(b"onnx")
            result["onnx"] = str(path)
        if kwargs["export_engine_enabled"]:
            path = Path(kwargs["engine_output_path"])
            path.write_bytes(b"engine")
            result["engine"] = str(path)
        return result

    def test_converts_selected_formats_and_updates_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir, model_dir = self._create_registered_model(Path(temp_dir))
            with patch(
                "tools.project_model_conversion.export_model_formats",
                side_effect=self._fake_export,
            ):
                result = run_task(
                    {
                        "command": "convert_project_model",
                        "params": {
                            "sop_project_dir": str(project_dir),
                            "model_id": "detector",
                            "model_version": "1.0.0",
                            "formats": ["torchscript", "onnx", "engine"],
                        },
                    }
                )

            self.assertEqual(result["conversions"]["torchscript"]["status"], "success")
            self.assertEqual(result["conversions"]["onnx"]["status"], "success")
            self.assertEqual(result["conversions"]["engine"]["status"], "success")
            manifest = json.loads((model_dir / "model_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifacts"]["torchscript"], "models/detector/1.0.0/best.torchscript")
            self.assertEqual(manifest["artifacts"]["onnx"], "models/detector/1.0.0/best.onnx")
            self.assertEqual(manifest["artifacts"]["engine"], "models/detector/1.0.0/best.engine")

    def test_engine_only_reuses_existing_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir, model_dir = self._create_registered_model(Path(temp_dir))
            (model_dir / "best.onnx").write_bytes(b"existing-onnx")
            with patch(
                "tools.project_model_conversion.export_model_formats",
                side_effect=self._fake_export,
            ) as export:
                run_task(
                    {
                        "command": "convert_project_model",
                        "params": {
                            "sop_project_dir": str(project_dir),
                            "model_id": "detector",
                            "model_version": "1.0.0",
                            "formats": ["engine"],
                        },
                    }
                )
            self.assertTrue(export.call_args.kwargs["reuse_existing_onnx"])

    def test_rejects_empty_or_unknown_formats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir, _ = self._create_registered_model(Path(temp_dir))
            base_params = {
                "sop_project_dir": str(project_dir),
                "model_id": "detector",
                "model_version": "1.0.0",
            }
            for formats in ([], ["openvino"]):
                with self.subTest(formats=formats), self.assertRaises(ValueError):
                    run_task(
                        {
                            "command": "convert_project_model",
                            "params": {**base_params, "formats": formats},
                        }
                    )


if __name__ == "__main__":
    unittest.main()
