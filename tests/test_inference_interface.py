from __future__ import annotations

import unittest

from scripts.detector import Detection as LegacyDetection
from scripts.detector import YOLODetector
from scripts.inference import Detection, DetectorBackend, build_detector
from scripts.inference.python_yolo import PythonYoloBackend


class InferenceInterfaceTests(unittest.TestCase):
    def test_legacy_imports_point_to_new_python_inference_types(self) -> None:
        self.assertIs(LegacyDetection, Detection)
        self.assertIs(YOLODetector, PythonYoloBackend)
        self.assertTrue(issubclass(PythonYoloBackend, DetectorBackend))

    def test_detection_serialization_exposes_legacy_and_unified_fields(self) -> None:
        value = Detection(
            class_name="tool",
            conf=0.91234,
            bbox=[1.234, 2.345, 30.456, 40.567],
            track_id=7,
            class_id=2,
        ).to_dict()

        self.assertEqual(value["class_id"], 2)
        self.assertEqual(value["class"], "tool")
        self.assertEqual(value["class_name"], "tool")
        self.assertEqual(value["conf"], value["confidence"])
        self.assertEqual(value["track_id"], 7)

    def test_unknown_backend_is_rejected_before_model_loading(self) -> None:
        with self.assertRaisesRegex(ValueError, "不支持推理后端"):
            build_detector("missing.onnx", backend="cpp")


if __name__ == "__main__":
    unittest.main()
