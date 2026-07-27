from __future__ import annotations

import unittest
from pathlib import Path

from scripts.detector import _resolve_inference_device


class DetectorDeviceTests(unittest.TestCase):
    def test_onnx_auto_device_uses_cpu(self) -> None:
        self.assertEqual(_resolve_inference_device(Path("best.onnx"), None), "cpu")
        self.assertEqual(_resolve_inference_device(Path("best.onnx"), "auto"), "cpu")

    def test_explicit_device_and_native_models_are_preserved(self) -> None:
        self.assertEqual(_resolve_inference_device(Path("best.onnx"), "0"), "0")
        self.assertIsNone(_resolve_inference_device(Path("best.pt"), None))
