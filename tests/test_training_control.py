from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path

from tools.training import _build_epoch_logger
from tools.training_control import initialize_training_control, stop_requested


class _FakeTrainer:
    epoch = 4
    epochs = 100
    tloss = None
    metrics = {"metrics/mAP50(B)": 0.75}
    lr = {"lr0": 0.001}
    fitness = 0.7
    stop = False

    @staticmethod
    def label_loss_items(_loss):
        return {"box_loss": 0.1}


class TrainingControlTests(unittest.TestCase):
    def test_initializes_new_task_without_stale_stop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            control_dir = Path(temp_dir)
            (control_dir / "stoptrain.json").write_text(
                json.dumps({"task_id": "old", "is_stop": True}), encoding="utf-8"
            )
            status_path, stop_path = initialize_training_control(
                control_dir,
                task_id="detector_1.0.0",
                model_id="detector",
                model_version="1.0.0",
                total_epochs=100,
            )
            self.assertFalse(stop_requested(stop_path, "detector_1.0.0"))
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["current_epoch"], 0)
            self.assertEqual(status["status"], "running")

    def test_epoch_callback_updates_progress_and_stops_after_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path, stop_path = initialize_training_control(
                temp_dir,
                task_id="detector_1.0.0",
                model_id="detector",
                model_version="1.0.0",
                total_epochs=100,
            )
            stop_path.write_text(
                json.dumps({"task_id": "detector_1.0.0", "is_stop": True}),
                encoding="utf-8",
            )
            state = {"current_epoch": 0, "stopped_by_user": False}
            callback = _build_epoch_logger(
                logging.getLogger("training-control-test"),
                training_state=state,
                status_path=status_path,
                stop_path=stop_path,
                task_id="detector_1.0.0",
                model_id="detector",
                model_version="1.0.0",
            )
            trainer = _FakeTrainer()
            callback(trainer)

            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertTrue(trainer.stop)
            self.assertTrue(state["stopped_by_user"])
            self.assertEqual(state["current_epoch"], 5)
            self.assertEqual(status["status"], "stopping")
            self.assertEqual(status["progress_percent"], 5.0)


if __name__ == "__main__":
    unittest.main()
