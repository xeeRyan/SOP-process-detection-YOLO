from __future__ import annotations

import unittest

import numpy as np

from scripts.inference import Detection, DetectorBackend, ModelRouter
from scripts.project.schema_validator import ProjectValidationError, validate_project


class _FakeBackend(DetectorBackend):
    backend_name = "fake"
    runtime_name = "test"

    def __init__(self, task: str) -> None:
        self.task = task
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        return [Detection(self.task, 0.9, [0, 0, 10, 10])]


class ModelRouterTests(unittest.TestCase):
    def test_tasks_are_inferred_from_nested_evidence(self) -> None:
        trigger = {
            "type": "composite",
            "conditions": [
                {"type": "object_in_roi", "evidence": {"type": "mask"}},
                {
                    "type": "duration",
                    "condition": {
                        "type": "object_in_roi",
                        "evidence": {"type": "keypoints"},
                    },
                },
            ],
        }

        self.assertEqual(ModelRouter.tasks_for_trigger(trigger), {"segment", "pose"})

    def test_router_caches_tasks_and_honors_task_intervals(self) -> None:
        detect = _FakeBackend("detect")
        segment = _FakeBackend("segment")
        router = ModelRouter(
            {"detect": detect, "segment": segment},
            default_interval=2,
            task_intervals={"segment": 1},
        )
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        trigger = {"type": "object_in_roi", "evidence": {"type": "mask"}}

        _, fresh_0 = router.infer(frame, trigger, 0)
        _, fresh_1 = router.infer(frame, trigger, 1)
        _, fresh_2 = router.infer(frame, trigger, 2)

        self.assertEqual(fresh_0, {"detect", "segment"})
        self.assertEqual(fresh_1, {"segment"})
        self.assertEqual(fresh_2, {"detect", "segment"})
        self.assertEqual(detect.calls, 2)
        self.assertEqual(segment.calls, 3)
        self.assertEqual(router.inference_counts, {"detect": 2, "segment": 3})

    def test_clear_cache_forces_first_inference_after_step_change(self) -> None:
        detect = _FakeBackend("detect")
        router = ModelRouter({"detect": detect}, default_interval=10)
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        router.infer(frame, {"type": "object_present"}, 0)
        router.clear_cache()
        _, fresh = router.infer(frame, {"type": "object_present"}, 1)

        self.assertEqual(fresh, {"detect"})
        self.assertEqual(detect.calls, 2)

    def test_project_rejects_invalid_task_interval(self) -> None:
        project = {
            "project_id": "router",
            "name": "router",
            "classes": [{"id": 0, "name": "item"}],
            "model_profiles": {
                "detect": {
                    "task": "detect",
                    "path": "models/detect.pt",
                    "inference_interval_frames": 0,
                }
            },
        }

        with self.assertRaisesRegex(ProjectValidationError, "inference_interval_frames"):
            validate_project(project)

    def test_factory_backends_are_loaded_only_when_task_is_used(self) -> None:
        created: list[str] = []

        def factory(task: str):
            def create() -> _FakeBackend:
                created.append(task)
                return _FakeBackend(task)

            return create

        router = ModelRouter(
            detector_factories={
                "detect": factory("detect"),
                "segment": factory("segment"),
            }
        )
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        self.assertEqual(router.loaded_tasks, set())
        router.infer(frame, {"type": "object_present"}, 0)
        self.assertEqual(created, ["detect"])
        self.assertEqual(router.loaded_tasks, {"detect"})
        router.infer(frame, {"type": "object_in_roi", "evidence": {"type": "mask"}}, 1)
        self.assertEqual(created, ["detect", "segment"])


if __name__ == "__main__":
    unittest.main()
