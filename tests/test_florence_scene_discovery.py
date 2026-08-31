import unittest

import numpy as np

from experiments.florence_scene_discovery import (
    labels_from_florence_result,
    normalize_candidate_label,
    scene_difference,
)


class FlorenceSceneDiscoveryTests(unittest.TestCase):
    def test_normalizes_candidate_label(self):
        self.assertEqual(normalize_candidate_label("  Air   Conditioner. "), "air conditioner")

    def test_collects_labels_from_nested_task_result(self):
        result = {
            "<OD>": {
                "bboxes": [[0, 0, 10, 10]],
                "labels": ["Air conditioner", "Painting"],
            },
            "metadata": {"score": 0.8},
        }
        self.assertEqual(
            labels_from_florence_result(result),
            ["air conditioner", "painting"],
        )

    def test_deduplicates_repeated_labels(self):
        result = {"labels": ["mirror", "Mirror", "mirror."]}
        self.assertEqual(labels_from_florence_result(result), ["mirror"])

    def test_scene_difference_ignores_identical_frames(self):
        frame = np.full((32, 32, 3), (20, 80, 180), dtype=np.uint8)
        self.assertAlmostEqual(scene_difference(frame, frame.copy()), 0.0)

    def test_scene_difference_detects_changed_colour_composition(self):
        first = np.full((32, 32, 3), (20, 80, 180), dtype=np.uint8)
        second = np.full((32, 32, 3), (180, 20, 40), dtype=np.uint8)
        self.assertGreater(scene_difference(first, second), 0.32)


if __name__ == "__main__":
    unittest.main()
