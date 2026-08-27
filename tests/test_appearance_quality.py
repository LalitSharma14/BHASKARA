import unittest

from tracking.appearance_quality import (
    describe_crop,
    required_identity_similarity,
)


class AppearanceQualityTests(unittest.TestCase):
    def test_tiny_crop_is_rejected_for_identity_embedding(self):
        description = describe_crop((10, 10, 18, 18), 1920, 1080)
        self.assertFalse(description["valid"])

    def test_edge_crop_is_marked_as_partial(self):
        description = describe_crop((0, 20, 100, 120), 640, 480)
        self.assertTrue(description["touches_edge"])
        self.assertLess(description["quality"], 1.0)

    def test_tiny_identity_requires_stronger_similarity(self):
        threshold = required_identity_similarity(
            {"appearance_tiny": True},
            0.92,
        )
        self.assertEqual(threshold, 0.96)


if __name__ == "__main__":
    unittest.main()
