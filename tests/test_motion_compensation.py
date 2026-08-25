import unittest

import cv2
import numpy as np

from tracking.motion_compensation import transport_box_with_optical_flow


class MotionCompensationTests(unittest.TestCase):
    def test_transports_textured_box_with_frame_translation(self):
        source = np.zeros((240, 320), dtype=np.uint8)
        box = (80, 60, 220, 180)

        random_generator = np.random.default_rng(42)
        source[60:180, 80:220] = random_generator.integers(
            0,
            256,
            size=(120, 140),
            dtype=np.uint8,
        )

        transform = np.float32([[1, 0, 14], [0, 1, 9]])
        current = cv2.warpAffine(source, transform, (320, 240))

        transported, diagnostics = transport_box_with_optical_flow(
            source,
            current,
            box,
        )

        self.assertTrue(diagnostics["success"])
        self.assertAlmostEqual(transported[0], box[0] + 14, delta=1)
        self.assertAlmostEqual(transported[1], box[1] + 9, delta=1)
        self.assertAlmostEqual(transported[2], box[2] + 14, delta=1)
        self.assertAlmostEqual(transported[3], box[3] + 9, delta=1)

    def test_textureless_box_falls_back_safely(self):
        source = np.zeros((100, 100), dtype=np.uint8)
        current = source.copy()
        box = (20, 20, 80, 80)

        transported, diagnostics = transport_box_with_optical_flow(
            source,
            current,
            box,
        )

        self.assertFalse(diagnostics["success"])
        self.assertEqual(transported, box)


if __name__ == "__main__":
    unittest.main()
