import unittest

from tracking.detection_timing import (
    calculate_result_age,
    select_reconciled_box,
    should_accept_result,
)


class DetectionTimingTests(unittest.TestCase):
    def test_result_age_uses_source_and_current_frames(self):
        self.assertEqual(calculate_result_age(125, 100), 25)

    def test_result_age_cannot_be_negative(self):
        self.assertEqual(calculate_result_age(100, 125), 0)

    def test_fresh_and_moderately_delayed_results_are_accepted(self):
        self.assertTrue(should_accept_result(0, 45))
        self.assertTrue(should_accept_result(30, 45))
        self.assertTrue(should_accept_result(45, 45))

    def test_excessively_stale_result_is_rejected(self):
        self.assertFalse(should_accept_result(46, 45))

    def test_fresh_result_uses_detector_box(self):
        detector_box = (100, 100, 200, 200)
        current_track_box = (110, 100, 210, 200)
        self.assertEqual(
            select_reconciled_box(detector_box, current_track_box, 0),
            detector_box,
        )

    def test_delayed_result_preserves_current_track_box(self):
        stale_detector_box = (100, 100, 200, 200)
        current_track_box = (140, 100, 240, 200)
        self.assertEqual(
            select_reconciled_box(stale_detector_box, current_track_box, 12),
            current_track_box,
        )


if __name__ == "__main__":
    unittest.main()
