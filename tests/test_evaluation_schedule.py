import unittest

from tracking.evaluation_schedule import (
    scheduled_evaluation_frames,
    should_schedule_evaluation_frame,
)


class EvaluationScheduleTests(unittest.TestCase):
    def test_schedule_is_fixed_by_frame_number(self):
        self.assertEqual(scheduled_evaluation_frames(61, 15), [15, 30, 45, 60])

    def test_final_frame_is_not_scheduled(self):
        self.assertFalse(should_schedule_evaluation_frame(60, 15, 60))

    def test_unknown_length_still_uses_interval(self):
        self.assertTrue(should_schedule_evaluation_frame(30, 15))

    def test_invalid_interval_is_rejected(self):
        self.assertFalse(should_schedule_evaluation_frame(30, 0, 100))


if __name__ == "__main__":
    unittest.main()
