import unittest

from tracking.flow_lifecycle import prepare_track_for_lost_pool


class FlowLifecycleTests(unittest.TestCase):
    def test_flow_failure_preserves_identity_and_last_valid_box(self):
        track = {
            "id": 17,
            "object": "keys",
            "box": (10, 20, 50, 70),
            "points": ["old"],
            "confirmations": 6,
        }
        refreshed = ["new"]

        result = prepare_track_for_lost_pool(track, refreshed, 120)

        self.assertIs(result, track)
        self.assertEqual(result["id"], 17)
        self.assertEqual(result["box"], (10, 20, 50, 70))
        self.assertEqual(result["confirmations"], 6)
        self.assertIs(result["points"], refreshed)
        self.assertFalse(result["fresh_detection"])
        self.assertEqual(result["lost_at_frame"], 120)


if __name__ == "__main__":
    unittest.main()
