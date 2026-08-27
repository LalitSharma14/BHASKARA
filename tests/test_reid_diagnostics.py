import unittest

from tracking.reconciliation import (
    find_lost_track_assignments,
    lost_geometry_is_compatible,
    similarity_bucket,
    track_age_bucket,
)


class ReidentificationDiagnosticTests(unittest.TestCase):
    @staticmethod
    def valid_box(box):
        return box[2] > box[0] and box[3] > box[1]

    def test_similarity_bands_have_stable_boundaries(self):
        self.assertEqual(similarity_bucket(0.69), "below_0.70")
        self.assertEqual(similarity_bucket(0.85), "0.85_to_0.90")
        self.assertEqual(similarity_bucket(0.92), "0.92_to_0.96")
        self.assertEqual(similarity_bucket(0.96), "0.96_and_above")

    def test_age_bands_have_stable_boundaries(self):
        self.assertEqual(track_age_bucket(30), "0_to_30")
        self.assertEqual(track_age_bucket(60), "31_to_60")
        self.assertEqual(track_age_bucket(120), "61_to_120")
        self.assertEqual(track_age_bucket(121), "121_and_above")

    def test_weak_candidate_records_label_age_tiny_and_edge(self):
        detections = [{
            "object": "keys",
            "box": (10, 10, 50, 50),
            "appearance_embedding": [1.0, 0.0],
            "appearance_tiny": True,
            "appearance_touches_edge": True,
        }]
        lost_tracks = [{
            "id": 4,
            "object": "keys",
            "box": (200, 200, 240, 240),
            "appearance_embedding": [0.8, 0.6],
            "lost_at_frame": 100,
        }]

        assignments, diagnostics = find_lost_track_assignments(
            detections,
            lost_tracks,
            1000,
            self.valid_box,
            return_diagnostics=True,
            current_frame_number=145,
        )

        self.assertEqual(assignments, {})
        self.assertEqual(diagnostics["below_threshold_by_label"], {"keys": 1})
        self.assertEqual(
            diagnostics["below_threshold_age_buckets"],
            {"31_to_60": 1},
        )
        self.assertEqual(diagnostics["below_threshold_tiny"], 1)
        self.assertEqual(diagnostics["below_threshold_edge"], 1)

    def test_recent_spatial_candidate_can_recover_at_point_85(self):
        detection = {
            "object": "bottle",
            "box": (100, 100, 160, 220),
            "appearance_embedding": [1.0, 0.0],
        }
        lost = [{
            "id": 8,
            "object": "bottle",
            "box": (102, 102, 162, 222),
            "appearance_embedding": [0.88, 0.475],
            "lost_at_frame": 100,
        }]
        assignments, diagnostics = find_lost_track_assignments(
            [detection], lost, 1000, self.valid_box,
            return_diagnostics=True, current_frame_number=115,
        )
        self.assertEqual(assignments[0][0]["id"], 8)
        self.assertEqual(diagnostics["recent_spatial_assigned"], 1)

    def test_recent_relaxation_does_not_apply_to_old_track(self):
        detection = {
            "object": "bottle",
            "box": (100, 100, 160, 220),
            "appearance_embedding": [1.0, 0.0],
        }
        lost = [{
            "id": 8,
            "object": "bottle",
            "box": (102, 102, 162, 222),
            "appearance_embedding": [0.88, 0.475],
            "lost_at_frame": 50,
        }]
        assignments = find_lost_track_assignments(
            [detection], lost, 1000, self.valid_box,
            current_frame_number=115,
        )
        self.assertEqual(assignments, {})

    def test_recent_relaxation_does_not_apply_without_spatial_evidence(self):
        detection = {
            "object": "bottle",
            "box": (100, 100, 160, 220),
            "appearance_embedding": [1.0, 0.0],
        }
        lost = [{
            "id": 8,
            "object": "bottle",
            "box": (500, 500, 560, 620),
            "appearance_embedding": [0.88, 0.475],
            "lost_at_frame": 100,
        }]
        assignments = find_lost_track_assignments(
            [detection], lost, 1000, self.valid_box,
            current_frame_number=115,
        )
        self.assertEqual(assignments, {})

    def test_recent_relaxation_does_not_apply_to_tiny_crop(self):
        detection = {
            "object": "keys",
            "box": (100, 100, 120, 120),
            "appearance_embedding": [1.0, 0.0],
            "appearance_tiny": True,
        }
        lost = [{
            "id": 8,
            "object": "keys",
            "box": (101, 101, 121, 121),
            "appearance_embedding": [0.88, 0.475],
            "lost_at_frame": 100,
        }]
        assignments = find_lost_track_assignments(
            [detection], lost, 1000, self.valid_box,
            current_frame_number=115,
        )
        self.assertEqual(assignments, {})

    def test_flexible_label_has_limited_shape_tolerance(self):
        detection = {"object": "clothes", "appearance_aspect_ratio": 0.2}
        track = {"appearance_aspect_ratio": 1.0}
        self.assertTrue(lost_geometry_is_compatible(detection, track))

        rigid_detection = {"object": "bottle", "appearance_aspect_ratio": 0.2}
        self.assertFalse(lost_geometry_is_compatible(rigid_detection, track))


if __name__ == "__main__":
    unittest.main()
