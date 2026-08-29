import unittest

import numpy as np

from collections import Counter

from tracking.sam2_tracking import (
    consensus_label,
    duplicates_existing_track,
    eligible_new_detection,
    label_candidates,
    mask_observation,
    mask_to_box,
    match_detections_to_tracks,
    tentative_track_action,
    semantic_promotion_candidates,
    semantic_promotion_approved,
    confirmed_track_action,
    requires_edge_safe_enrollment,
    touches_frame_edge,
)


class Sam2TrackingTests(unittest.TestCase):
    def test_mask_becomes_exclusive_xyxy_box(self):
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:6, 3:9] = True
        self.assertEqual(mask_to_box(mask), (3, 2, 9, 6))

    def test_tiny_mask_is_not_visible(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:3, 1:3] = True
        observation = mask_observation(4, 12, mask, minimum_area=8)
        self.assertFalse(observation.visible)
        self.assertEqual(observation.box, (0, 0, 0, 0))
        self.assertEqual(observation.area, 4)

    def test_batched_mask_shape_is_accepted(self):
        mask = np.zeros((1, 1, 8, 8), dtype=np.float32)
        mask[..., 2:6, 1:5] = 1
        self.assertEqual(mask_to_box(mask), (1, 2, 5, 6))

    def test_non_image_mask_is_rejected(self):
        with self.assertRaises(ValueError):
            mask_to_box(np.zeros((2, 3, 4)))

    def test_combined_detector_phrase_remains_explicitly_ambiguous(self):
        self.assertEqual(
            label_candidates("door cabinet shelf"),
            ["cabinet", "door", "shelf"],
        )

    def test_refresh_matching_is_unique(self):
        tracks = {1: (0, 0, 20, 20), 2: (30, 0, 50, 20)}
        detections = [
            {"box": (1, 1, 21, 21)},
            {"box": (31, 1, 51, 21)},
            {"box": (80, 80, 90, 90)},
        ]
        assignments, unmatched = match_detections_to_tracks(tracks, detections)
        self.assertEqual(assignments, {1: 0, 2: 1})
        self.assertEqual(unmatched, [2])

    def test_label_votes_change_without_changing_identity(self):
        votes = Counter({"desk": 2, "bed": 3})
        self.assertEqual(consensus_label(votes, newest_label="bed"), "bed")

    def test_new_track_enrollment_requires_quality(self):
        frame_size = (100, 100)
        self.assertTrue(eligible_new_detection(
            {"confidence": 0.7, "box": (10, 10, 40, 50)}, frame_size, 0.3
        ))
        self.assertFalse(eligible_new_detection(
            {"confidence": 0.2, "box": (10, 10, 40, 50)}, frame_size, 0.3
        ))
        self.assertFalse(eligible_new_detection(
            {"confidence": 0.7, "box": (0, 0, 95, 95)}, frame_size, 0.3
        ))

    def test_partial_same_label_detection_is_not_reenrolled(self):
        detection = {"object": "clothes", "box": (20, 20, 40, 70)}
        self.assertTrue(duplicates_existing_track(
            detection,
            {8: (10, 10, 60, 90)},
            {8: "clothes"},
        ))

    def test_contained_different_label_can_be_a_real_object(self):
        detection = {"object": "clothes", "box": (20, 20, 40, 70)}
        self.assertFalse(duplicates_existing_track(
            detection,
            {8: (0, 0, 100, 100)},
            {8: "door"},
        ))

    def test_separate_same_label_objects_can_both_be_enrolled(self):
        detection = {"object": "clothes", "box": (70, 10, 95, 80)}
        self.assertFalse(duplicates_existing_track(
            detection,
            {8: (10, 10, 45, 80)},
            {8: "clothes"},
        ))

    def test_tentative_track_promotes_on_second_confirmation(self):
        self.assertEqual(tentative_track_action(2, True, 0), "promote")

    def test_tentative_track_is_discarded_after_missed_refresh(self):
        self.assertEqual(tentative_track_action(1, False, 1), "discard")

    def test_tentative_track_does_not_promote_on_different_label(self):
        self.assertEqual(tentative_track_action(2, False, 1), "discard")

    def test_metal_ruler_promotion_checks_lanyard(self):
        candidates = semantic_promotion_candidates("metal ruler")
        self.assertIn("metal ruler", candidates)
        self.assertIn("lanyard", candidates)

    def test_low_risk_label_does_not_require_extra_model_call(self):
        self.assertEqual(semantic_promotion_candidates("clothes"), ())

    def test_semantic_promotion_requires_label_score_and_margin(self):
        good = {"best_label": "metal ruler", "best_score": 0.70, "margin": 0.30}
        wrong = {"best_label": "lanyard", "best_score": 0.70, "margin": 0.30}
        weak = {"best_label": "metal ruler", "best_score": 0.50, "margin": 0.30}
        ambiguous = {"best_label": "metal ruler", "best_score": 0.60, "margin": 0.10}
        self.assertTrue(semantic_promotion_approved("metal ruler", good))
        self.assertFalse(semantic_promotion_approved("metal ruler", wrong))
        self.assertFalse(semantic_promotion_approved("metal ruler", weak))
        self.assertFalse(semantic_promotion_approved("metal ruler", ambiguous))

    def test_confirmed_track_survives_short_detector_gap(self):
        self.assertEqual(confirmed_track_action(False, 2), "keep")

    def test_confirmed_track_retires_after_sustained_absence(self):
        self.assertEqual(confirmed_track_action(False, 3), "retire")

    def test_confirmed_match_resets_retirement_pressure(self):
        self.assertEqual(confirmed_track_action(True, 0), "keep")

    def test_id_card_is_edge_sensitive(self):
        self.assertTrue(requires_edge_safe_enrollment("id card"))
        self.assertFalse(requires_edge_safe_enrollment("fan"))

    def test_bottom_edge_crop_is_detected(self):
        self.assertTrue(touches_frame_edge((10, 90, 30, 100), (100, 100)))
        self.assertFalse(touches_frame_edge((10, 10, 30, 30), (100, 100)))


if __name__ == "__main__":
    unittest.main()
