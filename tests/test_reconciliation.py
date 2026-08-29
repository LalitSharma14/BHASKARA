import unittest

from tracking.reconciliation import (
    are_labels_compatible,
    best_appearance_similarity,
    calculate_containment,
    cosine_similarity,
    calculate_iou,
    calculate_match_score,
    diagnose_new_track_reason,
    find_global_assignments,
    find_lost_track_assignments,
    select_consensus_label,
    update_appearance_prototype,
    update_appearance_gallery,
    identity_appearance_similarity,
)
from tracking.appearance_quality import geometry_is_compatible


FRAME_DIAGONAL_1080P = (1920 ** 2 + 1080 ** 2) ** 0.5
MATCH_THRESHOLD = 0.30


class ReconciliationBaselineTests(unittest.TestCase):
    def test_iou_for_identical_boxes_is_one(self):
        box = (100, 100, 300, 300)
        self.assertEqual(calculate_iou(box, box), 1.0)

    def test_non_overlapping_boxes_have_no_overlap(self):
        box1 = (0, 0, 50, 50)
        box2 = (100, 100, 150, 150)
        self.assertEqual(calculate_iou(box1, box2), 0.0)
        self.assertEqual(calculate_containment(box1, box2), 0.0)

    def test_partial_desk_to_bed_can_retain_track(self):
        partial_desk = (300, 300, 500, 500)
        fuller_bed = (250, 250, 650, 650)

        score = calculate_match_score(
            fuller_bed,
            partial_desk,
            "bed",
            "desk",
            FRAME_DIAGONAL_1080P,
        )

        self.assertGreaterEqual(score, MATCH_THRESHOLD)

    def test_desk_and_bed_are_an_explicit_compatible_transition(self):
        self.assertTrue(are_labels_compatible("desk", "bed"))

    def test_unrelated_labels_are_not_compatible(self):
        self.assertFalse(are_labels_compatible("clothes", "door"))
        self.assertFalse(are_labels_compatible("fan", "metal ruler"))
        self.assertFalse(are_labels_compatible("charger", "usb cable"))

    def test_same_label_with_moderate_overlap_can_retain_track(self):
        old_box = (100, 100, 300, 300)
        new_box = (150, 100, 350, 300)

        score = calculate_match_score(
            new_box,
            old_box,
            "keys",
            "keys",
            FRAME_DIAGONAL_1080P,
        )

        self.assertGreaterEqual(score, MATCH_THRESHOLD)

    def test_newest_label_wins_an_exact_vote_tie(self):
        votes = {"desk": 2, "bed": 2}
        self.assertEqual(select_consensus_label(votes, "bed"), "bed")

    def test_majority_still_beats_newest_label(self):
        votes = {"desk": 2, "bed": 5}
        self.assertEqual(select_consensus_label(votes, "desk"), "bed")


class ReconciliationSafetyTests(unittest.TestCase):
    def test_unrelated_contained_object_must_not_merge(self):
        door = (100, 100, 700, 1000)
        clothes = (250, 350, 450, 700)

        score = calculate_match_score(
            clothes,
            door,
            "clothes",
            "door",
            FRAME_DIAGONAL_1080P,
        )

        self.assertLess(score, MATCH_THRESHOLD)

    def test_nearby_unrelated_objects_must_not_merge(self):
        old_fan_box = (500, 400, 700, 600)
        new_ruler_box = (525, 425, 675, 575)

        score = calculate_match_score(
            new_ruler_box,
            old_fan_box,
            "metal ruler",
            "fan",
            FRAME_DIAGONAL_1080P,
        )

        self.assertLess(score, MATCH_THRESHOLD)

    def test_identical_geometry_does_not_override_incompatible_labels(self):
        box = (100, 100, 400, 400)

        score = calculate_match_score(
            box,
            box,
            "keys",
            "usb cable",
            FRAME_DIAGONAL_1080P,
        )

        self.assertEqual(score, 0.0)


class GlobalAssignmentTests(unittest.TestCase):
    @staticmethod
    def valid_box(box):
        return box[2] > box[0] and box[3] > box[1]

    def test_each_track_can_be_assigned_only_once(self):
        tracks = [{"id": 1, "object": "bottle", "box": (100, 100, 200, 300)}]
        detections = [
            {"object": "bottle", "box": (102, 102, 202, 302)},
            {"object": "bottle", "box": (120, 110, 220, 310)},
        ]

        assignments = find_global_assignments(
            detections,
            tracks,
            FRAME_DIAGONAL_1080P,
            self.valid_box,
        )

        self.assertEqual(len(assignments), 1)
        self.assertIn(0, assignments)

    def test_assignment_chooses_strongest_pair_not_first_pair(self):
        tracks = [{"id": 1, "object": "bottle", "box": (100, 100, 200, 300)}]
        weaker_detection = {"object": "bottle", "box": (135, 100, 235, 300)}
        stronger_detection = {"object": "bottle", "box": (102, 100, 202, 300)}

        assignments = find_global_assignments(
            [weaker_detection, stronger_detection],
            tracks,
            FRAME_DIAGONAL_1080P,
            self.valid_box,
        )

        self.assertNotIn(0, assignments)
        self.assertIn(1, assignments)

    def test_assignment_is_stable_when_detection_order_changes(self):
        tracks = [
            {"id": 1, "object": "bottle", "box": (50, 50, 100, 150)},
            {"id": 2, "object": "bottle", "box": (200, 50, 250, 150)},
        ]
        detections = [
            {"object": "bottle", "box": (202, 52, 252, 152)},
            {"object": "bottle", "box": (52, 52, 102, 152)},
        ]

        forward = find_global_assignments(
            detections,
            tracks,
            FRAME_DIAGONAL_1080P,
            self.valid_box,
        )
        reverse = find_global_assignments(
            list(reversed(detections)),
            tracks,
            FRAME_DIAGONAL_1080P,
            self.valid_box,
        )

        forward_ids = {
            detections[index]["box"]: track["id"]
            for index, (track, _score) in forward.items()
        }
        reversed_detections = list(reversed(detections))
        reverse_ids = {
            reversed_detections[index]["box"]: track["id"]
            for index, (track, _score) in reverse.items()
        }
        self.assertEqual(forward_ids, reverse_ids)

    def test_recent_exact_label_track_can_be_recovered(self):
        lost_tracks = [
            {
                "id": 7,
                "object": "bottle",
                "box": (100, 100, 160, 240),
                "appearance_embedding": [1.0, 0.0, 0.0],
            }
        ]
        detections = [
            {
                "object": "bottle",
                "box": (105, 102, 165, 242),
                "appearance_embedding": [0.99, 0.01, 0.0],
            }
        ]

        assignments = find_lost_track_assignments(
            detections,
            lost_tracks,
            FRAME_DIAGONAL_1080P,
            self.valid_box,
        )

        self.assertEqual(assignments[0][0]["id"], 7)

    def test_lost_track_recovery_rejects_different_label(self):
        lost_tracks = [
            {
                "id": 7,
                "object": "bottle",
                "box": (100, 100, 160, 240),
                "appearance_embedding": [1.0, 0.0, 0.0],
            }
        ]
        detections = [
            {
                "object": "cup",
                "box": (100, 100, 160, 240),
                "appearance_embedding": [1.0, 0.0, 0.0],
            }
        ]

        assignments = find_lost_track_assignments(
            detections,
            lost_tracks,
            FRAME_DIAGONAL_1080P,
            self.valid_box,
        )

        self.assertEqual(assignments, {})

    def test_ambiguous_lost_identity_is_rejected(self):
        lost_tracks = [
            {
                "id": 1,
                "object": "bottle",
                "box": (100, 100, 160, 240),
                "appearance_embedding": [1.0, 0.0, 0.0],
            },
            {
                "id": 2,
                "object": "bottle",
                "box": (300, 100, 360, 240),
                "appearance_embedding": [0.999, 0.001, 0.0],
            },
        ]
        detections = [
            {
                "object": "bottle",
                "box": (200, 100, 260, 240),
                "appearance_embedding": [1.0, 0.0, 0.0],
            }
        ]

        assignments = find_lost_track_assignments(
            detections,
            lost_tracks,
            FRAME_DIAGONAL_1080P,
            self.valid_box,
        )
        self.assertEqual(assignments, {})


class AppearanceTests(unittest.TestCase):
    def test_context_cannot_rescue_different_object_crop(self):
        observation = {
            "appearance_embedding": [0.0, 1.0],
            "context_embedding": [1.0, 0.0],
        }
        identity = {
            "appearance_embedding": [1.0, 0.0],
            "appearance_gallery": [],
            "context_embedding": [1.0, 0.0],
            "context_gallery": [],
        }
        similarity = identity_appearance_similarity(observation, identity)
        self.assertLess(similarity, 0.20)

    def test_changed_context_does_not_penalize_strong_object_match(self):
        observation = {
            "appearance_embedding": [1.0, 0.0, 0.0],
            "context_embedding": [0.0, 1.0, 0.0],
        }
        identity = {
            "appearance_embedding": [0.999, 0.001, 0.0],
            "appearance_gallery": [],
            "context_embedding": [0.0, 0.0, 1.0],
            "context_gallery": [],
        }
        tight_similarity = best_appearance_similarity(
            observation["appearance_embedding"],
            identity,
        )
        combined_similarity = identity_appearance_similarity(
            observation,
            identity,
        )
        self.assertAlmostEqual(combined_similarity, tight_similarity)

    def test_context_can_only_add_a_small_supporting_boost(self):
        observation = {
            "appearance_embedding": [0.8, 0.6],
            "context_embedding": [1.0, 0.0],
        }
        identity = {
            "appearance_embedding": [1.0, 0.0],
            "appearance_gallery": [],
            "context_embedding": [1.0, 0.0],
            "context_gallery": [],
        }
        similarity = identity_appearance_similarity(observation, identity)
        self.assertAlmostEqual(similarity, 0.81, places=5)

    def test_severe_aspect_change_is_not_same_identity(self):
        observation = {"appearance_aspect_ratio": 0.2}
        identity = {"appearance_aspect_ratio": 2.0}
        self.assertFalse(geometry_is_compatible(observation, identity))


class FragmentationDiagnosticTests(unittest.TestCase):
    @staticmethod
    def valid_box(box):
        return box[2] > box[0] and box[3] > box[1]

    def test_reports_no_compatible_active_track(self):
        detection = {"object": "bottle", "box": (10, 10, 60, 100)}
        tracks = [{"id": 1, "object": "door", "box": (10, 10, 60, 100)}]
        reason = diagnose_new_track_reason(
            detection, tracks, [], FRAME_DIAGONAL_1080P, self.valid_box
        )
        self.assertEqual(reason, "no_compatible_active_track")

    def test_reports_active_spatial_gate(self):
        detection = {"object": "bottle", "box": (10, 10, 60, 100)}
        tracks = [{"id": 1, "object": "bottle", "box": (500, 500, 550, 590)}]
        reason = diagnose_new_track_reason(
            detection, tracks, [], FRAME_DIAGONAL_1080P, self.valid_box
        )
        self.assertEqual(reason, "active_spatial_gate")

    def test_reports_active_appearance_gate(self):
        detection = {
            "object": "bottle",
            "box": (10, 10, 60, 100),
            "appearance_embedding": [1.0, 0.0],
        }
        tracks = [{
            "id": 1,
            "object": "bottle",
            "box": (12, 12, 62, 102),
            "appearance_embedding": [0.0, 1.0],
        }]
        reason = diagnose_new_track_reason(
            detection, tracks, [], FRAME_DIAGONAL_1080P, self.valid_box
        )
        self.assertEqual(reason, "active_appearance_gate")

    def test_reports_assignment_conflict_for_an_eligible_pair(self):
        detection = {
            "object": "bottle",
            "box": (10, 10, 60, 100),
            "appearance_embedding": [1.0, 0.0],
        }
        tracks = [{
            "id": 1,
            "object": "bottle",
            "box": (12, 12, 62, 102),
            "appearance_embedding": [1.0, 0.0],
        }]
        reason = diagnose_new_track_reason(
            detection, tracks, [], FRAME_DIAGONAL_1080P, self.valid_box
        )
        self.assertEqual(reason, "active_assignment_conflict")

    def test_reports_weak_lost_appearance(self):
        detection = {
            "object": "bottle",
            "box": (10, 10, 60, 100),
            "appearance_embedding": [1.0, 0.0],
        }
        lost = [{
            "id": 7,
            "object": "bottle",
            "box": (300, 300, 350, 390),
            "appearance_embedding": [0.8, 0.6],
        }]
        reason = diagnose_new_track_reason(
            detection, [], lost, FRAME_DIAGONAL_1080P, self.valid_box
        )
        self.assertEqual(reason, "lost_appearance_gate")

    def test_cosine_similarity_distinguishes_identity(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_prototype_update_remains_normalized(self):
        prototype = update_appearance_prototype([1, 0], [0.8, 0.2])
        self.assertAlmostEqual(float((prototype ** 2).sum()), 1.0, places=5)

    def test_gallery_keeps_diverse_views(self):
        gallery = update_appearance_gallery([], [1.0, 0.0])
        gallery = update_appearance_gallery(gallery, [0.0, 1.0])
        self.assertEqual(len(gallery), 2)

    def test_gallery_rejects_near_duplicate_view(self):
        gallery = update_appearance_gallery([], [1.0, 0.0])
        gallery = update_appearance_gallery(gallery, [0.999, 0.001])
        self.assertEqual(len(gallery), 1)

    def test_best_similarity_uses_any_gallery_view(self):
        identity = {
            "appearance_embedding": [1.0, 0.0, 0.0],
            "appearance_gallery": [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        }
        similarity = best_appearance_similarity([0.0, 1.0, 0.0], identity)
        self.assertAlmostEqual(similarity, 1.0)

    def test_compatible_labels_still_need_spatial_evidence(self):
        old_desk = (100, 100, 300, 300)
        distant_bed = (1200, 600, 1700, 1000)

        score = calculate_match_score(
            distant_bed,
            old_desk,
            "bed",
            "desk",
            FRAME_DIAGONAL_1080P,
        )

        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
