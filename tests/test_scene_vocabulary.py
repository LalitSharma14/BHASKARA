import json
import tempfile
import unittest
from pathlib import Path

from detection.scene_vocabulary import (
    dynamic_verification_candidates,
    canonical_scene_label,
    load_scene_proposals,
    merge_verified_scene_proposals,
    load_scene_vocabulary,
    merge_search_classes,
    normalize_scene_label,
    refined_scene_label,
    possible_scene_label,
    scene_label_requires_parent_context,
)
from tracking.sam2_tracking import label_candidates
from tracking.sam2_tracking import semantic_promotion_approved


class SceneVocabularyTests(unittest.TestCase):
    def test_long_dense_caption_is_not_used_as_detector_class(self):
        self.assertIsNone(normalize_scene_label(
            "painting of woman in yellow dress with red hair and yellow headscarf"
        ))

    def test_loads_short_unique_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            report.write_text(json.dumps({
                "candidate_vocabulary": ["Air conditioner", "mirror", "Mirror"]
            }), encoding="utf-8")
            self.assertEqual(
                load_scene_vocabulary([str(report)]),
                ["air conditioner", "mirror"],
            )

    def test_dynamic_labels_extend_instead_of_replace_defaults(self):
        self.assertEqual(
            merge_search_classes(["keys", "door"], ["mirror"]),
            ["keys", "door", "mirror"],
        )

    def test_dynamic_detector_phrase_can_be_resolved(self):
        self.assertEqual(label_candidates("a wall mirror", ["mirror"]), ["mirror"])

    def test_air_conditioner_is_verified_against_box_like_confusions(self):
        candidates = dynamic_verification_candidates("air conditioner")
        self.assertIn("air conditioner", candidates)
        self.assertIn("medicine box", candidates)

    def test_generic_appliance_can_refine_to_air_conditioner(self):
        candidates = dynamic_verification_candidates("home appliance")
        self.assertIn("air conditioner", candidates)
        self.assertIn("medicine box", candidates)
        self.assertEqual(
            refined_scene_label("home appliance", {"best_label": "air conditioner"}),
            "air conditioner",
        )

    def test_generic_appliance_is_not_stored_as_authoritative_label(self):
        self.assertIsNone(refined_scene_label(
            "home appliance", {"best_label": "home appliance"}
        ))

    def test_specific_scene_label_cannot_silently_change_class(self):
        self.assertIsNone(refined_scene_label(
            "mirror", {"best_label": "window"}
        ))

    def test_fan_part_description_normalizes_to_fan(self):
        self.assertEqual(canonical_scene_label("ceiling fan blades"), "fan")
        self.assertIn("ceiling", dynamic_verification_candidates("fan"))

    def test_unverified_fan_part_remains_possible_observation(self):
        self.assertEqual(
            possible_scene_label("fan", "ceiling fan blades"), "fan"
        )
        self.assertIsNone(possible_scene_label("window", "window"))

    def test_wheel_is_component_evidence_not_immediate_memory(self):
        self.assertTrue(scene_label_requires_parent_context("wheel"))
        self.assertEqual(possible_scene_label("wheel", "wheel"), "wheel")
        self.assertFalse(scene_label_requires_parent_context("suitcase"))

    def test_mirror_is_verified_against_window_and_painting(self):
        candidates = dynamic_verification_candidates("mirror")
        self.assertIn("window", candidates)
        self.assertIn("painting", candidates)

    def test_dense_ac_caption_uses_discovered_short_label(self):
        self.assertEqual(
            canonical_scene_label(
                "white ductless mini split air conditioner", ["air conditioner"]
            ),
            "air conditioner",
        )

    def test_verified_scene_box_replaces_conflicting_detector_guess(self):
        detections = [{"object": "medicine box", "box": (10, 10, 90, 50)}]
        proposals = [{
            "object": "air conditioner", "box": (8, 8, 92, 52), "confidence": 0.9
        }]
        merged = merge_verified_scene_proposals(detections, proposals)
        self.assertEqual([item["object"] for item in merged], ["air conditioner"])

    def test_verified_proposals_have_priority_over_remaining_detections(self):
        detections = [{"object": "door", "box": (0, 0, 20, 20)}]
        proposals = [{
            "object": "mirror", "box": (50, 50, 90, 90), "confidence": 0.9
        }]
        merged = merge_verified_scene_proposals(detections, proposals)
        self.assertEqual(merged[0]["object"], "mirror")

    def test_scene_proposals_retain_evidence_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            report.write_text(json.dumps({
                "candidate_vocabulary": ["air conditioner"],
                "frames": [{
                    "frame_index": 12,
                    "tasks": {"<OD>": {"parsed": {"<OD>": {
                        "bboxes": [[1, 2, 30, 40]],
                        "labels": ["air conditioner"],
                    }}}},
                }],
            }), encoding="utf-8")
            proposals = load_scene_proposals([str(report)])
            self.assertEqual(proposals[12][0]["object"], "air conditioner")

    def test_dual_model_landmark_can_clear_scene_gate(self):
        result = {
            "best_label": "picture frame",
            "best_score": 0.53,
            "margin": 0.13,
        }
        self.assertTrue(semantic_promotion_approved(
            "picture frame", result, minimum_score=0.50, minimum_margin=0.10
        ))


if __name__ == "__main__":
    unittest.main()
