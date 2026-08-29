import unittest

from memory.semantic_gate import semantic_memory_decision


def charger_track(**updates):
    track = {
        "object": "charger",
        "semantic_verified_label": "charger",
        "semantic_score": 0.62,
        "semantic_margin": 0.24,
    }
    track.update(updates)
    return track


class SemanticMemoryGateTests(unittest.TestCase):
    def test_clear_semantic_winner_is_admitted_and_cached(self):
        track = charger_track()
        decision = semantic_memory_decision(track)
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["reason"], "verified")
        self.assertTrue(track["semantic_memory_approved"])

        cached = semantic_memory_decision(track)
        self.assertTrue(cached["accepted"])
        self.assertEqual(cached["reason"], "cached")

    def test_wall_socket_evidence_cannot_enter_charger_memory(self):
        decision = semantic_memory_decision(
            charger_track(semantic_verified_label="power socket")
        )
        self.assertFalse(decision["accepted"])

    def test_ambiguous_result_is_rejected(self):
        decision = semantic_memory_decision(
            charger_track(semantic_margin=0.04)
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "ambiguous")

    def test_low_score_is_rejected(self):
        decision = semantic_memory_decision(
            charger_track(semantic_score=0.32)
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "low_score")

    def test_new_conflicting_evidence_invalidates_cached_approval(self):
        track = charger_track(
            semantic_memory_approved=True,
            semantic_memory_label="charger",
            semantic_verified_label="usb cable",
        )
        decision = semantic_memory_decision(track)
        self.assertFalse(decision["accepted"])

    def test_low_risk_label_uses_existing_memory_gate(self):
        decision = semantic_memory_decision({"object": "bottle"})
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["reason"], "not_required")


if __name__ == "__main__":
    unittest.main()
