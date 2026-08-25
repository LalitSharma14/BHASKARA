import unittest
import tempfile

from memory.identity_audit import IdentityAudit
from memory.object_memory import (
    clear_memory,
    get_memory,
    set_identity_audit,
    update_memory,
)


def make_track(track_id, embedding, box=(10, 10, 60, 80)):
    return {
        "id": track_id,
        "object": "bottle",
        "confidence": 0.8,
        "box": box,
        "confirmations": 8,
        "label_votes": {"bottle": 8},
        "appearance_embedding": embedding,
    }


class ObjectMemoryIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        set_identity_audit(
            IdentityAudit(self.temporary_directory.name, "memory_test")
        )
        clear_memory()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_same_track_updates_same_memory_identity(self):
        update_memory([make_track(10, [1.0, 0.0])], 100)
        update_memory([make_track(10, [0.99, 0.01])], 120)

        memory = get_memory()
        self.assertEqual(len(memory), 1)
        self.assertEqual(next(iter(memory.values()))["track_ids"], [10])

    def test_fragmented_track_can_reuse_appearance_identity(self):
        update_memory([make_track(10, [1.0, 0.0])], 100)
        update_memory([make_track(22, [0.99, 0.01])], 200)

        memory = get_memory()
        self.assertEqual(len(memory), 1)
        self.assertEqual(next(iter(memory.values()))["track_ids"], [10, 22])

    def test_visually_different_same_label_objects_remain_separate(self):
        update_memory([make_track(10, [1.0, 0.0])], 100)
        update_memory([make_track(22, [0.0, 1.0])], 200)

        self.assertEqual(len(get_memory()), 2)

    def test_memory_gallery_recognizes_an_alternate_trusted_view(self):
        first_track = make_track(10, [0.707, 0.707, 0.0])
        first_track["appearance_gallery"] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        update_memory([first_track], 100)

        alternate_view = make_track(22, [0.0, 0.999, 0.001])
        update_memory([alternate_view], 200)

        memory = get_memory()
        self.assertEqual(len(memory), 1)
        self.assertEqual(next(iter(memory.values()))["track_ids"], [10, 22])


if __name__ == "__main__":
    unittest.main()
