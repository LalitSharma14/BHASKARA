import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from memory.identity_audit import IdentityAudit


class IdentityAuditTests(unittest.TestCase):
    def test_writes_crop_metadata_and_summary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            audit = IdentityAudit(temporary_directory, "test_run")
            crop = np.full((30, 40, 3), 127, dtype=np.uint8)

            identity_directory = audit.record_identity_event(
                memory_id=1,
                label="blue bottle",
                track_id=7,
                frame_number=100,
                confidence=0.8,
                crop=crop,
                event="created",
            )

            metadata = json.loads(
                (identity_directory / "identity.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["track_ids"], [7])
            self.assertTrue((identity_directory / metadata["events"][0]["crop"]).exists())

            summary_path = audit.write_summary(
                {
                    1: {
                        "object": "blue bottle",
                        "track_ids": [7],
                        "appearance_gallery": [[1.0, 0.0]],
                        "confidence": 0.8,
                        "confirmations": 5,
                        "last_seen_frame": 100,
                        "box": (1, 2, 3, 4),
                    }
                },
                {"recoveries": 1},
            )
            self.assertTrue(summary_path.exists())


if __name__ == "__main__":
    unittest.main()
