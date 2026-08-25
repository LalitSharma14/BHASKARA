"""Visual evidence writer for BHASKARA physical-memory identities."""

import json
from datetime import datetime
from pathlib import Path

import cv2


def _safe_label(label):
    return "".join(
        character if character.isalnum() else "_"
        for character in label.lower()
    ).strip("_") or "object"


class IdentityAudit:
    """Write reviewable crops and JSON metadata for identity decisions."""

    def __init__(self, root="runs/identity_audit", run_name=None):
        if run_name is None:
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        self.run_directory = Path(root) / run_name
        self.run_directory.mkdir(parents=True, exist_ok=True)

    def record_identity_event(
        self,
        memory_id,
        label,
        track_id,
        frame_number,
        confidence,
        crop,
        event,
        similarity=None,
        second_best_similarity=None,
        margin=None,
    ):
        identity_directory = self.run_directory / (
            f"memory_{memory_id:03d}_{_safe_label(label)}"
        )
        identity_directory.mkdir(parents=True, exist_ok=True)

        crop_filename = None
        if crop is not None and getattr(crop, "size", 0) > 0:
            crop_filename = f"track_{track_id:04d}_frame_{frame_number:06d}.jpg"
            cv2.imwrite(str(identity_directory / crop_filename), crop)

        metadata_path = identity_directory / "identity.json"
        metadata = {
            "memory_id": memory_id,
            "label": label,
            "track_ids": [],
            "events": [],
        }

        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if track_id not in metadata["track_ids"]:
            metadata["track_ids"].append(track_id)
            metadata["track_ids"].sort()

        metadata["events"].append({
            "event": event,
            "track_id": track_id,
            "frame_number": frame_number,
            "confidence": confidence,
            "similarity": similarity,
            "second_best_similarity": second_best_similarity,
            "margin": margin,
            "crop": crop_filename,
        })

        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        return identity_directory

    def write_summary(self, memory, diagnostics):
        identities = []

        for memory_id, data in memory.items():
            identities.append({
                "memory_id": memory_id,
                "object": data["object"],
                "track_ids": data.get("track_ids", []),
                "appearance_views": len(data.get("appearance_gallery", [])),
                "confidence": data["confidence"],
                "confirmations": data["confirmations"],
                "last_seen_frame": data["last_seen_frame"],
                "box": list(data["box"]),
            })

        summary = {
            "created_at": datetime.now().isoformat(),
            "identity_count": len(identities),
            "identities": identities,
            "diagnostics": diagnostics,
        }

        summary_path = self.run_directory / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary_path
