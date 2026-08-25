"""Trusted physical-object memory for BHASKARA."""

from datetime import datetime

from memory.identity_audit import IdentityAudit
from tracking.reconciliation import (
    best_appearance_similarity,
    update_appearance_gallery,
    update_appearance_prototype,
)


object_memory = {}
track_to_memory_id = {}
next_memory_id = 1
memory_stats = {
    "appearance_consolidations": 0,
    "new_memory_identities": 0,
}
identity_audit = IdentityAudit()


def _find_appearance_identity(track, minimum_similarity=0.94, minimum_margin=0.03):
    """Return an unambiguous same-label memory identity, if one exists."""

    track_embedding = track.get("appearance_embedding")
    if track_embedding is None:
        return None

    candidates = []

    for memory_id, data in object_memory.items():
        if data["object"] != track["object"]:
            continue

        similarity = best_appearance_similarity(
            track_embedding,
            data,
        )

        if similarity is not None and similarity >= minimum_similarity:
            candidates.append((similarity, memory_id))

    candidates.sort(reverse=True)
    if not candidates:
        return None

    if len(candidates) > 1:
        if candidates[0][0] - candidates[1][0] < minimum_margin:
            return None

    best_similarity, memory_id = candidates[0]
    second_best_similarity = candidates[1][0] if len(candidates) > 1 else None

    return {
        "memory_id": memory_id,
        "similarity": best_similarity,
        "second_best_similarity": second_best_similarity,
        "margin": (
            best_similarity - second_best_similarity
            if second_best_similarity is not None
            else None
        ),
    }


def _resolve_memory_id(track):
    """Resolve a transient track to an existing or new permanent identity."""

    global next_memory_id

    track_id = track["id"]
    existing_memory_id = track_to_memory_id.get(track_id)
    if existing_memory_id is not None:
        return existing_memory_id, {"event": "updated"}

    appearance_identity = _find_appearance_identity(track)
    if appearance_identity is not None:
        appearance_memory_id = appearance_identity["memory_id"]
        track_to_memory_id[track_id] = appearance_memory_id
        memory_stats["appearance_consolidations"] += 1
        return appearance_memory_id, {
            "event": "consolidated",
            "similarity": appearance_identity["similarity"],
            "second_best_similarity": appearance_identity[
                "second_best_similarity"
            ],
            "margin": appearance_identity["margin"],
        }

    memory_id = next_memory_id
    next_memory_id += 1
    track_to_memory_id[track_id] = memory_id
    memory_stats["new_memory_identities"] += 1
    return memory_id, {"event": "created"}


def update_memory(trusted_tracks, frame_number):
    """Update permanent memory using tracks that passed the trust gate."""

    current_time = datetime.now()

    for track in trusted_tracks:
        x1, y1, x2, y2 = track["box"]
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue

        memory_id, identity_event = _resolve_memory_id(track)
        previous = object_memory.get(memory_id, {})
        track_ids = set(previous.get("track_ids", []))
        track_ids.add(track["id"])

        appearance_embedding = update_appearance_prototype(
            previous.get("appearance_embedding"),
            track.get("appearance_embedding"),
            observation_weight=0.05,
        )

        appearance_gallery = list(previous.get("appearance_gallery", []))
        track_prototypes = list(track.get("appearance_gallery", []) or [])

        if not track_prototypes and track.get("appearance_embedding") is not None:
            track_prototypes = [track["appearance_embedding"]]

        for prototype in track_prototypes:
            appearance_gallery = update_appearance_gallery(
                appearance_gallery,
                prototype,
                maximum_size=12,
                diversity_threshold=0.985,
            )

        object_memory[memory_id] = {
            "memory_id": memory_id,
            "object": track["object"],
            "confidence": track["confidence"],
            "box": track["box"],
            "confirmations": track.get("confirmations", 0),
            "label_votes": track.get("label_votes", {}).copy(),
            "appearance_embedding": appearance_embedding,
            "appearance_gallery": appearance_gallery,
            "track_ids": sorted(track_ids),
            "last_seen_frame": frame_number,
            "last_seen_time": current_time,
        }

        if identity_event["event"] != "updated":
            identity_audit.record_identity_event(
                memory_id=memory_id,
                label=track["object"],
                track_id=track["id"],
                frame_number=track.get("evidence_frame", frame_number),
                confidence=track.get(
                    "evidence_confidence",
                    track["confidence"],
                ),
                crop=track.get("evidence_crop"),
                event=identity_event["event"],
                similarity=identity_event.get("similarity"),
                second_best_similarity=identity_event.get(
                    "second_best_similarity"
                ),
                margin=identity_event.get("margin"),
            )


def get_memory():
    return object_memory


def get_memory_stats():
    return memory_stats.copy()


def get_audit_run_directory():
    return identity_audit.run_directory


def set_identity_audit(audit):
    global identity_audit
    identity_audit = audit


def write_audit_summary(diagnostics):
    return identity_audit.write_summary(object_memory, diagnostics)


def find_object(object_name):
    return [
        data.copy()
        for data in object_memory.values()
        if data["object"].lower() == object_name.lower()
    ]


def get_last_seen(object_name):
    matches = find_object(object_name)
    if not matches:
        return None
    return max(matches, key=lambda item: item["last_seen_time"])


def clear_memory():
    global next_memory_id

    object_memory.clear()
    track_to_memory_id.clear()
    next_memory_id = 1
    memory_stats["appearance_consolidations"] = 0
    memory_stats["new_memory_identities"] = 0
